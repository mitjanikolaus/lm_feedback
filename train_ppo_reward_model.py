import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Callable, List, Dict, Tuple, Any

import torch
from accelerate.utils import gather_object
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import pandas as pd
from transformers.trainer_pt_utils import nested_detach
from trl.trainer.utils import print_rich_table

from data import compute_reward_value
from utils import CHILDES_RL_DATA_FILE

tqdm.pandas()

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification, PreTrainedModel, \
    PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset, DatasetDict

from trl import RewardConfig, ModelConfig, \
    get_quantization_config, get_kbit_device_map, RewardTrainer, get_peft_config


os.environ["WANDB_PROJECT"] = "lm_feedback_reward_model"
os.environ["WANDB_LOG_MODEL"] = "false"

TEST_SET_SIZE = 0.1
SPLIT_RANDOM_STATE = 1


def compute_mse(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    mse = ((predictions.squeeze() - labels.squeeze())**2).mean().item()

    return {"mse": mse}


@dataclass
class CFRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples_batched = []
        for sample in samples:
            samples_batched.append(
                {
                    "input_ids": sample["input_ids"],
                    "attention_mask": sample["attention_mask"],
                }
            )
        batch = self.tokenizer.pad(
            samples_batched,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        ).data

        batch["return_loss"] = True
        batch["reward"] = torch.tensor([sample["reward"] for sample in samples], dtype=torch.float)
        return batch


class CFRewardTrainer(RewardTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        data_collator = CFRewardDataCollatorWithPadding(tokenizer, max_length=max_length)
        compute_metrics = compute_mse
        super(CFRewardTrainer, self).__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks,
            optimizers, preprocess_logits_for_metrics, max_length, peft_config
        )

    def evaluate(self, *args, **kwargs):
        # num_print_samples = kwargs.pop("num_print_samples", 10)
        # self.visualize_samples(num_print_samples)
        return super(RewardTrainer, self).evaluate(*args, **kwargs)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, output_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()

        logits = output_dict["logits"]
        logits = nested_detach(logits)

        labels = inputs["reward"]

        return loss, logits, labels

    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `10`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            text = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            table["text"].extend(gather_object(text))
            table["reward"].extend(gather_object(inputs["reward"].cpu()))
            table["logits"].extend(gather_object(logits.squeeze().cpu()))
            if num_print_samples >= 0 and len(table["text"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({f"completions": wandb.Table(dataframe=df)})

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )["logits"]
        if "margin" in inputs:
            raise NotImplementedError()
        else:
            loss = nn.functional.mse_loss(logits.squeeze(), target=inputs["reward"])

        if return_outputs:
            return loss, {
                "logits": logits,
            }
        return loss


def build_reward_model_trainer_datasets(fb_data_path=CHILDES_RL_DATA_FILE):
    data_fb = pd.read_csv(fb_data_path)
    data_fb["reward"] = data_fb.apply(compute_reward_value, axis=1)
    del data_fb["response_is_clarification_request"]
    del data_fb["response_is_acknowledgement"]

    data_train, data_test = train_test_split(data_fb, test_size=TEST_SET_SIZE, shuffle=True,
                                            random_state=SPLIT_RANDOM_STATE)

    ds_train = Dataset.from_pandas(data_train)
    ds_test = Dataset.from_pandas(data_test)

    ds_train.set_format(type="torch")
    ds_test.set_format(type="torch")

    datasets = DatasetDict({"train": ds_train, "test": ds_test})
    return datasets


def main():
    trainer_config_args = RewardConfig
    reward_config_fields = trainer_config_args.__dataclass_fields__
    reward_config_fields["bf16"].default = True
    reward_config_fields["per_device_train_batch_size"].default = 64
    reward_config_fields["per_device_eval_batch_size"].default = 1024
    reward_config_fields["logging_steps"].default = 10
    reward_config_fields["num_train_epochs"].default = 1
    reward_config_fields["learning_rate"].default = 1.41e-5
    reward_config_fields["optim"].default = "adamw_torch"
    reward_config_fields["max_length"].default = 128
    reward_config_fields["remove_unused_columns"].default = False
    reward_config_fields["load_best_model_at_end"].default = True
    reward_config_fields["metric_for_best_model"].default = "mse"
    reward_config_fields["greater_is_better"].default = False
    reward_config_fields["save_total_limit"].default = 1
    reward_config_fields["save_steps"].default = 50
    reward_config_fields["evaluation_strategy"].default = "steps"
    reward_config_fields["eval_steps"].default = 50
    reward_config_fields["eval_on_start"].default = True

    model_config_args = ModelConfig
    model_config_fields = model_config_args.__dataclass_fields__
    model_config_fields["lora_task_type"].default = "SEQ_CLS"

    parser = HfArgumentParser((trainer_config_args, model_config_args))

    trainer_config, model_config = parser.parse_args_into_dataclasses()
    trainer_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    raw_datasets = build_reward_model_trainer_datasets()

    def preprocess_function(sample):
        tokenized = tokenizer(sample["utt_transcript_clean"], truncation=True, max_length=trainer_config.max_length)
        tokenized["reward"] = sample["reward"]

        return tokenized

    # Preprocess the dataset and truncate examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = CFRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    trainer._load_best_model()

    trainer.model.visualize_samples(100)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
