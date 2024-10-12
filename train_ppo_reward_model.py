import glob
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, List, Dict, Tuple, Any

import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import nested_detach
from trl.trainer.utils import print_rich_table

from data import compute_reward_value
from utilities import CONVERSATIONS_DATA_FILE

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification, PreTrainedModel, \
    PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset, DatasetDict

from trl import RewardConfig, ModelConfig, \
    get_quantization_config, get_kbit_device_map, RewardTrainer, get_peft_config

os.environ["WANDB_LOG_MODEL"] = "false"

TEST_SET_SIZE = 0.1
SPLIT_RANDOM_STATE = 1


def compute_mse(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    mse = ((predictions.squeeze() - labels.squeeze()) ** 2).mean().item()

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


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=50):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 50.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.global_step % self.freq == 0:
            predictions = self.trainer.predict(self.sample_dataset)

            table = defaultdict(list)
            table["text"] = self.sample_dataset[
                "transcript_clean"]  # self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            table["reward"] = self.sample_dataset["reward"].cpu()
            table["logits"] = predictions.predictions.squeeze()

            df = pd.DataFrame(table)
            print_rich_table(df[:10])
            records_table = self._wandb.Table(dataframe=df)
            self._wandb.log({"sample_predictions": records_table})


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
        outputs = F.sigmoid(logits.squeeze())
        outputs = nested_detach(outputs)

        labels = inputs["reward"]

        return loss, outputs, labels

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
            outputs = F.sigmoid(logits.squeeze())
            loss = nn.functional.mse_loss(outputs, target=inputs["reward"])

        if return_outputs:
            return loss, {
                "logits": logits,
            }
        return loss


def build_reward_model_trainer_datasets(fb_data_paths, reward_cr, reward_ack, reward_other):
    all_data = []
    for fb_data_path in fb_data_paths:
        if os.path.isfile(fb_data_path):
            print(f"Loading {fb_data_path}")
            if fb_data_path.endswith(".csv"):
                data = pd.read_csv(fb_data_path)
            elif fb_data_path.endswith(".txt"):
                with open(fb_data_path, "r") as f:
                    data = f.read().splitlines()
                data = pd.DataFrame({"transcript_clean": data, "reward": [1] * len(data)})
            else:
                raise RuntimeError(f"Unknown data format: {fb_data_path}")
        else:
            data = []
            filenames = list(glob.glob(fb_data_path + "/*.csv")) + list(glob.glob(fb_data_path + "/*.jsonl"))
            print(f"Loading files from : {fb_data_path}: {filenames}")

            for filename in filenames:
                if filename.endswith(".csv"):
                    data.append(pd.read_csv(os.path.join(fb_data_path, filename)))
                elif filename.endswith(".jsonl"):
                    data.append(pd.read_json(os.path.join(fb_data_path, filename), lines=True, orient="records"))
            data = pd.concat(data, ignore_index=True)

        if "transcript_clean" in data.columns and "reward" in data.columns:
            pass
        elif "is_cr" in data.columns:
            print("Building reward model dataset based on DNN CR annotations")
            print("Not taking into account acknowledgements (reward_ack = reward_other)")
            data["response_is_clarification_request"] = data["is_cr"]
            data["response_is_acknowledgement"] = 0
            data["reward"] = data.apply(
                compute_reward_value, axis=1, reward_cr=reward_cr, reward_ack=reward_other, reward_other=reward_other
            )
            data["transcript_clean"] = data["utt_transcript_clean"]
        elif "response_is_clarification_request" in data.columns and "response_is_acknowledgement" in data.columns:
            print("Building reward model dataset based on CR and ACK data")
            data["reward"] = data.apply(
                compute_reward_value, axis=1, reward_cr=reward_cr, reward_ack=reward_ack, reward_other=reward_other
            )
            data["transcript_clean"] = data["utt_transcript_clean"]
        elif "is_grammatical" in data.columns:
            print("Building reward model dataset based on grammaticality")
            data.dropna(subset=["is_grammatical"], inplace=True)
            data["reward"] = data["is_grammatical"].apply(lambda x: (x + 1) / 2)  # map to values 0, 0.5, 1
        elif "sentence_good" in data.columns and "sentence_bad" in data.columns:
            print("Building reward model dataset based on zorro data")
            data_good = data[["sentence_good"]].copy()
            data_good.rename(columns={"sentence_good": "transcript_clean"}, inplace=True)
            data_good["reward"] = 1
            data_bad = data[["sentence_bad"]].copy()
            data_bad.rename(columns={"sentence_bad": "transcript_clean"}, inplace=True)
            data_bad["reward"] = 0
            data = pd.concat([data_good, data_bad], ignore_index=True)
        else:
            raise RuntimeError("Unknown data format in ", fb_data_path)

        data = data[["transcript_clean", "reward"]]
        all_data.append(data)

    all_data = pd.concat(all_data, ignore_index=True)

    data_train, data_test = train_test_split(all_data, test_size=TEST_SET_SIZE, shuffle=True,
                                             random_state=SPLIT_RANDOM_STATE)

    ds_train = Dataset.from_pandas(data_train)
    ds_test = Dataset.from_pandas(data_test)

    ds_train.set_format(type="torch")
    ds_test.set_format(type="torch")

    datasets = DatasetDict({"train": ds_train, "test": ds_test})
    return datasets


@dataclass
class CFRewardTrainerConfig(RewardConfig):
    data_paths: List[str] = field(default_factory=lambda: [CONVERSATIONS_DATA_FILE])

    reward_cr: float = 0
    reward_ack: float = 1
    reward_other: float = 0.5


def main():
    trainer_config_args = CFRewardTrainerConfig
    trainer_config_fields = trainer_config_args.__dataclass_fields__
    trainer_config_fields["bf16"].default = True
    trainer_config_fields["per_device_train_batch_size"].default = 128
    trainer_config_fields["per_device_eval_batch_size"].default = 1024
    trainer_config_fields["logging_steps"].default = 10
    trainer_config_fields["num_train_epochs"].default = 1
    trainer_config_fields["learning_rate"].default = 1.41e-5
    trainer_config_fields["optim"].default = "adamw_torch"
    trainer_config_fields["max_length"].default = 128
    trainer_config_fields["remove_unused_columns"].default = False
    trainer_config_fields["load_best_model_at_end"].default = True
    trainer_config_fields["metric_for_best_model"].default = "mse"
    trainer_config_fields["greater_is_better"].default = False
    trainer_config_fields["save_total_limit"].default = 1
    trainer_config_fields["save_steps"].default = 50
    trainer_config_fields["save_only_model"].default = True
    trainer_config_fields["eval_strategy"].default = "steps"
    trainer_config_fields["eval_steps"].default = 50
    trainer_config_fields["eval_on_start"].default = True

    model_config_args = ModelConfig
    model_config_fields = model_config_args.__dataclass_fields__
    model_config_fields["lora_task_type"].default = "SEQ_CLS"

    parser = HfArgumentParser((trainer_config_args, model_config_args))

    trainer_config, model_config = parser.parse_args_into_dataclasses()
    trainer_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    wandb.init(
        name=trainer_config.run_name,
        project="lm_feedback_reward_model",
        config=parser.parse_args(),
    )

    ################
    # Model & Tokenizer
    ################
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
    raw_datasets = build_reward_model_trainer_datasets(
        trainer_config.data_paths, reward_cr=trainer_config.reward_cr, reward_ack=trainer_config.reward_ack,
        reward_other=trainer_config.reward_other
    )

    def preprocess_function(sample):
        tokenized = tokenizer(sample["transcript_clean"], truncation=True, max_length=trainer_config.max_length)
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
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=eval_dataset,
        num_samples=1000,
        freq=trainer_config.eval_steps,
    )
    trainer.add_callback(progress_callback)

    trainer.train()

    trainer._load_best_model()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
