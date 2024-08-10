import warnings
from typing import Optional, Union, Callable, List, Dict, Tuple, Any

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import pandas as pd

from data import compute_reward_value
from utils import CHILDES_RL_DATA_FILE

tqdm.pandas()

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification, PreTrainedModel, \
    DataCollator, PreTrainedTokenizerBase, TrainerCallback, EvalPrediction
from datasets import Dataset, DatasetDict

from trl import RewardConfig, ModelConfig, \
    get_quantization_config, get_kbit_device_map, RewardTrainer, get_peft_config


TEST_SET_SIZE = 0.1
SPLIT_RANDOM_STATE = 1


class CFRewardTrainer(RewardTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        super(CFRewardTrainer, self).__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks,
            optimizers, preprocess_logits_for_metrics, max_length, peft_config
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            raise NotImplementedError()
        else:
            loss = -nn.functional.logsigmoid(logits * inputs["reward"]).mean()

        if return_outputs:
            return loss, {
                "logits": logits,
                "rewards": inputs["reward"],
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
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

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
        tokenized = tokenizer(sample["utt_transcript_clean"], truncation=True, max_length=config.max_length)
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
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
