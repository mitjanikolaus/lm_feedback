from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Callable, List, Dict, Tuple, Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import wandb
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import nested_detach
from trl.trainer.utils import print_rich_table

from data import preprocess_childes_utterance

tqdm.pandas()

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification, PreTrainedModel, \
    PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset, DatasetDict

from trl import RewardConfig, ModelConfig, \
    get_quantization_config, get_kbit_device_map, RewardTrainer, get_peft_config

TEST_SET_SIZE = 0.2
SPLIT_RANDOM_STATE = 1
DEFAULT_MAX_LENGTH = 256


def compute_acc(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    acc = ((predictions > 0.5) == labels)
    acc_cf = acc[labels == 1].mean().item()
    acc_other = acc[labels == 0].mean().item()
    return {"acc": acc.mean().item(), "acc_cf": acc_cf, "acc_other": acc_other}


@dataclass
class CFClassifierDataCollatorWithPadding:
    r"""
    DataCollator class that pads the inputs to the maximum length of the batch.
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
    target_column: str = None

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
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        ).data

        batch["return_loss"] = True

        if self.target_column is not None:
            batch[self.target_column] = torch.stack([sample[self.target_column] for sample in samples])
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
                 num_samples=100, freq=50, target_column=None):
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
        self.target_column = target_column

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.global_step % self.freq == 0:
            predictions = self.trainer.predict(self.sample_dataset)

            table = defaultdict(list)
            table["text"] = [utt + " " + resp for utt, resp in zip(self.sample_dataset["utt_transcript_clean"],
                                                                   self.sample_dataset["response_transcript_clean"])]
            table[self.target_column] = self.sample_dataset[self.target_column].cpu()
            table["prediction"] = predictions.predictions.squeeze()

            df = pd.DataFrame(table)
            print_rich_table(df[:10])
            records_table = self._wandb.Table(dataframe=df)
            self._wandb.log({"sample_predictions": records_table})


class CFClassifierTrainer(RewardTrainer):

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
            target_column: str = None,
            class_weight=None,
    ):
        data_collator = CFClassifierDataCollatorWithPadding(tokenizer, max_length=max_length,
                                                            target_column=target_column)
        compute_metrics = compute_acc
        self.target_column = target_column
        self.class_weight = class_weight
        super(CFClassifierTrainer, self).__init__(
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

        labels = inputs[self.target_column]

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

        targets = inputs[self.target_column].to(torch.float)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), target=targets, pos_weight=self.class_weight
        )

        if return_outputs:
            return loss, {
                "logits": logits,
            }
        return loss


def build_cf_classifier_datasets(data_path, target_column):
    data = pd.read_csv(data_path)
    data = data[["utt_transcript_clean", "response_transcript_clean", target_column]]

    data["utt_transcript_clean"] = data["utt_transcript_clean"].apply(preprocess_childes_utterance)
    data["response_transcript_clean"] = data["response_transcript_clean"].apply(preprocess_childes_utterance)

    data_train, data_test = train_test_split(data, test_size=TEST_SET_SIZE, shuffle=True,
                                             random_state=SPLIT_RANDOM_STATE)

    print(f"train data stats:\n{data_train[target_column].value_counts()}")
    print(f"test data stats:\n{data_test[target_column].value_counts()}")
    ds_train = Dataset.from_pandas(data_train)
    ds_test = Dataset.from_pandas(data_test)

    ds_train.set_format(type="torch")
    ds_test.set_format(type="torch")

    datasets = DatasetDict({"train": ds_train, "test": ds_test})
    return datasets


def calc_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    return class_weights


@dataclass
class CFClassfierConfig():
    data_path: str
    target_column: str
    use_class_weights: bool = True


def main():
    trainer_config_args = RewardConfig
    reward_config_fields = trainer_config_args.__dataclass_fields__
    reward_config_fields["bf16"].default = True
    reward_config_fields["per_device_train_batch_size"].default = 32
    reward_config_fields["per_device_eval_batch_size"].default = 32
    reward_config_fields["logging_steps"].default = 1
    reward_config_fields["num_train_epochs"].default = 20
    reward_config_fields["learning_rate"].default = 1.41e-5
    reward_config_fields["optim"].default = "adamw_torch"
    reward_config_fields["max_length"].default = DEFAULT_MAX_LENGTH
    reward_config_fields["remove_unused_columns"].default = False
    reward_config_fields["load_best_model_at_end"].default = True
    reward_config_fields["metric_for_best_model"].default = "acc"
    reward_config_fields["greater_is_better"].default = True
    reward_config_fields["save_total_limit"].default = 1
    reward_config_fields["save_steps"].default = 5
    reward_config_fields["save_only_model"].default = True
    reward_config_fields["eval_strategy"].default = "steps"
    reward_config_fields["eval_steps"].default = 5
    reward_config_fields["eval_on_start"].default = True

    model_config_args = ModelConfig
    model_config_fields = model_config_args.__dataclass_fields__
    model_config_fields["lora_task_type"].default = "SEQ_CLS"

    parser = HfArgumentParser((trainer_config_args, model_config_args, CFClassfierConfig))

    trainer_config, model_config, _ = parser.parse_args_into_dataclasses()
    trainer_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    args = parser.parse_args()
    wandb.init(
        name=trainer_config.run_name,
        project="lm_feedback_cf_classifiers",
        config=args,
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

    raw_datasets = build_cf_classifier_datasets(args.data_path, args.target_column)

    def preprocess_function(samples):
        texts = [utt + tokenizer.sep_token + resp for utt, resp in
                 zip(samples["utt_transcript_clean"], samples["response_transcript_clean"])]
        tokenized = tokenizer(texts, truncation=True, max_length=trainer_config.max_length)
        tokenized[args.target_column] = samples[args.target_column]

        return tokenized

    # Preprocess the dataset and truncate examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    class_weight = None
    if args.use_class_weights:
        class_weights = calc_class_weights(train_dataset[args.target_column].numpy())
        class_weight = torch.tensor(class_weights[1])
        print(f"Model loss pos class weight: {class_weight}")

    ################
    # Training
    ################
    trainer = CFClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        target_column=args.target_column,
        class_weight=class_weight,
    )
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=eval_dataset,
        num_samples=100,
        freq=trainer_config.eval_steps,
        target_column=args.target_column
    )
    trainer.add_callback(progress_callback)

    trainer.train()

    trainer._load_best_model()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
