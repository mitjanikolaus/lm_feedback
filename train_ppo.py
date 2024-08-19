import os
import warnings
from dataclasses import dataclass, field

import torch
from trl.commands.scripts.ppo import query_tensors
from trl.trainer.ppo_config import JSONDict

import wandb
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from utils import CHILDES_LM_DATA_FILE

from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from datasets import Dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from lm_eval import evaluator

tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = "ckpts_ppo"


# class ChildesPPOTrainer(PPOTrainer):
#     def __init__(
#             self,
#             config: Optional[PPOConfig] = None,
#             model: Optional[PreTrainedModelWrapper] = None,
#             ref_model: Optional[PreTrainedModelWrapper] = None,
#             tokenizer: Optional[PreTrainedTokenizerBase] = None,
#             dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
#             optimizer: Optional[torch.optim.Optimizer] = None,
#             data_collator: Optional[typing.Callable] = None,
#             num_shared_layers: Optional[int] = None,
#             lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#             training_data_collator: Optional[typing.Callable] = None,
#     ):
#         super(ChildesPPOTrainer, self).__init__(config, model, ref_model, tokenizer, dataset, optimizer, data_collator,
#                                                 num_shared_layers, lr_scheduler, training_data_collator)
#
#     def generate(
#         self,
#         query_tensor: Optional[Union[torch.Tensor, typing.List[torch.Tensor]]] = None,
#         length_sampler: Optional[typing.Callable] = None,
#         batch_size: int = 4,
#         return_prompt: bool = True,
#         generate_ref_response: bool = False,
#         **generation_kwargs,
#     ):
#         """
#         Generate response with the model given the query tensor.
#         call the `generate` method of the model.
#
#         Args:
#             query_tensor (`torch.LongTensor`):
#                 A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
#             length_sampler (`Callable`, *optional*):
#                 Callable that returns the number of newly generated tokens.
#             batch_size (`int`, *optional):
#                 Batch size used for generation, defaults to `4`.
#             return_prompt (`bool`, *optional*):
#                 If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
#             generate_ref_response (`bool`, *optional*):
#                 If set to `True` the reference response is also generated, defaults to `False`.
#             generation_kwargs (dict[str, Any]):
#                 Keyword arguments for generation.
#
#         Returns:
#             `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
#         """
#         if generate_ref_response:
#             ref_model = self.model if self.is_peft_model else self.ref_model
#         if isinstance(query_tensor, List):
#             response = self._generate_batched(
#                 self.model,
#                 query_tensor,
#                 length_sampler=length_sampler,
#                 batch_size=batch_size,
#                 return_prompt=return_prompt,
#                 **generation_kwargs,
#             )
#             if generate_ref_response:
#                 ref_response = self._generate_batched(
#                     ref_model,
#                     query_tensor,
#                     length_sampler=length_sampler,
#                     batch_size=batch_size,
#                     return_prompt=return_prompt,
#                     **generation_kwargs,
#                 )
#
#         else:
#             if query_tensor is None:
#                 # No query given
#                 with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
#                     response = unwrapped_model.generate(**generation_kwargs)
#             else:
#                 if len(query_tensor.shape) == 2:
#                     raise ValueError(
#                         "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
#                     )
#
#                 if length_sampler is not None:
#                     generation_kwargs["max_new_tokens"] = length_sampler()
#
#                 with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
#                     response = unwrapped_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)
#
#             if generate_ref_response:
#                 with unwrap_model_for_generation(
#                     ref_model, self.accelerator, is_peft_model=self.is_peft_model
#                 ) as unwrapped_model:
#                     ref_response = unwrapped_model.generate(
#                         input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
#                     )
#
#             if not return_prompt and not self.is_encoder_decoder:
#                 response = response[:, query_tensor.shape[0] :]
#                 if generate_ref_response:
#                     ref_response = ref_response[:, query_tensor.shape[0] :]
#
#         if generate_ref_response:
#             return response, ref_response
#         return response

    # def _generate_batched(
    #     self,
    #     model: PreTrainedModelWrapper,
    #     query_tensors: List[torch.Tensor],
    #     length_sampler: Optional[typing.Callable] = None,
    #     batch_size: int = 4,
    #     return_prompt: bool = True,
    #     pad_to_multiple_of: Optional[int] = None,
    #     remove_padding: bool = True,
    #     **generation_kwargs,
    # ):
    #     outputs = []
    #
    #     padding_side_default = self.tokenizer.padding_side
    #     if not self.is_encoder_decoder:
    #         self.tokenizer.padding_side = "left"
    #
    #     # in case we have fewer examples than bs
    #     batch_size = min(len(query_tensors), batch_size)
    #
    #     for i in range(0, len(query_tensors), batch_size):
    #         if length_sampler is not None:
    #             generation_kwargs["max_new_tokens"] = length_sampler()
    #
    #         # prevent overflow if query tensors are not even multiple of bs
    #         end_index = min(len(query_tensors), i + batch_size)
    #
    #         batch = query_tensors[i:end_index]
    #         batch_mask = [torch.ones_like(element) for element in batch]
    #         inputs = {"input_ids": batch, "attention_mask": batch_mask}
    #
    #         padded_inputs = self.tokenizer.pad(
    #             inputs,
    #             padding=True,
    #             max_length=None,
    #             pad_to_multiple_of=pad_to_multiple_of,
    #             return_tensors="pt",
    #         ).to(self.current_device)
    #
    #         with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    #             generations = unwrapped_model.generate(**padded_inputs, **generation_kwargs)
    #
    #         for generation, mask in zip(generations, padded_inputs["attention_mask"]):
    #             if not self.is_encoder_decoder:
    #                 output = generation[(1 - mask).sum() :]  # remove padding
    #             else:
    #                 output = generation
    #
    #             if not return_prompt and not self.is_encoder_decoder:
    #                 output = output[(mask).sum() :]  # remove prompt
    #
    #             if remove_padding and self.tokenizer.eos_token_id in output[1:]:
    #                 pad_mask = output[1:] == self.tokenizer.eos_token_id
    #                 pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item() + 1
    #                 output = output[: pad_start + 1]  # keep the eos token at the end
    #
    #             outputs.append(output)
    #
    #     self.tokenizer.padding_side = padding_side_default
    #     return outputs


def build_policy_trainer_dataset(tokenizer, query_data_path, min_length=1, max_length=4):
    data_queries = pd.read_csv(query_data_path)
    # data_queries = data_queries.iloc[:1000]

    if "utt_transcript_clean" in data_queries.columns:
        data_queries["transcript_clean"] = data_queries["utt_transcript_clean"]
        del data_queries["utt_transcript_clean"]

    data_queries = data_queries[["transcript_clean"]]

    # data_queries = data_queries.iloc[:1000]
    ds = Dataset.from_pandas(data_queries)

    ds = ds.filter(lambda x: len(x["transcript_clean"]) > 10, batched=False)

    input_size = LengthSampler(min_length, max_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["transcript_clean"])[: input_size()+2]
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=10)
    ds.set_format(type="torch")
    return ds


def eval_babylm(model, model_args, tasks, ppo_trainer, device, eval_batch_size=1024):
    print("Evaluating babylm metrics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=tasks,
            batch_size=eval_batch_size,
            device=f"cuda:{device}",
            cache_requests=True,
        )

    results = {key.replace("_", "/"): val["acc,none"] for key, val in out["results"].items() if "acc,none" in val}
    ppo_trainer.accelerator.log(results)

@dataclass
class CfPPOConfig(PPOConfig):
    model_name: str = "childes-gpt"
    tracker_project_name: str = "lm_feedback_ppo"

    policy_model: str = None
    value_model: str = None

    output_min_length: int = 7
    output_max_length: int = 20

    generation_top_p: float = 1.0

    query_data_path: str = CHILDES_LM_DATA_FILE
    query_min_length: int = 1
    query_max_length: int = 0

    eval_freq: int = 100
    log_freq: int = 10

    log_with: str = "wandb"

    accelerator_kwargs: JSONDict = field(default_factory=lambda: {"mixed_precision": "bf16"})


def main():
    parser = HfArgumentParser(CfPPOConfig)
    config = parser.parse_args_into_dataclasses()[0]

    if config.log_with == "wandb":
        wandb.init(
            name=config.exp_name,
            project="lm_feedback_ppo",
            config=config,
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.policy_model)
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model)

    if config.query_max_length > 0:
        dataset = build_policy_trainer_dataset(tokenizer, query_data_path=config.query_data_path,
                                               min_length=config.query_min_length, max_length=config.query_max_length)
    else:
        dataset = None

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.policy_model)

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    value_model = AutoModelForSequenceClassification.from_pretrained(config.value_model)
    value_model_tokenizer = AutoTokenizer.from_pretrained(config.value_model)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": config.generation_top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    def eval_babylm_metrics():
        model.save_pretrained(os.path.join(CKPT_DIR, config.exp_name))
        tokenizer.save_pretrained(os.path.join(CKPT_DIR, config.exp_name))

        eval_babylm(model="hf", model_args=f"pretrained={os.path.join(CKPT_DIR, config.exp_name)},add_bos_token=True",
                    tasks=["zorro", "blimp_filtered"],
                    ppo_trainer=ppo_trainer, device=ppo_trainer.accelerator.device.index)

    def generate_without_query():
        #### Generate text
        response_tensors = []
        batch = dict()
        query_tensors = []

        query = [torch.tensor([tokenizer.bos_token_id], device=ppo_trainer.current_device)] * config.mini_batch_size
        query_tensors.extend(query)
        # generate until enough sentences of min lengths
        while len(response_tensors) < len(query):
            generation_kwargs["max_new_tokens"] = config.output_max_length
            responses = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
            response_tensors.extend(
                [resp.squeeze() for resp in responses if resp.shape[-1] - 1 >= config.output_min_length])

        response_tensors = response_tensors[:config.mini_batch_size]

        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True).strip() for r in response_tensors]
        batch["query"] = [""] * config.batch_size
        return batch, response_tensors, query_tensors

    def generate(batch):
        query_tensors = batch["input_ids"]

        #### Generate text
        response_tensors = []
        i = 0
        while len(response_tensors) < len(query_tensors):
            query = query_tensors[i % len(query_tensors)]
            generation_kwargs["max_new_tokens"] = config.output_max_length
            response = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
            if response.shape[-1] - 1 >= config.output_min_length:
                response_tensors.append(response.squeeze())
            i = i + 1

        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        return batch, response_tensors, query_tensors

    def compute_rewards(batch):
        texts = [(q + r).strip() for q, r in zip(batch["query"], batch["response"])]

        texts_encoded = value_model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                                              max_length=config.output_max_length + 10)
        value_model_outputs = value_model(**texts_encoded)
        rewards = value_model_outputs.logits.squeeze()
        rewards = F.sigmoid(rewards)
        rewards = [torch.tensor(r.item()) for r in rewards]

        return rewards

    if config.query_max_length > 0:
        for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
            if step % config.eval_freq == 0:
                eval_babylm_metrics()

            batch, response_tensors, query_tensors = generate(batch)
            rewards = compute_rewards(batch)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % config.log_freq == 0:
                ppo_trainer.log_stats(stats, batch, rewards)

    else:
        for step in tqdm(range(config.steps)):
            if step % config.eval_freq == 0:
                eval_babylm_metrics()

            batch, response_tensors, query_tensors = generate_without_query()
            rewards = compute_rewards(batch)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % config.log_freq == 0:
                ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    main()
