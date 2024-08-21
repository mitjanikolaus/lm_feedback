import copy
import os
import typing
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch
from accelerate.utils import gather_object
from trl.trainer.ppo_config import JSONDict

import wandb
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from utils import CHILDES_LM_DATA_FILE

from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, PreTrainedTokenizerBase
from datasets import Dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, PreTrainedModelWrapper
from trl.core import LengthSampler, PPODecorators, entropy_from_logits, masked_mean, clip_by_value, masked_var, \
    flatten_dict
from lm_eval import evaluator

tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = "ckpts_ppo"


class ChildesPPOTrainer(PPOTrainer):
    def __init__(
            self,
            config: Optional[PPOConfig] = None,
            model: Optional[PreTrainedModelWrapper] = None,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            data_collator: Optional[typing.Callable] = None,
            num_shared_layers: Optional[int] = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            training_data_collator: Optional[typing.Callable] = None,
    ):
        super(ChildesPPOTrainer, self).__init__(config, model, ref_model, tokenizer, dataset, optimizer, data_collator,
                                                num_shared_layers, lr_scheduler, training_data_collator)

    @PPODecorators.empty_device_cache()
    def step(
            self,
            queries: List[torch.LongTensor],
            responses: List[torch.LongTensor],
            scores: List[torch.FloatTensor],
            response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        self.current_step += 1
        return super(ChildesPPOTrainer, self).step(queries, responses, scores, response_masks)

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        loss, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        entropy = masked_mean(entropy_from_logits(logits), mask)
        entropy_loss = - entropy

        loss = pg_loss + self.config.vf_coef * vf_loss
        if self.config.entropy_reg_coef > 0:
            loss += self.config.entropy_reg_coef * entropy_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            entropy_loss = entropy_loss * 0.0
            loss = loss * 0.0

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), entropy=entropy_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return loss, flatten_dict(stats)

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: typing.List[torch.FloatTensor],
        columns_to_log: typing.Iterable[str] = ("query", "response"),
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """

        # all gather stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)
        rewards = self.accelerator.gather(rewards).flatten()

        if self.config.log_with == "wandb":
            import wandb

            if any(column_to_log not in batch.keys() for column_to_log in columns_to_log):
                raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif self.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            self.accelerator.log(
                logs,
                step=self.current_step,
                log_kwargs={"commit": True},
            )

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
    ppo_trainer.accelerator.log(results, step=ppo_trainer.current_step, log_kwargs={"commit": True})


@dataclass
class CfPPOConfig(PPOConfig):
    model_name: str = "childes-gpt"
    tracker_project_name: str = "lm_feedback_ppo"

    policy_model: str = None
    value_model: str = None

    output_min_length: int = 3
    output_max_length: int = 20

    generation_top_p: float = 1.0
    generation_top_k: int = 0
    generation_temperature: float = 1.0

    entropy_reg_coef: float = 0.0
    length_reward_coef: float = 0.0
    score_clip: float = None

    query_data_path: str = CHILDES_LM_DATA_FILE
    query_min_length: int = 1
    query_max_length: int = 0

    eval_freq: int = 100
    log_freq: int = 20

    log_with: str = "wandb"

    accelerator_kwargs: JSONDict = field(default_factory=lambda: {"mixed_precision": "bf16"})


def main():
    parser = HfArgumentParser(CfPPOConfig)
    config = parser.parse_args_into_dataclasses()[0]

    if config.log_with == "wandb":
        wandb_config = copy.deepcopy(config)
        if wandb_config.score_clip is None:
            wandb_config.score_clip = -1
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

    ppo_trainer = ChildesPPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    value_model = AutoModelForSequenceClassification.from_pretrained(config.value_model)
    value_model_tokenizer = AutoTokenizer.from_pretrained(config.value_model)

    generation_kwargs = {
        "min_length": -1,
        "top_k": config.generation_top_k,
        "top_p": config.generation_top_p,
        "temperature": config.generation_temperature,
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
        batch = dict()
        query_tensors = [torch.tensor([tokenizer.bos_token_id], device=ppo_trainer.current_device)] * config.mini_batch_size
        generation_kwargs["max_new_tokens"] = config.output_max_length
        responses = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        response_tensors = [resp for resp in responses]

        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True).strip() for r in response_tensors]
        batch["query"] = [""] * config.batch_size
        return batch, response_tensors, query_tensors

    def generate(batch):
        query_tensors = batch["input_ids"]

        #### Generate text
        response_tensors = []
        for query in query_tensors:
            generation_kwargs["max_new_tokens"] = config.output_max_length
            response = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
            response_tensors.append(response[0])

        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        return batch, response_tensors, query_tensors

    def compute_rewards(batch, response_tensors):
        texts = [(q + r).strip() for q, r in zip(batch["query"], batch["response"])]

        texts_encoded = value_model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                                              max_length=config.output_max_length + 10)
        value_model_outputs = value_model(**texts_encoded)
        rewards = value_model_outputs.logits.squeeze()
        rewards = F.sigmoid(rewards)
        rewards = [torch.tensor(r.item()) for r in rewards]

        # rejection sampling: replace reward with -1 if produced sample is too short
        response_lengths = [len(resp) - 1 for resp in response_tensors]
        rewards = [r if length >= config.output_min_length else torch.tensor(-1.0) for r, length in
                   zip(rewards, response_lengths)]

        # length reward
        rewards = [r + config.length_reward_coef * length if r > 0.5 else r for r, length in zip(rewards, response_lengths)]

        return rewards

    if config.query_max_length > 0:
        for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
            if (config.eval_freq != -1) and (step % config.eval_freq == 0):
                eval_babylm_metrics()

            batch, response_tensors, query_tensors = generate(batch)
            rewards = compute_rewards(batch, response_tensors)

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % config.log_freq == 0:
                ppo_trainer.log_stats(stats, batch, rewards)

            if step >= config.steps:
                break

    else:
        for step in tqdm(range(config.steps)):
            if (config.eval_freq != -1) and (step % config.eval_freq == 0):
                eval_babylm_metrics()

            batch, response_tensors, query_tensors = generate_without_query()
            rewards = compute_rewards(batch, response_tensors)

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % config.log_freq == 0:
                ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    main()
