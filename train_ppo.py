import copy
import os
import time
import typing
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List

import numpy as np
import torch
from accelerate.utils import gather_object
from trl.trainer.ppo_config import JSONDict

import wandb
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from utils import CHILDES_LM_DATA_FILE, BLIMP_METRIC_TO_PHENOMENON

from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, PreTrainedTokenizerBase
from datasets import Dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, PreTrainedModelWrapper
from trl.core import LengthSampler, PPODecorators, entropy_from_logits, masked_mean, clip_by_value, masked_var, \
    flatten_dict, logprobs_from_logits, stack_dicts, WANDB_PADDING, stats_to_np, convert_to_scalar
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
            lm_inputs: Optional[List[torch.LongTensor]] = None,
            lm_loss_coef: float = 0,
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
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            "lm_inputs": lm_inputs
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                        "lm_inputs": [batch_dict["lm_inputs"][i] for i in mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                            mini_batch_dict["lm_inputs"],
                            lm_loss_coef,
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def train_lm_minibatch(self, lm_inputs):
        batch = {"input_ids": lm_inputs}
        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(self.current_device)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        self.model.train()
        lm_output = self.model.pretrained_model(**batch)
        lm_loss = lm_output["loss"]
        return lm_loss

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
            lm_inputs: torch.LongTensor,
            lm_loss_coef: float,
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
        if lm_loss_coef > 0:
            lm_loss = self.train_lm_minibatch(lm_inputs)
            loss = lm_loss_coef * lm_loss + (1 - lm_loss_coef) * loss
            train_stats["loss/language_modeling"] = lm_loss.detach()

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
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), entropy=entropy_loss.detach(),
                      total=loss.detach()),
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


def build_policy_trainer_dataset(tokenizer, query_data_path):
    data_queries = pd.read_csv(query_data_path)
    # data_queries = data_queries.iloc[:1000]

    if "utt_transcript_clean" in data_queries.columns:
        data_queries["transcript_clean"] = data_queries["utt_transcript_clean"]
        del data_queries["utt_transcript_clean"]

    data_queries = data_queries[["transcript_clean"]]

    ds = Dataset.from_pandas(data_queries)

    ds = ds.filter(lambda x: len(x["transcript_clean"]) > 10, batched=False)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["transcript_clean"])
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

    results = {
        "blimp": out["results"].pop("blimp_filtered")["acc,none"],
        "zorro": out["results"].pop("zorro")["acc,none"]
    }

    phenomenon_results = dict()
    for key, val in out["results"].items():
        val = val["acc,none"]
        metric_category = key.split("_")[0]
        key = key[key.index("_") + 1:]
        if metric_category == "zorro":
            phenomenon = key.split("-")[0]
            metric = key[key.index("-") + 1:]
        elif metric_category == "blimp":
            metric = key.replace("_filtered", "")
            phenomenon = BLIMP_METRIC_TO_PHENOMENON[metric]
        else:
            raise RuntimeError("Unknown metric key: ", key)
        prefix = metric_category + '/' + phenomenon
        results[prefix + '-' + metric] = val
        prefix_phen = metric_category + "_phenomena" + '/' + phenomenon
        if prefix_phen in phenomenon_results:
            phenomenon_results[prefix_phen].append(val)
        else:
            phenomenon_results[prefix_phen] = [val]
    phenomenon_results = {key: np.mean(values) for key, values in phenomenon_results.items()}
    results.update(phenomenon_results)

    ppo_trainer.accelerator.log(results, step=ppo_trainer.current_step, log_kwargs={"commit": True})


@dataclass
class CfPPOConfig(PPOConfig):
    model_name: str = "childes-gpt"
    tracker_project_name: str = "lm_feedback_ppo"

    policy_model: str = None
    value_model: str = None

    lm_loss_coef: float = 0

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
            name=wandb_config.exp_name,
            project="lm_feedback_ppo",
            config=wandb_config,
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.policy_model)
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model)

    dataset = build_policy_trainer_dataset(tokenizer, query_data_path=config.query_data_path)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.policy_model)

    ppo_trainer = ChildesPPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    value_model = AutoModelForSequenceClassification.from_pretrained(config.value_model)
    value_model_tokenizer = AutoTokenizer.from_pretrained(config.value_model)

    def eval_babylm_metrics():
        model.save_pretrained(os.path.join(CKPT_DIR, config.exp_name))
        tokenizer.save_pretrained(os.path.join(CKPT_DIR, config.exp_name))

        eval_babylm(model="hf", model_args=f"pretrained={os.path.join(CKPT_DIR, config.exp_name)},add_bos_token=True",
                    tasks=["zorro", "blimp_filtered"],
                    ppo_trainer=ppo_trainer, device=ppo_trainer.accelerator.device.index)

    def generate(batch, query_length_sampler, use_queries):
        generation_kwargs = {
            "min_length": -1,
            "max_new_tokens": config.output_max_length,
            "top_k": config.generation_top_k,
            "top_p": config.generation_top_p,
            "temperature": config.generation_temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if use_queries:
            caregiver_utts = batch["input_ids"]
            batch_query_length = query_length_sampler() + 1  # +1 for BOS token
            query_tensors = [utt[:batch_query_length] for utt in caregiver_utts]
        else:
            bos_tensor = torch.tensor([tokenizer.bos_token_id], device=ppo_trainer.current_device)
            query_tensors = config.batch_size * [bos_tensor]

        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["query"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in query_tensors]
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        return batch, response_tensors, query_tensors

    def compute_rewards(queries, responses, response_tensors):
        texts = [(q + r).strip() for q, r in zip(queries, responses)]

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
        rewards = [r + config.length_reward_coef * length if r > 0.5 else r for r, length in
                   zip(rewards, response_lengths)]

        return rewards


    query_length_sampler = LengthSampler(config.query_min_length, config.query_max_length + 1)
    for step, batch in enumerate(tqdm(ppo_trainer.dataloader, total=config.steps)):
        if (config.eval_freq != -1) and (step % config.eval_freq == 0):
            eval_babylm_metrics()

        use_queries = config.query_max_length > 0
        batch, response_tensors, query_tensors = generate(batch, query_length_sampler, use_queries)
        rewards = compute_rewards(batch["query"], batch["response"], response_tensors)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards, lm_inputs=batch["input_ids"],
                                 lm_loss_coef=config.lm_loss_coef)

        if step % config.log_freq == 0:
            ppo_trainer.log_stats(stats, batch, rewards)

        if step >= config.steps:
            break

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    main()
