import copy
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from transformers import RobertaConfig, RobertaForMaskedLM
from torch.optim import AdamW
import pytorch_lightning as pl
from data import BabyLMDataModule
from datasets import Dataset

import torch.nn.functional as F


def logprobs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


class BabyLMModel(pl.LightningModule):
    def __init__(self, vocab_size=32000, initial_lr=1e-4, rl_loss_weight=0, kl_coef=0.2, ppo_epochs=4, ppo_batch_size=16, target_kl=0.1):
        super().__init__()

        self.save_hyperparameters()

        max_len = 128

        config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_len,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        self.model = RobertaForMaskedLM(config=config)

    def save_huggingface_checkpoint(self):
        """Self checkpoint that is compatible with huggingface"""
        print("Saving huggingface-compatible checkpoint")

        huggingface_ckpt_dir = os.path.join(self.logger.log_dir, "ckpt_huggingface")
        os.makedirs(huggingface_ckpt_dir, exist_ok=True)

        self.model.save_pretrained(huggingface_ckpt_dir)
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(huggingface_ckpt_dir)

    def on_fit_start(self) -> None:
        self.save_huggingface_checkpoint()

        if self.trainer.datamodule.fb:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()

    def compute_rewards(self, scores, logprobs, ref_logprobs, attention_masks):
        rewards = []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, attention_masks):
            # compute KL penalty (from difference in logprobs)
            kl = logprob - ref_logprob
            reward = -self.hparams.kl_coef * kl
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards)

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.
        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `hidden_dim`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        # for t in reversed(range(gen_len)):
        #     nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        #     delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
        #     lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
        #     advantages_reversed.append(lastgaelam)
        # advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
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
        return pg_loss #, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def minibatch_step_ppo(self, old_logprobs, rewards, logits, logprobs, mask):
        loss, train_stats = self.loss(old_logprobs, rewards, logits, logprobs, mask)
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        # if self.config.max_grad_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(
        #         filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_grad_norm
        #     )

        self.optimizer.step()
        return train_stats

    def training_step_ppo(self, batch):
        bs = batch.input_ids.shape[0]
        lengths = batch["length"]

        with torch.no_grad():
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                             token_type_ids=batch.token_type_ids)
            all_logprobs = logprobs_from_logits(out["logits"], batch.input_ids) # labels = input_ids?

            out_ref = self.ref_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                             token_type_ids=batch.token_type_ids)
            ref_logprobs = logprobs_from_logits(out_ref["logits"], batch.input_ids)

        rewards = self.compute_rewards(batch["reward"], all_logprobs, ref_logprobs, batch.attention_mask)

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                # if key in ["queries", "responses"]:
                #     return_dict[key] = [d[key] for d in data]
                # else:
                return_dict[key] = torch.stack([d[key] for d in data]).to(self.model.device)
            return return_dict

        mini_batch_dict = {
            "logprobs": all_logprobs.to(torch.float32),
            "rewards": rewards,
            "attention_mask": batch.attention_mask,
            "input_ids": batch["input_ids"],
            "token_type_ids": batch["token_type_ids"],
        }

        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.hparams.ppo_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        all_stats = []
        early_stop = False
        for _ in range(self.hparams.ppo_epochs):
            if early_stop:
                break
            for batch in mini_batch_dataloader:
                    # model_inputs = {k: batch[k] for k in model_inputs_names}
                out_ppo = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         token_type_ids=batch["token_type_ids"]
                )
                logits = out_ppo["logits"]
                logprobs = logprobs_from_logits(out["logits"], batch["input_ids"])  # labels = input_ids?

                train_stats = self.minibatch_step_ppo(
                    batch["logprobs"],
                    batch["rewards"],
                    logprobs,
                    logits,
                    batch["attention_mask"],
                )

                all_stats.append(train_stats)

                if self.config.early_stopping:
                    policykl = train_stats["policy/policykl"]
                    early_stop = self._early_stop(policykl)
                    if early_stop:
                        break

        # train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        # train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        # train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        # train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        # stats = self.record_step_stats(
        #     scores=scores,
        #     logprobs=all_logprobs,
        #     ref_logprobs=ref_logprobs,
        #     non_score_reward=non_score_reward,
        #     train_stats=train_stats,
        #     kl_coef=self.kl_ctl.value,
        #     masks=masks,
        # )
        # Gather/Reduce stats from all processes
        # if self.is_distributed:
        #     stats = self.gather_stats(stats)
        # stats = stats_to_np(stats)
        # timing["time/ppo/calc_stats"] = time.time() - t
        # stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        # self.kl_ctl.update(stats["objective/kl"], self.config.batch_size * self.accelerator.num_processes)

        # Log the total ppo time
        # timing["time/ppo/total"] = time.time() - t0
        # stats.update(timing)

        # post-process stats for tensorboard and other loggers
        # if self.config.log_with != "wandb":
        #     stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def training_step(self, batch, batch_idx):
        if self.trainer.datamodule.fb:
            _, batch_fb = batch["lm"], batch["fb"]
            # out_lm = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
            #                  token_type_ids=batch.token_type_ids)

            out_fb = self.training_step_ppo(batch_fb)
            # with torch.no_grad():
            #     all_logprobs, _, values, masks = self.batched_forward_pass(self.model, queries, responses, model_inputs)
            out_fb = self.model(input_ids=batch_fb.input_ids, attention_mask=batch_fb.attention_mask,
                                token_type_ids=batch_fb.token_type_ids)
            logits = out_fb["logits"]
            target_logits = [logit[range(logit.shape[0]), input] for logit, input in zip(logits, batch_fb.input_ids)]
            target_logits = torch.stack(target_logits)

            effective_log_prob = target_logits.sum(dim=1) / batch_fb["length"]

            policy_loss = -(batch_fb["reward"] * effective_log_prob).mean()

            # entropy_loss = effective_entropy.mean() * args.entropy_coeff

            # loss_lm = (1-self.hparams.rl_loss_weight) * out_lm["loss"]
            # loss_rl = self.hparams.rl_loss_weight * policy_loss

            # self.log(f"train_loss_lm", loss_lm)
            # self.log(f"train_loss_rl", loss_rl)
            #
            # loss = loss_lm + loss_rl
        else:
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                             token_type_ids=batch.token_type_ids)
            loss = out["loss"]

        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                         token_type_ids=batch.token_type_ids)
        self.log(f"val_loss", out["loss"], prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        self.save_huggingface_checkpoint()

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.initial_lr)
        return optimizer

    # def on_fit_start(self):
    #     # Set which metrics to use for hyperparameter tuning
    #     metrics = ["val_loss"]
    #     self.logger.log_hyperparams(self.hparams, {m: 100 for m in metrics})


def cli_main():
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=False,
                                          filename="{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min",
                                        min_delta=0.01)

    LightningCLI(
        BabyLMModel,
        BabyLMDataModule,
        seed_everything_default=1,
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": 1000,
            "check_val_every_n_epoch": None,
            "val_check_interval": 10000,
            "log_every_n_steps": 1000,
            "num_sanity_val_steps": 3,
            "limit_val_batches": 100,
            "max_time": "00:60:00:00",  # 60 hours
            "precision": "16-mixed",
            "reload_dataloaders_every_n_epochs": 1,
        },
   )


if __name__ == "__main__":
    cli_main()
