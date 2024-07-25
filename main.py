import math
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from transformers import RobertaConfig, RobertaForMaskedLM
from torch.optim import AdamW
import pytorch_lightning as pl
from data import BabyLMDataModule


class BabyLMModel(pl.LightningModule):
    def __init__(self, vocab_size=32000, initial_lr=1e-4, rl_loss_weight=0, max_len=128):
        super().__init__()

        self.save_hyperparameters()

        self.max_len = max_len

        config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.max_len,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        self.model = RobertaForMaskedLM(config=config)

        self.best_val_loss = math.inf

    def save_huggingface_checkpoint(self, is_best):
        """Self checkpoint that is compatible with huggingface"""
        path = "ckpt_huggingface_best" if is_best else "ckpt_huggingface_last"
        print(f"Saving huggingface-compatible checkpoint to {path}")

        huggingface_ckpt_dir = os.path.join(self.logger.log_dir, path)
        os.makedirs(huggingface_ckpt_dir, exist_ok=True)

        self.model.save_pretrained(huggingface_ckpt_dir)
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(huggingface_ckpt_dir)

    def on_fit_start(self) -> None:
        self.save_huggingface_checkpoint(is_best=True)

    def training_step(self, batch, batch_idx):
        if self.trainer.datamodule.fb:
            batch, batch_fb = batch["lm"], batch["fb"]
            out_lm = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                             token_type_ids=batch.token_type_ids)

            out_fb = self.model(input_ids=batch_fb.input_ids, attention_mask=batch_fb.attention_mask,
                                token_type_ids=batch_fb.token_type_ids)
            logits = out_fb["logits"]
            target_logits = [logit[range(logit.shape[0]), input] for logit, input in zip(logits, batch_fb.input_ids)]
            target_logits = torch.stack(target_logits)

            effective_log_prob = target_logits.sum(dim=1) / batch_fb["length"]

            policy_loss = -(batch_fb["reward"] * effective_log_prob).mean()

            # entropy_loss = effective_entropy.mean() * args.entropy_coeff

            loss_lm = (1-self.hparams.rl_loss_weight) * out_lm["loss"]
            loss_rl = self.hparams.rl_loss_weight * policy_loss

            self.log(f"train_loss_lm", loss_lm)
            self.log(f"train_loss_rl", loss_rl)

            loss = loss_lm + loss_rl
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
        new_best_val_loss = checkpoint["callbacks"]["EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"]["best_score"].item()
        if new_best_val_loss < self.best_val_loss:
            print("saving best checkpoint")
            self.best_val_loss = new_best_val_loss
            self.save_huggingface_checkpoint(is_best=True)
        else:
            print("saving last checkpoint")
            self.save_huggingface_checkpoint(is_best=False)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.initial_lr)
        return optimizer

    # def on_fit_start(self):
    #     # Set which metrics to use for hyperparameter tuning
    #     metrics = ["val_loss"]
    #     self.logger.log_hyperparams(self.hparams, {m: 100 for m in metrics})


def cli_main():
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True,
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
    # os.environ["TOKENIZERS_PARALLELISM"] = "False"
    cli_main()
