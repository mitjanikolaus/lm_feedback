import math
import os
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from transformers import RobertaConfig, RobertaForMaskedLM, LlamaForCausalLM, LlamaConfig, \
    get_cosine_schedule_with_warmup
from torch.optim import AdamW
from data import BabyLMDataModule, SEQUENCE_START_TOKEN, MASK_TOKEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BabyLMModel(LightningModule):
    def __init__(self, vocab_size=5000, initial_lr=1e-3, rl_loss_weight=0, max_len=128, model_name="babyllama",
                 warmup_steps=200):
        super().__init__()

        self.save_hyperparameters()

        self.max_len = max_len
        self.vocab_size = vocab_size

        self.model_name = model_name
        self.model_family = "causal" if model_name == "babyllama" else "masked"

    def configure_model(self):
        # config = RobertaConfig(
        #     vocab_size=vocab_size,
        #     max_position_embeddings=self.max_len,
        #     num_attention_heads=12,
        #     num_hidden_layers=12,
        #     type_vocab_size=1,
        #     hidden_size=384,
        #     intermediate_size=1024,
        # )
        tokenizer = self.trainer.datamodule.tokenizer
        config = LlamaConfig(**{
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "hidden_act": "silu",
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": 2*self.max_len,
        })

        # self.model = RobertaForMaskedLM(config=config)
        # self.model = AutoModelForMaskedLM.from_pretrained("lgcharpe/ELC_BERT_small_baby_10M", trust_remote_code=True) #TODO , config=config
        self.model = LlamaForCausalLM(config)

        # self.model.init_weights()

    def save_huggingface_checkpoint(self, is_best):
        """Self checkpoint that is compatible with huggingface"""
        path = "ckpt_huggingface_best" if is_best else "ckpt_huggingface_last"
        print(f"Saving huggingface-compatible checkpoint to {path}")

        if isinstance(self.logger, WandbLogger):
            huggingface_ckpt_dir = os.path.join("lightning_logs", f"version_{self.logger.version}", path)
        else:
            huggingface_ckpt_dir = os.path.join(self.logger.log_dir, path)

        os.makedirs(huggingface_ckpt_dir, exist_ok=True)

        self.model.save_pretrained(huggingface_ckpt_dir)
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(huggingface_ckpt_dir)

    def on_fit_start(self) -> None:
        self.best_val_loss = math.inf
        self.save_huggingface_checkpoint(is_best=True)

    def training_step(self, batch, batch_idx):
        if self.trainer.datamodule.fb:
            batch, batch_fb = batch["lm"], batch["fb"]
            if self.model_family == "causal":
                out_lm = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            else:
                out_lm = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                                    token_type_ids=batch.token_type_ids)

            if self.model_family == "causal":
                out_fb = self.model(input_ids=batch_fb.input_ids, attention_mask=batch_fb.attention_mask,
                                    labels=batch_fb.labels)
            else:
                out_fb = self.model(input_ids=batch_fb.input_ids, attention_mask=batch_fb.attention_mask,
                                    labels=batch_fb.labels,
                                    token_type_ids=batch_fb.token_type_ids)

            logits = out_fb["logits"]
            target_logits = [logit[range(logit.shape[0]), input] for logit, input in zip(logits, batch_fb.input_ids)]
            target_logits = torch.stack(target_logits)

            effective_log_prob = target_logits.sum(dim=1) / batch_fb["length"]

            policy_loss = -(batch_fb["reward"] * effective_log_prob).mean()

            # entropy_loss = effective_entropy.mean() * args.entropy_coeff

            loss_lm = (1 - self.hparams.rl_loss_weight) * out_lm["loss"]
            loss_rl = self.hparams.rl_loss_weight * policy_loss

            self.log(f"train_loss_lm", loss_lm)
            self.log(f"train_loss_rl", loss_rl)

            loss = loss_lm + loss_rl
        else:
            if self.model_family == "causal":
                out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            else:
                out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                                 token_type_ids=batch.token_type_ids)

            loss = out["loss"]

        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_family == "causal":
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        else:
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                             token_type_ids=batch.token_type_ids)
        self.log(f"val_loss", out["loss"], prog_bar=True, sync_dist=True)
        return out

    def generate_sample_sentences(self):
        tokenizer = self.trainer.datamodule.tokenizer

        generation_prefixes = ["", "it", "it's", "she", "hello", "do"]
        print("\nGenerated samples:")
        for prefix in generation_prefixes:
            sequence = SEQUENCE_START_TOKEN + prefix
            for step in range(10):
                if self.model_family == "causal":
                    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False).to(device)
                else:
                    inputs = tokenizer(sequence + MASK_TOKEN, return_tensors="pt", add_special_tokens=False).to(device)

                with torch.no_grad():
                    out = self.model(**inputs)
                predicted_token = out.logits[0, -1].argmax().cpu().item()
                sequence += tokenizer.decode(predicted_token)

            print(sequence.replace(SEQUENCE_START_TOKEN, ""))

    def on_validation_epoch_end(self):
        self.generate_sample_sentences()

    def on_save_checkpoint(self, checkpoint):
        new_best_val_loss = checkpoint["callbacks"]["EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"][
            "best_score"].item()
        if new_best_val_loss < self.best_val_loss:
            print("saving best checkpoint")
            self.best_val_loss = new_best_val_loss
            self.save_huggingface_checkpoint(is_best=True)
        else:
            print("saving last checkpoint")
            self.save_huggingface_checkpoint(is_best=False)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.initial_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


def cli_main():
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True,
                                          filename="{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min",
                                        min_delta=0.01)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    LightningCLI(
        BabyLMModel,
        BabyLMDataModule,
        seed_everything_default=1,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stop_callback, lr_monitor],
            "max_steps": 15000,
            "accumulate_grad_batches": 10,
            "check_val_every_n_epoch": 1,
            # "val_check_interval": 10000,
            "log_every_n_steps": 1000,
            "num_sanity_val_steps": 3,
            "limit_val_batches": 100,
            "precision": "16-mixed",
            "reload_dataloaders_every_n_epochs": 1,
        },
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.set_float32_matmul_precision('medium')

    cli_main()
