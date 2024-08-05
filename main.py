import math
import os
import warnings

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from transformers import LlamaForCausalLM, LlamaConfig
from torch.optim import AdamW
from data import SEQUENCE_START_TOKEN, MASK_TOKEN, ChildesDataModule

from lm_eval import evaluator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BabyLMModel(LightningModule):
    def __init__(self, initial_lr=1e-4, rl_loss_weight=0, model_name="babyllama"):
        super().__init__()

        self.save_hyperparameters()

        self.initial_lr = initial_lr
        self.model_name = model_name
        self.model_family = "causal" if model_name == "babyllama" else "masked"

    def configure_model(self):
        # config = RobertaConfig(
        #     vocab_size=vocab_size,
        #     max_position_embeddings=max_len,
        #     num_attention_heads=12,
        #     num_hidden_layers=12,
        #     type_vocab_size=1,
        #     hidden_size=384,
        #     intermediate_size=1024,
        # )
        self.vocab_size = self.trainer.datamodule.vocab_size
        self.max_len = self.trainer.datamodule.max_len

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

    def get_hf_cktp_path(self, best=False):
        path = "ckpt_huggingface_best" if best else "ckpt_huggingface_last"

        if isinstance(self.logger, WandbLogger):
            huggingface_ckpt_dir = os.path.join("lightning_logs", self.logger.version, path)
        else:
            huggingface_ckpt_dir = os.path.join(self.logger.log_dir, path)
        return huggingface_ckpt_dir

    def save_huggingface_checkpoint(self, is_best=False):
        """Self checkpoint that is compatible with huggingface"""
        huggingface_ckpt_dir = self.get_hf_cktp_path(best=is_best)
        print(f"Saving huggingface-compatible checkpoint to {huggingface_ckpt_dir}")

        os.makedirs(huggingface_ckpt_dir, exist_ok=True)

        self.model.save_pretrained(huggingface_ckpt_dir)
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(huggingface_ckpt_dir)

    def on_fit_start(self) -> None:
        self.best_val_loss = math.inf
        self.save_huggingface_checkpoint(is_best=True)

    def forward_step_lm(self, batch):
        if self.model_family == "causal":
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        else:
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                             token_type_ids=batch.token_type_ids)

        return out["loss"]

    def forward_step_fb(self, batch):
        if self.model_family == "causal":
            out_fb = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        else:
            out_fb = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                token_type_ids=batch.token_type_ids)

        logits = out_fb["logits"]
        target_logits = [logit[range(logit.shape[0]), input] for logit, input in zip(logits, batch.input_ids)]
        target_logits = torch.stack(target_logits)
        effective_log_prob = target_logits.sum(dim=1) / batch["length"]

        policy_loss = -(batch["reward"] * effective_log_prob).mean()
        return policy_loss

    def training_step(self, batch, batch_idx):
        if self.trainer.datamodule.fb:
            batch, batch_fb = batch["lm"], batch["fb"]
            lm_loss = self.forward_step_lm(batch)
            policy_loss = self.forward_step_fb(batch_fb)

            # entropy_loss = effective_entropy.mean() * args.entropy_coeff

            loss_lm = (1 - self.hparams.rl_loss_weight) * lm_loss
            loss_rl = self.hparams.rl_loss_weight * policy_loss

            self.log(f"train_loss_lm", loss_lm)
            self.log(f"train_loss_rl", loss_rl)
            self.log(f"train_loss_lm_raw", lm_loss, prog_bar=True)
            self.log(f"train_loss_rl_raw", policy_loss, prog_bar=True)

            loss = loss_lm + loss_rl
        else:
            loss = self.forward_step_lm(batch)

        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            loss = self.forward_step_lm(batch)
            self.log(f"val_loss", loss, prog_bar=True, sync_dist=True)

        elif dataloader_idx == 1:
            policy_loss = self.forward_step_fb(batch)
            self.log(f"val_loss_rl_raw", policy_loss, prog_bar=True)

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

    def eval_babylm(self):
        print("Evaluating babylm metrics")
        self.save_huggingface_checkpoint()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = evaluator.simple_evaluate(
                model="hf" if self.model_family == "causal" else "hf-mlm",
                model_args=f"pretrained={self.get_hf_cktp_path()}",
                tasks=["blimp_filtered", "blimp_supplement"],
                batch_size=1024,
                device=f"cuda:{self.trainer.device_ids[0]}",
                cache_requests=True,
            )

        for key, val in out["results"].items():
            if key == "blimp_filtered":
                self.log(key, val["acc,none"], prog_bar=True, sync_dist=True)
            else:
                self.log(key, val["acc,none"])

    def on_validation_epoch_end(self):
        self.generate_sample_sentences()
        if not self.trainer.state.stage == 'sanity_check':
            self.eval_babylm()

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
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.trainer.max_steps,
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler
        }


def cli_main():
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True,
                                          filename="{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min",
                                        min_delta=0.01)
    LightningCLI(
        BabyLMModel,
        ChildesDataModule,
        seed_everything_default=1,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_steps": 150000,
            "accumulate_grad_batches": 1,
            "val_check_interval": 0.2,
            "log_every_n_steps": 1000,
            "num_sanity_val_steps": 3,
            "limit_val_batches": 100,
            "precision": "16-mixed",
            "reload_dataloaders_every_n_epochs": 1,
            "gradient_clip_val": 1,
        },
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.set_float32_matmul_precision('medium')

    cli_main()
