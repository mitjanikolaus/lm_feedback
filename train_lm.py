import math
import os
import warnings

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from transformers import LlamaForCausalLM, LlamaConfig, GPT2Config
from torch.optim import AdamW
from data import ChildesDataModule, SEQUENCE_START_TOKEN

from lm_eval import evaluator

from model import ChildesGPT
from utilities import parse_babylm_metrics_results

os.environ["WANDB_PROJECT"] = "lm_feedback_baseline"

MODEL_BABYLLAMA = "babyllama"
MODEL_GPT2 = "gpt2"
MODELS_CAUSAL = [MODEL_BABYLLAMA, MODEL_GPT2]
DEFAULT_EVAL_METRICS = ["blimp_filtered_childes", "zorro_filtered_childes"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BabyLMModel(LightningModule):
    def __init__(self, initial_lr=1e-4, model_name=MODEL_GPT2, num_hidden_layers=2,
                 eval_batch_size=1024, hidden_size=512, num_attention_heads=8,
                 eval_metrics=None):
        super().__init__()

        self.save_hyperparameters()

        self.initial_lr = initial_lr
        self.model_name = model_name
        self.num_hidden_layers = num_hidden_layers

        self.eval_batch_size = eval_batch_size
        self.eval_metrics = eval_metrics if eval_metrics else DEFAULT_EVAL_METRICS

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.model_family = "causal" if model_name in MODELS_CAUSAL else "masked"

    def configure_model(self):
        self.vocab_size = self.trainer.datamodule.vocab_size
        self.max_len = self.trainer.datamodule.max_len
        tokenizer = self.trainer.datamodule.tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        if self.model_name == MODEL_BABYLLAMA:
            config = LlamaConfig(**{
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "hidden_act": "silu",
                "hidden_size": self.hidden_size,
                "initializer_range": 0.02,
                "intermediate_size": 1024,
                "num_attention_heads": self.num_attention_heads,
                "num_hidden_layers": self.num_hidden_layers,
                "num_key_value_heads": 8,
                "pretraining_tp": 1,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": 2 * self.max_len,
            })

            self.model = LlamaForCausalLM(config)

        elif self.model_name == MODEL_GPT2:
            config = GPT2Config(
                vocab_size=self.vocab_size,
                n_positions=2*self.max_len,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            self.model = ChildesGPT(config, tokenizer, self.eval_batch_size, self.max_len)

        else:
            raise RuntimeError("Unknown model name: ", self.model_name)

    def get_hf_cktp_path(self):
        path = "ckpt_huggingface_best"
        if isinstance(self.logger, WandbLogger):
            huggingface_ckpt_dir = os.path.join("lightning_logs", self.logger.version, path)
        else:
            huggingface_ckpt_dir = os.path.join(self.logger.log_dir, path)
        return huggingface_ckpt_dir

    def save_huggingface_checkpoint(self):
        """Self checkpoint that is compatible with huggingface"""
        huggingface_ckpt_dir = self.get_hf_cktp_path()
        print(f"Saving huggingface-compatible checkpoint to {huggingface_ckpt_dir}")

        os.makedirs(huggingface_ckpt_dir, exist_ok=True)

        self.model.save_pretrained(huggingface_ckpt_dir)
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer.save_pretrained(huggingface_ckpt_dir)

    def on_fit_start(self) -> None:
        self.best_val_loss = math.inf

    def forward_step_lm(self, batch):
        if self.model_family == "causal":
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        else:
            out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                             token_type_ids=batch.token_type_ids)

        return out["loss"]

    def training_step(self, batch, batch_idx):
        loss = self.forward_step_lm(batch)
        self.log(f"train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step_lm(batch)
        self.log(f"val_loss", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

    def generate_sample_sentences(self):
        tokenizer = self.trainer.datamodule.tokenizer

        generation_prefixes = ["it", "i", "she", "hello", "do"]
        print("\nGenerated samples:")
        for prefix in generation_prefixes:
            sequence = prefix
            if tokenizer.add_bos_token:
                sequence = SEQUENCE_START_TOKEN + prefix
            for step in range(10):
                if self.model_family == "causal":
                    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False,
                                       return_token_type_ids=False).to(device)
                else:
                    raise NotImplementedError()

                with torch.no_grad():
                    out = self.model(**inputs)
                predicted_token = out.logits[0, -1].argmax().cpu()
                prev_inputs = inputs['input_ids'].cpu().numpy()
                encoded_seq = np.concatenate((prev_inputs[0], [predicted_token]))
                sequence = tokenizer.decode(encoded_seq)
                if predicted_token.item() == tokenizer.eos_token_id:
                    break

            print(sequence.replace(SEQUENCE_START_TOKEN, ""))

    def eval_babylm(self):
        print("Evaluating babylm metrics")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = evaluator.simple_evaluate(
                self.model,
                tasks=self.eval_metrics,
                batch_size=self.eval_batch_size,
                device=f"cuda:{self.trainer.device_ids[0]}",
                cache_requests=True,
            )
        results = parse_babylm_metrics_results(out)
        self.log_dict(results)

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
            self.save_huggingface_checkpoint()

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
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=False,
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
            "val_check_interval": 0.25,
            "log_every_n_steps": 100,
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
