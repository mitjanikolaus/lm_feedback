from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
import pytorch_lightning as pl

from data import BabyLMDataModule


class BabyLMModel(pl.LightningModule):
    def __init__(self, vocab_size=32000):
        super().__init__()

        self.save_hyperparameters()

        max_len = 512

        config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_len,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        self.model = RobertaForMaskedLM(config=config)

    def training_step(self, batch, batch_idx):
        out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                         token_type_ids=batch.token_type_ids)
        self.log(f"train_loss", out["loss"], prog_bar=True)

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels,
                         token_type_ids=batch.token_type_ids)
        self.log(f"val_loss", out["loss"], prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters())
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
            "max_epochs": 10,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1000,
            "num_sanity_val_steps": 3,
            "limit_val_batches": 100,
            "max_time": "00:19:00:00",  # 19 hours
            "precision": "16-mixed",
        },
   )


if __name__ == "__main__":
    cli_main()
