import os
from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments

from data import BabyLMDataset
from utils import DATA_DIR, TRAINING_TRACK_SMALL, TRAINING_TRACK_DEFAULT

TRAINING_TRACK = TRAINING_TRACK_DEFAULT


def train_tokenizer(save_dir):
    print("Training tokenizer.. ")
    base_path = os.path.join(DATA_DIR, TRAINING_TRACK)
    paths = [str(x) for x in Path(base_path).glob("*.train")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])


    tokenizer.save_model(save_dir)
    print(f"Saved trained tokenizer to {save_dir}")


def train():
    out_dir = os.path.join("checkpoints", "lm_feedback_"+TRAINING_TRACK)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(os.path.join(out_dir, "vocab.json")):
        # Train tokenizer if it doesn't exist yet
        train_tokenizer(out_dir)

    data_path = os.path.join(DATA_DIR, TRAINING_TRACK)
    dataset = BabyLMDataset(data_path, tokenizer_dir=out_dir)

    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(out_dir, max_len=512)

    model = RobertaForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(out_dir)


if __name__ == '__main__':
    train()
