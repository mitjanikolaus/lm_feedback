import argparse
import os
from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments

from data import BabyLMDataset, FeedbackDataset
from utils import DATA_DIR, TRAINING_TRACK_SMALL, TRAINING_TRACK_DEFAULT

TRAINING_TRACK = TRAINING_TRACK_DEFAULT
PATH_DEV = "babylm_dev"


def train_tokenizer(save_dir, vocab_size):
    print("Training tokenizer.. ")
    base_path = os.path.join(DATA_DIR, TRAINING_TRACK)
    paths = [str(x) for x in Path(base_path).glob("*.train")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model(save_dir)
    print(f"Saved trained tokenizer to {save_dir}")


def train():
    args = parse_args()

    out_dir = os.path.join("checkpoints", f"lm_feedback_{TRAINING_TRACK}_vocab_{args.vocab_size}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(os.path.join(out_dir, "vocab.json")):
        # Train tokenizer if it doesn't exist yet
        train_tokenizer(out_dir, vocab_size=args.vocab_size)

    data_path_dev = os.path.join(DATA_DIR, PATH_DEV)
    dataset_dev = BabyLMDataset(data_path_dev, tokenizer_dir=out_dir)

    data_path_train = os.path.join(DATA_DIR, TRAINING_TRACK)
    dataset = BabyLMDataset(data_path_train, tokenizer_dir=out_dir)


    # fb_dataset = FeedbackDataset("~/data/lm_feedback/conversations.csv", tokenizer_dir=out_dir)

    max_len = 512

    config = RobertaConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=max_len+2,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(out_dir, max_len=max_len)

    model = RobertaForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
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
        eval_dataset=dataset_dev,
    )

    trainer.train()

    trainer.save_model(out_dir)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    argparser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
    )

    args = argparser.parse_args()

    return args


if __name__ == '__main__':
    train()
