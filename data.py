import os
from pathlib import Path

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import pandas as pd
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

from utils import DATA_DIR, PATH_DEV, TRAINING_TRACK_DEFAULT

data_path_dev = os.path.join(DATA_DIR, PATH_DEV)


def train_tokenizer(save_dir, vocab_size, training_track):
    print(f"Training tokenizer for {vocab_size} (track {training_track}) .. ")
    base_path = os.path.join(DATA_DIR, training_track)
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


class BabyLMDataModule(pl.LightningDataModule):
    def __init__(self, training_track=TRAINING_TRACK_DEFAULT, vocab_size=32000, max_len=512, batch_size=16, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        tokenizer_dir = os.path.join("tokenizers", f"lm_feedback_{training_track}_vocab_{vocab_size}")
        os.makedirs(tokenizer_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
            # Train tokenizer if it doesn't exist yet
            train_tokenizer(tokenizer_dir, vocab_size, training_track)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_len=max_len)

        self.dataset_dev = BabyLMDataset(data_path_dev, tokenizer=self.tokenizer)

        data_path_train = os.path.join(DATA_DIR, training_track)
        self.train_dataset = BabyLMDataset(data_path_train, tokenizer=self.tokenizer)

        # fb_dataset = FeedbackDataset("~/data/lm_feedback/conversations.csv", tokenizer_dir=out_dir)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                               num_workers=self.num_workers, collate_fn=self.collate_fn)
        return validation_dataloader


class BabyLMDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        # TODO needed?
        # self.tokenizer._tokenizer.post_processor = BertProcessing(
        #     (self.tokenizer.eos_token, self.tokenizer.eos_token_id),
        #     (self.tokenizer.cls_token, self.tokenizer.cls_token_id),
        # )
        # self.tokenizer.enable_truncation(max_length=512)

        self.examples = []

        print("Loading data: ")
        src_files = Path(data_path).glob("*")
        for src_file in src_files:
            if not src_file.name.startswith("."):
                print(src_file)
                lines = src_file.read_text(encoding="utf-8").splitlines()
                self.examples += lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.tokenizer.encode_plus(self.examples[i], add_special_tokens=True, return_special_tokens_mask=True,
                                          return_token_type_ids=True)

def get_reward_value(utt):
    if utt.response_is_clarification_request:
        return -1
    elif utt.response_is_acknowledgement:
        return 1
    else:
        return 0.5


class FeedbackDataset(Dataset):
    def __init__(self, data_path, tokenizer_dir):
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt"),
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        print("Loading and encoding data: ")
        data = pd.read_csv(data_path)

        utts_encoded = tokenizer.encode_batch(data.utt_transcript_clean.to_list())
        utts_encoded = [x.ids for x in utts_encoded]
        rewards = data.apply(get_reward_value, axis=1)

        self.examples = list(zip(utts_encoded, rewards))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0]), self.examples[i][1]