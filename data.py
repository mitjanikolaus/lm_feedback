import os
from pathlib import Path

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import pandas as pd
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast, DataCollatorWithPadding

from utils import DATA_DIR, PATH_DEV, TRAINING_TRACK_DEFAULT, SPEAKER_CODES_CAREGIVER

SEQUENCE_START_TOKEN = "<s>"
MASK_TOKEN = "<mask>"

data_path_dev = os.path.join(DATA_DIR, PATH_DEV)


def train_tokenizer(save_dir, vocab_size, training_track):
    print(f"Training tokenizer for vocab size {vocab_size} (track {training_track}) .. ")
    base_path = os.path.join(DATA_DIR, training_track)
    paths = [str(x) for x in Path(base_path).glob("*.train")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        SEQUENCE_START_TOKEN,
        "<pad>",
        "</s>",
        "<unk>",
        MASK_TOKEN,
    ])

    tokenizer.save_model(save_dir)
    print(f"Saved trained tokenizer to {save_dir}")


DATA_NAMES = ["childes", "bnc_spoken", "cbt", "children_stories", "gutenberg", "open_subtitles", "qed",
              "simple_wikipedia", "switchboard", "wikipedia"]


class BabyLMDataModule(pl.LightningDataModule):
    def __init__(self, training_track=TRAINING_TRACK_DEFAULT, fb=False, fb_data_path=None, vocab_size=10000,
                 max_len=128, batch_size=128, num_workers=4, subset=None):
        super().__init__()
        if subset is None:
            subset = DATA_NAMES
        elif isinstance(subset, str):
            subset = [subset]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fb = fb

        tokenizer_dir = os.path.join("tokenizers", f"lm_feedback_{training_track}_vocab_{vocab_size}")
        os.makedirs(tokenizer_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
            # Train tokenizer if it doesn't exist yet
            train_tokenizer(tokenizer_dir, vocab_size, training_track)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_len=max_len)

        self.dataset_dev = BabyLMDataset(data_path_dev, tokenizer=self.tokenizer, max_len=max_len, subset=subset, split="dev")

        data_path_train = os.path.join(DATA_DIR, training_track)
        self.train_dataset = BabyLMDataset(data_path_train, tokenizer=self.tokenizer, max_len=max_len, subset=subset, split="train")

        if self.fb:
            self.train_fb_dataset = FeedbackDataset(fb_data_path, self.tokenizer, max_len)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        self.collate_fn_fb = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        lm_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.train_fb_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                               num_workers=self.num_workers, collate_fn=self.collate_fn)
        return validation_dataloader


SPEAKER_CODES_CAREGIVER_PREFIXES = ['*' + speaker_code + ':\t' for speaker_code in SPEAKER_CODES_CAREGIVER]


def preprocess_childes_data(lines):
    for line in lines:
        if line[:6] in SPEAKER_CODES_CAREGIVER_PREFIXES:
            yield line[6:].strip()


class BabyLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, subset, split):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        print("Loading LM data: ")
        for src_file in subset:
            src_file_path = os.path.join(data_path, f"{src_file}.{split}")
            print(src_file_path)
            lines = Path(src_file_path).read_text(encoding="utf-8").splitlines()
            lines = [l for l in lines if l]     # Discard empty lines

            if src_file == "childes":
                lines = preprocess_childes_data(lines)

            self.examples += lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        out = self.tokenizer(self.examples[i], add_special_tokens=True, return_special_tokens_mask=True,
                             return_token_type_ids=True, truncation=True, max_length=self.max_len-2)
        return out


def get_reward_value(utt):
    if utt.response_is_clarification_request:
        return 0
    elif utt.response_is_acknowledgement:
        return 1
    else:
        return 0.5


class FeedbackDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        print("Loading FB data.. ", end="")
        data = pd.read_csv(data_path)

        utts_encoded = data.utt_transcript_clean.to_list()
        rewards = data.apply(get_reward_value, axis=1)

        self.examples = list(zip(utts_encoded, rewards))
        print("Done.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        encoded_sample = self.tokenizer(self.examples[i][0], add_special_tokens=True, return_special_tokens_mask=True,
                             return_token_type_ids=True, truncation=True, max_length=self.max_len - 2)
        reward = self.examples[i][1]
        encoded_sample.data["reward"] = reward
        encoded_sample.data["length"] = len(encoded_sample.input_ids)

        return encoded_sample