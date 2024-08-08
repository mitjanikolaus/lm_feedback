import os
from pathlib import Path
import re

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer

from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import sent_tokenize

import pandas as pd
from transformers import DataCollatorForLanguageModeling, LlamaTokenizerFast, DataCollatorWithPadding, \
    GPT2TokenizerFast

from utils import BABYLM_DATA_DIR, SPEAKER_CODES_CAREGIVER, BABYLM_DATA_DIR_CLEAN, BABYLM_DATA_PATH_DEV_CLEAN, \
    DEV_SET, TRAINING_TRACK_STRICT_SMALL, TRAIN_SET, CHILDES_LM_DATA_FILE, CHILDES_RL_DATA_FILE

DEV_SET_SIZE = 0.1
SPLIT_RANDOM_STATE = 1

SEQUENCE_START_TOKEN = "<|endoftext|>"
SEQUENCE_END_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|endoftext|>"
MASK_TOKEN = "<mask>"

SPECIAL_TOKENS = [PAD_TOKEN, SEQUENCE_END_TOKEN, SEQUENCE_START_TOKEN, UNK_TOKEN]


def train_tokenizer(save_path, vocab_size, data_iterator=None, data_file_names=None, training_track=None):
    print(f"Training tokenizer for vocab size {vocab_size} .. ")

    tokenizer = ByteLevelBPETokenizer()

    if data_iterator is not None:
        tokenizer.train_from_iterator(data_iterator, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    else:
        paths = [os.path.join(BABYLM_DATA_DIR_CLEAN, training_track, f"{name}.train") for name in data_file_names]
        tokenizer.train(files=paths, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer.save_model(save_path)
    print(f"Saved trained tokenizer to {save_path}")


DATA_FILE_CHILDES = "childes"
DATA_FILE_BNC = "bnc_spoken"
DATA_FILE_GUTENBERG = "gutenberg"
DATA_FILE_OPEN_SUBTITLES = "open_subtitles"
DATA_FILE_WIKI = "simple_wiki"
DATA_FILE_SWITCHBOARD = "switchboard"

DATA_NAMES = [DATA_FILE_CHILDES, DATA_FILE_BNC, DATA_FILE_GUTENBERG, DATA_FILE_OPEN_SUBTITLES, DATA_FILE_WIKI,
              DATA_FILE_SWITCHBOARD]


class ChildesDataModule(LightningDataModule):
    def __init__(self, lm_data_path=CHILDES_LM_DATA_FILE, fb=False, fb_data_path=CHILDES_RL_DATA_FILE, vocab_size=10000,
                 max_len=128, batch_size=128, num_workers=4, causal=True, capitalize_bos=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.fb = fb

        tokenizer_dir = os.path.join("tokenizers", f"childes_vocab_{vocab_size}")
        os.makedirs(tokenizer_dir, exist_ok=True)

        data_df = pd.read_csv(lm_data_path)
        if capitalize_bos:
            data_df["transcript_clean"] = data_df["transcript_clean"].apply(lambda x: x[0].capitalize() + x[1:])
        data = data_df.transcript_clean.to_list()

        data_train, data_dev = train_test_split(data, test_size=DEV_SET_SIZE, shuffle=True,
                                                random_state=SPLIT_RANDOM_STATE)

        if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
            train_tokenizer(tokenizer_dir, vocab_size, data_train)

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            tokenizer_dir, return_token_type_ids=False, add_prefix_space=True, pad_token=PAD_TOKEN
        )

        self.dataset_dev = ChildesLMDataset(data_dev, tokenizer=self.tokenizer, max_len=max_len)
        self.dataset_train = ChildesLMDataset(data_train, tokenizer=self.tokenizer, max_len=max_len)

        if self.fb:
            print("Loading FB data.. ", end="")
            data_fb = pd.read_csv(fb_data_path)
            data_fb["reward"] = data_fb.apply(compute_reward_value, axis=1)
            data_fb = data_fb[["utt_transcript_clean", "reward"]]
            if capitalize_bos:
                data_fb["utt_transcript_clean"] = data_fb["utt_transcript_clean"].apply(
                    lambda x: x[0].capitalize() + x[1:])

            data_fb_train, data_fb_dev = train_test_split(data_fb, test_size=DEV_SET_SIZE, shuffle=True,
                                                          random_state=SPLIT_RANDOM_STATE)
            print("Done.")
            self.dataset_fb_train = FeedbackDataset(data_fb_train, self.tokenizer, max_len)
            self.dataset_fb_dev = FeedbackDataset(data_fb_dev, self.tokenizer, max_len)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=not causal, mlm_probability=0.15 if not causal else None
        )
        self.collate_fn_fb = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                   shuffle=True, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True, collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader

    def val_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_dev, batch_size=self.batch_size, num_workers=self.num_workers,
                                       collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader


class ChildesLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        out = self.tokenizer(self.examples[i], add_special_tokens=True, truncation=True, max_length=self.max_len - 2)
        return out


class BabyLMDataModule(LightningDataModule):
    def __init__(self, training_track=TRAINING_TRACK_STRICT_SMALL, fb=False, fb_data_path=CHILDES_RL_DATA_FILE,
                 vocab_size=10000,
                 max_len=128, batch_size=128, num_workers=4, subset=None, causal=True):
        super().__init__()
        if subset is None:
            data_file_names = DATA_NAMES
        elif isinstance(subset, str):
            data_file_names = [subset]

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.fb = fb

        subset_name = ""
        if data_file_names != DATA_NAMES:
            subset_name = "_subset_" + "_".join(data_file_names)

        for split in [training_track, DEV_SET]:
            data_path = os.path.join(BABYLM_DATA_DIR, split)
            data_path_clean = os.path.join(BABYLM_DATA_DIR_CLEAN, split)

            for data_file_name in data_file_names:
                file_suffix = split if split == DEV_SET else TRAIN_SET
                if not os.path.isfile(os.path.join(data_path_clean, f"{data_file_name}.{file_suffix}")):
                    raw_src_file_path = os.path.join(data_path, f"{data_file_name}.{file_suffix}")
                    print("preprocessing: ", raw_src_file_path)
                    lines = Path(raw_src_file_path).read_text(encoding="utf-8").splitlines()
                    preprocessed = '\n'.join(preprocess(data_file_name, lines))
                    clean_src_file_path = os.path.join(data_path_clean, f"{data_file_name}.{file_suffix}")
                    os.makedirs(data_path_clean, exist_ok=True)
                    Path(clean_src_file_path).write_text(preprocessed, encoding="utf-8")

        tokenizer_dir = os.path.join("tokenizers", f"llama_{training_track}_vocab_{vocab_size}{subset_name}")
        os.makedirs(tokenizer_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
            # Train tokenizer if it doesn't exist yet
            train_tokenizer(tokenizer_dir, vocab_size, data_file_names=data_file_names, training_track=training_track)

        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir, max_len=max_len)

        self.dataset_dev = BabyLMDataset(BABYLM_DATA_PATH_DEV_CLEAN, tokenizer=self.tokenizer, max_len=max_len,
                                         data_file_names=data_file_names,
                                         split="dev")

        data_path_train = os.path.join(BABYLM_DATA_DIR_CLEAN, training_track)
        self.dataset_train = BabyLMDataset(data_path_train, tokenizer=self.tokenizer, max_len=max_len,
                                           data_file_names=data_file_names,
                                           split="train")

        if self.fb:
            self.dataset_fb_train = FeedbackDataset(fb_data_path, self.tokenizer, max_len)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=not causal, mlm_probability=0.15 if not causal else None
        )
        self.collate_fn_fb = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                   shuffle=True, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True, collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=self.collate_fn)
        return validation_dataloader


SPEAKER_CODES_CAREGIVER_PREFIXES = ['*' + speaker_code + ':\t' for speaker_code in SPEAKER_CODES_CAREGIVER]


def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text


def preprocess(src_file_name, lines):
    lines = [l for l in lines if l]  # Discard empty lines

    if src_file_name == DATA_FILE_CHILDES:
        for line in lines:
            if line[:6] in SPEAKER_CODES_CAREGIVER_PREFIXES:
                yield cleanup_extra_spaces(line[6:])
    elif src_file_name == DATA_FILE_WIKI:
        for line in lines:
            if not line.startswith("= = ="):
                for sentence in sent_tokenize(line):
                    yield cleanup_extra_spaces(sentence)

    elif src_file_name == DATA_FILE_GUTENBERG:
        for line in lines:
            if not (line.startswith("*CHAPTER") or line.startswith("= = =")):
                line = line.replace("*", "")
                for sentence in sent_tokenize(line):
                    yield cleanup_extra_spaces(sentence)

    elif src_file_name == DATA_FILE_SWITCHBOARD:
        for line in lines:
            if line.startswith("A:\t") or line.startswith("B:\t"):
                yield cleanup_extra_spaces(line[3:])
            else:
                yield cleanup_extra_spaces(line)

    elif src_file_name in [DATA_FILE_BNC, DATA_FILE_OPEN_SUBTITLES]:
        for line in lines:
            yield cleanup_extra_spaces(line)
    else:
        raise RuntimeError("Unexpected src_file_name: " + src_file_name)


class BabyLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, data_file_names, split):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        print("Loading LM data: ")
        for src_file in data_file_names:
            src_file_path = os.path.join(data_path, f"{src_file}.{split}")
            print(src_file_path)
            lines = Path(src_file_path).read_text(encoding="utf-8").splitlines()
            self.examples += lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        out = self.tokenizer(self.examples[i], add_special_tokens=True, return_special_tokens_mask=True,
                             return_token_type_ids=True, truncation=True, max_length=self.max_len - 2)
        return out


def compute_reward_value(utt):
    if utt.response_is_clarification_request:
        return 0
    elif utt.response_is_acknowledgement:
        return 1
    else:
        return 0.5


class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data.iloc[i]
        encoded_sample = self.tokenizer(item["utt_transcript_clean"], add_special_tokens=True,
                                        truncation=True, max_length=self.max_len - 2)
        encoded_sample.data["reward"] = item["reward"]
        encoded_sample.data["length"] = len(encoded_sample.input_ids)

        return encoded_sample
