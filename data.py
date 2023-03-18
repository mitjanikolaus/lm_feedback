import os
from pathlib import Path

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset


class BabyLMDataset(Dataset):
    def __init__(self, data_path, tokenizer_dir, evaluate: bool = False):
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
        src_files = Path(data_path).glob("*.dev") if evaluate else Path(data_path).glob("*.train")
        for src_file in src_files:
            print(src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])