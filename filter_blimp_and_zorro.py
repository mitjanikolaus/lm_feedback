import glob
import os
import pickle
from collections import Counter

import pandas as pd
import nltk
from tqdm import tqdm

from utilities import BABYLM_DATA_DIR, CHILDES_LM_TRAIN_DATA_FILE

BLIMP_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "blimp_filtered")
BLIMP_FILTERED_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "blimp_filtered_childes")

ZORRO_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "zorro")
ZORRO_FILTERED_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "zorro_filtered_childes")


def filter_task_files(eval_files_dir, out_eval_files_dir, lowercase=True):
    os.makedirs(out_eval_files_dir, exist_ok=True)

    vocab_path = os.path.join(BABYLM_DATA_DIR, "vocab_allowed.p")
    if os.path.isfile(vocab_path):
        vocab_allowed = pickle.load(open(vocab_path, "rb"))
    else:
        vocab_allowed = set()
        with open(CHILDES_LM_TRAIN_DATA_FILE, "r") as file:
            data = file.read().split("\n")
        for sentence in tqdm(data):
            vocab_allowed = vocab_allowed | set(nltk.word_tokenize(sentence))

        pickle.dump(vocab_allowed, open(vocab_path, "wb"))

    if lowercase:
        vocab_allowed = set([v.lower() for v in vocab_allowed])
    missing_words = Counter()
    for file in glob.glob(os.path.join(eval_files_dir, "*.jsonl")):
        print(file)

        def contains_out_of_vocab_words(row):
            for sentence in [row.sentence_good, row.sentence_bad]:
                if lowercase:
                    sentence = sentence.lower()

                for word in nltk.word_tokenize(sentence):
                    if word not in vocab_allowed:
                        missing_words.update([word])
                        return True
            return False

        df = pd.read_json(path_or_buf=file, lines=True)
        print(f"Len before: {len(df)}")

        df["contains_out_of_vocab_words"] = df.apply(contains_out_of_vocab_words, axis=1)
        df_filtered = df[~df["contains_out_of_vocab_words"]].copy()
        del df_filtered["contains_out_of_vocab_words"]
        print(f"Len after: {len(df_filtered)}\n")

        df_filtered.to_json(os.path.join(out_eval_files_dir, os.path.basename(file)), lines=True, orient="records")

    print("missing words: ", missing_words.most_common(100))


if __name__ == '__main__':
    filter_task_files(BLIMP_EVAL_DIR, BLIMP_FILTERED_EVAL_DIR)
    filter_task_files(ZORRO_EVAL_DIR, ZORRO_FILTERED_EVAL_DIR)
