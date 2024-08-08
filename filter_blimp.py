import glob
import os
import pickle

import pandas as pd
import nltk
from tqdm import tqdm

from utils import BABYLM_DATA_DIR, CHILDES_LM_DATA_FILE

BLIMP_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "blimp_filtered")
BLIMP_FILTERED_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "blimp_filtered_childes")


def filter_blimp():
    os.makedirs(BLIMP_FILTERED_EVAL_DIR, exist_ok=True)

    if os.path.isfile(os.path.join(BABYLM_DATA_DIR, "vocab_allowed.p")):
        vocab_allowed = pickle.load(open(os.path.join(BABYLM_DATA_DIR, "vocab_allowed.p"), "rb"))
    else:
        vocab_allowed = set()
        data_df = pd.read_csv(CHILDES_LM_DATA_FILE)
        data_df["transcript_clean"] = data_df["transcript_clean"].apply(lambda x: x[0].capitalize() + x[1:])
        data = data_df.transcript_clean.to_list()
        for sentence in tqdm(data):
            vocab_allowed = vocab_allowed | set(nltk.word_tokenize(sentence))

        pickle.dump(vocab_allowed, open(os.path.join(BABYLM_DATA_DIR, "vocab_allowed.p"), "wb"))

    missing_words = set()
    for file in glob.glob(os.path.join(BLIMP_EVAL_DIR, "*.jsonl")):
        print(file)

        def contains_out_of_vocab_words(row):
            for sentence in [row.sentence_good, row.sentence_bad]:
                for word in nltk.word_tokenize(sentence):
                    if word not in vocab_allowed:
                        missing_words.add(word)
                        return True
            return False

        df = pd.read_json(path_or_buf=file, lines=True)
        print(f"Len before: {len(df)}")
        df["contains_out_of_vocab_words"] = df.apply(contains_out_of_vocab_words, axis=1)
        df_filtered = df[~df["contains_out_of_vocab_words"]].copy()
        del df_filtered["contains_out_of_vocab_words"]
        print(f"Len after: {len(df_filtered)}\n")
        df_filtered.to_json(os.path.join(BLIMP_FILTERED_EVAL_DIR, os.path.basename(file)), lines=True, orient="records")

    print("missing words: ", missing_words)


if __name__ == '__main__':
    filter_blimp()

