import glob
import os
import pickle

import pandas as pd
import nltk
from tqdm import tqdm

from utils import BABYLM_DATA_DIR, CHILDES_LM_DATA_FILE

ZORRO_EVAL_DIR = os.path.join("UnMasked", "test_suites", "zorro")
ZORRO_BABYLM_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "zorro")


def convert_to_babylm_df(lines, filename, capitalize_sentences):
    items = []
    for id, (sentence_good, sentence_bad) in enumerate(zip(lines[0::2], lines[1::2])):
        if capitalize_sentences:
            sentence_good = sentence_good[0].capitalize() + sentence_good[1:]
            sentence_bad = sentence_bad[0].capitalize() + sentence_bad[1:]

        item = {
            "sentence_good": sentence_good,
            "sentence_bad": sentence_bad,
            "field": "syntax",
            "linguistic_term": os.path.basename(filename).split("-")[0],
            "UID": os.path.basename(filename).split("-")[1].split(".txt")[0],
            "simple_LM_method": True,
            "one_prefix_method": False,
            "two_prefix_method": False,
            "lexically_identical": True,
            "pair_id": id
        }
        items.append(item)

    return pd.DataFrame.from_records(items)



def filter_zorro(filter_by_vocab=False, capitalize_sentences=True):
    os.makedirs(ZORRO_BABYLM_EVAL_DIR, exist_ok=True)

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
    all_task_names = []

    for filename in glob.glob(os.path.join(ZORRO_EVAL_DIR, "*.txt")):

        def contains_out_of_vocab_words(row):
            for sentence in [row.sentence_good, row.sentence_bad]:
                for word in nltk.word_tokenize(sentence):
                    if word not in vocab_allowed:
                        missing_words.add(word)
                        return True
            return False

        with open(filename) as file:
            print(filename)
            all_task_names.append(os.path.basename(filename).split(".txt")[0])

            lines = [line.rstrip() for line in file]
            df = convert_to_babylm_df(lines, filename, capitalize_sentences)

            if filter_by_vocab:
                print(f"Len before: {len(df)}")
                df["contains_out_of_vocab_words"] = df.apply(contains_out_of_vocab_words, axis=1)
                df = df[~df["contains_out_of_vocab_words"]].copy()
                del df["contains_out_of_vocab_words"]
                print(f"Len after: {len(df)}\n")

            filename_out = os.path.basename(filename).replace(".txt", ".jsonl")
            df.to_json(os.path.join(ZORRO_BABYLM_EVAL_DIR, filename_out), lines=True, orient="records")

    print(all_task_names)
    if filter_by_vocab:
        print("missing words: ", missing_words)


if __name__ == '__main__':
    filter_zorro()

