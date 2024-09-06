import glob
import os

import pandas as pd

from utils import BABYLM_DATA_DIR

ZORRO_EVAL_DIR = os.path.join("UnMasked", "test_suites", "zorro")
ZORRO_BABYLM_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "zorro")


def convert_to_babylm_df(lines, filename):
    items = []
    linguistic_term = os.path.basename(filename).split("-")[0]
    uid = os.path.basename(filename).split("-")[1].split(".txt")[0]

    for id, (sentence_good, sentence_bad) in enumerate(zip(lines[0::2], lines[1::2])):
        if "\n" in sentence_bad:
            print(sentence_good)
        sentence_good = sentence_good.replace(" .", ".")
        sentence_bad = sentence_bad.replace(" .", ".")
        sentence_good = sentence_good.replace(" ?", "?")
        sentence_bad = sentence_bad.replace(" ?", "?")
        sentence_good = sentence_good.replace(" !", "!")
        sentence_bad = sentence_bad.replace(" !", "!")

        # Capitalize "I"
        if sentence_good[:2] == "i ":
            sentence_good = sentence_good[0].capitalize() + sentence_good[1:]
        if sentence_bad[:2] == "i ":
            sentence_bad = sentence_bad[0].capitalize() + sentence_bad[1:]
        sentence_good = sentence_good.replace(" i ", " I ")
        sentence_bad = sentence_bad.replace(" i ", " I ")

        item = {
            "sentence_good": sentence_good,
            "sentence_bad": sentence_bad,
            "field": "syntax",
            "linguistic_term": linguistic_term,
            "UID": uid,
            "simple_LM_method": True,
            "pair_id": id
        }
        items.append(item)

    return pd.DataFrame.from_records(items)


def convert_zorro():
    os.makedirs(ZORRO_BABYLM_EVAL_DIR, exist_ok=True)

    all_task_names = []

    for filename in glob.glob(os.path.join(ZORRO_EVAL_DIR, "*.txt")):

        with open(filename) as file:
            print(filename)
            all_task_names.append(os.path.basename(filename).split(".txt")[0])

            lines = [line.rstrip() for line in file]
            df = convert_to_babylm_df(lines, filename)

            filename_out = os.path.basename(filename).replace(".txt", ".jsonl")
            df.to_json(os.path.join(ZORRO_BABYLM_EVAL_DIR, filename_out), lines=True, orient="records")

    print(all_task_names)


if __name__ == '__main__':
    convert_zorro()

