import glob
import os

import pandas as pd

from filter_blimp_and_zorro import ZORRO_FILTERED_EVAL_DIR
from utilities import BABYLM_DATA_DIR, SIMPLE_SYNTAX_EVAL_DIR

ZORRO_EVAL_DIR = os.path.join("UnMasked", "test_suites", "zorro")
ZORRO_BABYLM_EVAL_DIR = os.path.join(BABYLM_DATA_DIR, "evaluation_data", "zorro")


def create():
    os.makedirs(SIMPLE_SYNTAX_EVAL_DIR, exist_ok=True)

    dfs_agreement_subject_verb = []
    filename = "agreement_subject_verb-across_relative_clause.jsonl"
    df = pd.read_json(os.path.join(ZORRO_FILTERED_EVAL_DIR, filename), lines=True, orient="records")

    def remove_relative_clause(sent):
        return " ".join(sent.split(" ")[:2] + sent.split(" ")[5:])

    df["sentence_good"] = df.sentence_good.apply(remove_relative_clause)
    df["sentence_bad"] = df.sentence_bad.apply(remove_relative_clause)

    dfs_agreement_subject_verb.append(df)

    filename = "agreement_subject_verb-across_prepositional_phrase.jsonl"
    df = pd.read_json(os.path.join(ZORRO_FILTERED_EVAL_DIR, filename), lines=True, orient="records")

    print(df)
    def remove_prep_phrase(sent):
        return " ".join(sent.split(" ")[:2] + sent.split(" ")[5:])

    df["sentence_good"] = df.sentence_good.apply(remove_prep_phrase)
    df["sentence_bad"] = df.sentence_bad.apply(remove_prep_phrase)

    dfs_agreement_subject_verb.append(df)

    dfs_agreement_subject_verb = pd.concat(dfs_agreement_subject_verb, ignore_index=True)
    df.drop_duplicates(subset=["sentence_good", "sentence_bad"], inplace=True)

    filename_out = "agreement_subject_verb.jsonl"
    dfs_agreement_subject_verb.to_json(os.path.join(SIMPLE_SYNTAX_EVAL_DIR, filename_out), lines=True, orient="records")

    dfs_irregular = []
    filename = "irregular-verb.jsonl"
    df = pd.read_json(os.path.join(ZORRO_FILTERED_EVAL_DIR, filename), lines=True, orient="records")

    wrong_past_forms = {"chose": "choosed", "chosen": "choosed", "begun": "beginned", "began": "beginned", "taken": "taked",
                  "took": "taked", "spoken": "speaked", "spoke": "speaked", "grown": "growed", "grew": "growed",
                  "come": "comed", "came": "comed", "given": "gived", "gave": "gived", "known": "knowed",
                  "knew": "knowed", "written": "writed", "wrote": "writed", "drawn": "drawed", "drew": "drawed",
                  "become": "becomed", "became": "becomed", "seen": "seed", "saw": "seed"}

    def replace_by_regular_inflection(sent):
        verb = sent.split(" ")[1]
        beginning = sent.split(" ")[:1]
        ending = sent.split(" ")[2:]
        if verb == "had":
            verb = sent.split(" ")[2]
            beginning = sent.split(" ")[:2]
            ending = sent.split(" ")[3:]

        verb = wrong_past_forms[verb]

        return " ".join(beginning + [verb] + ending)

    df["sentence_bad"] = df.sentence_bad.apply(replace_by_regular_inflection)

    dfs_irregular.append(df)
    dfs_irregular = pd.concat(dfs_irregular, ignore_index=True)
    df.drop_duplicates(subset=["sentence_good", "sentence_bad"], inplace=True)

    filename_out = "irregular-past-tense.jsonl"
    dfs_irregular.to_json(os.path.join(SIMPLE_SYNTAX_EVAL_DIR, filename_out), lines=True, orient="records")


    # all_task_names = []
    #
    # for filename in glob.glob(os.path.join(ZORRO_EVAL_DIR, "*.txt")):
    #
    #     with open(filename) as file:
    #         print(filename)
    #         all_task_names.append(os.path.basename(filename).split(".txt")[0])
    #
    #         lines = [line.rstrip() for line in file]
    #         df = convert_to_babylm_df(lines, filename)
    #
    #         filename_out = os.path.basename(filename).replace(".txt", ".jsonl")
    #         df.to_json(os.path.join(ZORRO_BABYLM_EVAL_DIR, filename_out), lines=True, orient="records")

    # print(all_task_names)


if __name__ == '__main__':
    create()
