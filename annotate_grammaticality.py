import argparse
import math

import torch
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from eval import load_childes_grammar_model, compute_scores_childes_grammaticality
from utilities import CONVERSATIONS_ANNOTATED_DATA_FILE
import seaborn as sns

BATCH_SIZE = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    model, tokenizer = load_childes_grammar_model(args.eval_model_path)
    model.to(device)

    data = pd.read_csv(args.data_path)

    annotations = []
    num_batches = math.ceil(data.shape[0] / BATCH_SIZE)
    for is_cr in range(num_batches):
        data_batch = data.iloc[is_cr * BATCH_SIZE:(is_cr + 1) * BATCH_SIZE]
        is_grammatical = compute_scores_childes_grammaticality(data_batch["utt_transcript_clean"], model, tokenizer)
        annotations.extend(is_grammatical)
    data["is_grammatical"] = annotations

    out_path = args.data_path.replace(".csv", "_annotated_grammar.csv")
    print(f"Saving results to {out_path}")
    data.to_csv(out_path, index=False)

    print(f"\nStats for annotated data:")

    print("\n CR:")
    print("mean: ", data[data.is_cr == 1].is_grammatical.mean())
    print(data[data.is_cr == 1].is_grammatical.value_counts())

    print("\n Other:")
    print("mean: ", data[data.is_cr == 0].is_grammatical.mean())
    print(data[data.is_cr == 0].is_grammatical.value_counts())

    print(f"\nSamples of annotated data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(data.sample(50))

    plt.figure()
    entries = []
    for transcript_file in tqdm(data.transcript_file.unique()):
        data_transcript = data[data.transcript_file == transcript_file]
        if len(data_transcript) > 100:
            for is_cr in [0, 1]:
                data_filtered = data_transcript[data_transcript.is_cr == is_cr]
                counts = data_filtered.is_grammatical.value_counts(normalize=True)
                for is_grammatical, count in zip(counts.index, counts):
                    grammaticality = "ungrammatical" if is_grammatical == -1 else "grammatical" if is_grammatical == 1 else "ambiguous"
                    entries.append({"is_cr": "clarification request" if is_cr else "other", "grammaticality": grammaticality, "proportion": count, "transcript_file": transcript_file})

    df = pd.DataFrame(entries)
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    fig = sns.barplot(df, x="grammaticality", y="proportion", hue="is_cr", errorbar="ci")
    plt.xlabel("")
    fig.legend_.set_title("")
    fig.get_figure().savefig("grammaticality.png", dpi=300)

    # plt.figure()
    # entries = []
    # for is_grammatical in [-1, 0, 1]:
    #     data_filtered = data[data.is_grammatical == is_grammatical]
    #     counts = data_filtered.is_cr.value_counts(normalize=True)
    #     print(counts)
    #     for is_cr, count in zip(counts.index, counts):
    #         grammaticality = "ungrammatical" if is_grammatical == -1 else "grammatical" if is_grammatical == 1 else "ambiguous"
    #         entries.append({"is_cr": "clarification request" if is_cr else "other", "grammaticality": grammaticality, "proportion": count})
    #
    # df = pd.DataFrame(entries)
    # df.sort_values(by='is_cr', inplace=True)
    # sns.set_palette("Set2")
    # print(df)
    # fig = sns.barplot(df, x="grammaticality", y="proportion", hue="is_cr")
    # plt.xlabel("")
    # fig.legend_.set_title(None)
    # fig.get_figure().savefig("grammaticality2.png", dpi=300)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=CONVERSATIONS_ANNOTATED_DATA_FILE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
