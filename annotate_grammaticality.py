import argparse

import torch
from tqdm import tqdm
import pandas as pd

from eval import load_childes_grammar_model, compute_scores_childes_grammaticality
from utilities import CONVERSATIONS_ANNOTATED_DATA_FILE

tqdm.pandas()

BATCH_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    model, tokenizer = load_childes_grammar_model(args.eval_model_path)
    model.to(device)

    data = pd.read_csv(args.data_path)

    annotations = compute_scores_childes_grammaticality(data["utt_transcript_clean"], model, tokenizer)
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=CONVERSATIONS_ANNOTATED_DATA_FILE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
