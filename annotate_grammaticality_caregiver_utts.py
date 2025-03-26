import argparse
import math

import torch
import pandas as pd

from eval import load_childes_grammar_model, compute_scores_childes_grammaticality
from utilities import CONVERSATIONS_ANNOTATED_DATA_FILE

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
        is_grammatical = compute_scores_childes_grammaticality(data_batch["response_transcript_clean"], model, tokenizer)
        annotations.extend(is_grammatical)
    data["is_grammatical"] = annotations

    out_path = args.data_path.replace(".csv", "_annotated_grammar_caregiver_utts.csv")
    print(f"Saving results to {out_path}")
    data.to_csv(out_path, index=False)

    print(f"Mean grammaticality: {data.is_grammatical.values.mean()}")
    print(f"Grammaticality counts: {data.is_grammatical.values.count()}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=CONVERSATIONS_ANNOTATED_DATA_FILE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
