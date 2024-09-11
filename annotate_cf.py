import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from data import preprocess_childes_utterance
from train_cf_classifier import DEFAULT_MAX_LENGTH, CFClassifierDataCollatorWithPadding
from utils import CHILDES_RL_DATA_FILE

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

tqdm.pandas()

BATCH_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data[["utt_transcript_clean", "response_transcript_clean"]]

    data["utt_transcript_clean"] = data["utt_transcript_clean"].apply(preprocess_childes_utterance)
    data["response_transcript_clean"] = data["response_transcript_clean"].apply(preprocess_childes_utterance)

    return data


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
    )
    model.eval()
    model.to(device)

    data = load_data(args.data_path)

    data_to_annotate = data
    if args.target_column == "is_cr":
        data_to_annotate = data[data.response_transcript_clean.str.endswith("?")].copy()

    dataset = Dataset.from_pandas(data_to_annotate)
    dataset.set_format(type="torch")

    def preprocess_function(samples):
        texts = [utt + tokenizer.sep_token + resp for utt, resp in
                 zip(samples["utt_transcript_clean"], samples["response_transcript_clean"])]
        tokenized = tokenizer(texts, truncation=True, max_length=DEFAULT_MAX_LENGTH)

        return tokenized

    # Preprocess the dataset and truncate examples that are longer than args.max_length
    dataset = dataset.map(
        preprocess_function,
        batched=True,
    )

    data_collator = CFClassifierDataCollatorWithPadding(tokenizer, max_length=DEFAULT_MAX_LENGTH)
    dloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    all_preds = []
    for batch in tqdm(dloader):
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_dict=True,
            )
            preds_proba = F.sigmoid(out["logits"].squeeze())
            preds = (preds_proba > 0.5)
            all_preds.extend(preds.cpu().numpy().astype(int))

    data[args.target_column] = 0
    data.loc[data_to_annotate.index.values, args.target_column] = all_preds
    out_path = args.data_path.replace(".csv", "_annotated.csv")
    print(f"Saving results to {out_path}")
    data.to_csv(out_path, index=False)

    print(f"\nStats for annotated data:")
    print(data.loc[data_to_annotate.index.values, args.target_column].value_counts())

    annotated_data = data.iloc[data_to_annotate.index.values].copy()
    print(f"\nSamples of annotated data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(annotated_data[annotated_data[args.target_column] == 0].sample(20))
    print(annotated_data[annotated_data[args.target_column] == 1].sample(20))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--data_path", type=str, default=CHILDES_RL_DATA_FILE)
    parser.add_argument("--target_column", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
