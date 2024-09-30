import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from train_ppo import DEFAULT_MAX_GENERATION_LEN, compute_rewards, DEFAULT_MIN_GENERATION_LEN
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    reward_models = {}
    reward_model_tokenizers = {}
    for reward_model_path in args.reward_model_paths:
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        reward_model.eval()
        reward_models[reward_model_path] = reward_model
        reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        reward_model_tokenizers[reward_model_path] = reward_model_tokenizer

    def generate(model, tokenizer, batch_size, output_max_length):
        batch = dict()
        generation_kwargs = {
            "min_length": -1,
            "max_new_tokens": output_max_length,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        bos_tensor = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device)

        with torch.no_grad():
            batch["utts"] = model.generate(bos_tensor, **generation_kwargs)
        batch["utts_decoded"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in batch["utts"]]

        return batch

    def compute_scores(batch, reward_model, reward_model_tokenizer):
        texts_encoded = reward_model_tokenizer(batch["utts_decoded"], padding=True, return_tensors="pt")
        texts_encoded = texts_encoded.to(device)
        with torch.no_grad():
            value_model_outputs = reward_model(**texts_encoded)

        logits = value_model_outputs["logits"]
        scores = torch.argmax(logits, dim=1)
        scores = scores - 1
        return scores.cpu().numpy()

    all_scores = {path: [] for path in args.reward_model_paths}
    for _ in tqdm(range(args.num_batches)):
        batch = generate(model, tokenizer, args.batch_size, args.output_max_length)
        for reward_model_path in args.reward_model_paths:
            rewards = compute_rewards(
                batch["utts_decoded"], reward_models[reward_model_path], reward_model_tokenizers[reward_model_path],
                DEFAULT_MIN_GENERATION_LEN, DEFAULT_MAX_GENERATION_LEN, score_clip=None, length_reward_coef=None
            )
            all_scores[reward_model_path].extend(rewards)

    #TODO rank-order correlation -> spearman?
    print(f"Score for {args.model_path} (avg over {len(all_scores)} samples): {np.mean(all_scores):.3f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--reward_model_paths", type=str, nargs="+")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval(args)
