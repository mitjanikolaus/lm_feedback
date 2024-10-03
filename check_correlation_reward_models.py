import argparse
import itertools

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

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

    all_scores = {path: [] for path in args.reward_model_paths}
    utts_dict = {}
    for i in tqdm(range(args.num_batches)):
        batch = generate(model, tokenizer, args.batch_size, args.output_max_length)
        if i == 0:
            utts_dict = {"utterances": batch["utts_decoded"]}

        for r, reward_model_path in enumerate(args.reward_model_paths):
            utterances = batch["utts_decoded"]
            utt_lengths = [(utt != torch.tensor(tokenizer.pad_token_id)).sum() - 1 for utt in batch["utts"]]
            utts_contain_eos = [tokenizer.eos_token_id in resp for resp in batch["utts"]]

            utterances = [utt for utt, utt_len, contains_eos in zip(utterances, utt_lengths, utts_contain_eos) if contains_eos and (utt_len > DEFAULT_MIN_GENERATION_LEN)]
            utt_lengths = [utt_len for utt_len, contains_eos in zip(utt_lengths, utts_contain_eos) if contains_eos and (utt_len > DEFAULT_MIN_GENERATION_LEN)]
            utts_contain_eos = [contains_eos for contains_eos, utt_len in zip(utts_contain_eos, utt_lengths) if contains_eos and (utt_len > DEFAULT_MIN_GENERATION_LEN)]

            rewards = compute_rewards(
                utterances, utt_lengths, utts_contain_eos, reward_models[reward_model_path],
                reward_model_tokenizers[reward_model_path], DEFAULT_MIN_GENERATION_LEN, DEFAULT_MAX_GENERATION_LEN,
                score_clip=None, length_reward_coef=None
            )
            rewards = torch.stack(rewards).cpu().numpy()
            all_scores[reward_model_path].extend(rewards)

            if i == 0:
                utts_dict[f"scores_{r}"] = rewards

        if i == 0:
            pd.set_option('display.max_rows', 100)
            pd.set_option('display.width', 2000)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.precision', 2)
            pd.set_option("expand_frame_repr", False)

            shortest = np.min([len(utts_dict[f"scores_{r}"]) for r in range(len(args.reward_model_paths))])
            for key in utts_dict.keys():
                utts_dict[key] = utts_dict[key][:shortest]
            sample_df = pd.DataFrame.from_dict(utts_dict)
            print("\n")
            print(sample_df[sample_df.utterances.str.len() < 100].sort_values('scores_0'))

    for pair in itertools.combinations(all_scores.keys(), r=2):
        print(pair)
        correlation = spearmanr(all_scores[pair[0]], all_scores[pair[1]])
        print("spearman corr: ", correlation)
        correlation_pearson = pearsonr(all_scores[pair[0]], all_scores[pair[1]])
        print("pearson corr: ", correlation_pearson)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--reward_model_paths", type=str, nargs="+")

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval(args)
