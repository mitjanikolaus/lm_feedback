import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml

from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from train_ppo import DEFAULT_MAX_GENERATION_LEN, DEFAULT_MIN_GENERATION_LEN
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_generations(args):
    hparams = yaml.safe_load(open(os.path.join(args.eval_model_path, "hparams.yaml")))
    eval_model_tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    eval_model_checkpoints = list(glob.glob(os.path.join(args.eval_model_path, "checkpoints", "epoch*.ckpt")))
    assert len(eval_model_checkpoints) == 1, f"No or multiple checkpoints found in dir {args.eval_model_path}: {eval_model_checkpoints}"
    eval_model_checkpoint = eval_model_checkpoints[0]
    print(f"Model checkpoint: {eval_model_checkpoint}")
    eval_model = CHILDESGrammarModel.load_from_checkpoint(eval_model_checkpoint).to(device)
    eval_model.eval()

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

    def compute_scores(batch, value_model, value_model_tokenizer, tokenizer):
        utterances = batch["utts_decoded"]
        utt_lengths = [(utt != torch.tensor(tokenizer.pad_token_id)).sum() - 1 for utt in batch["utts"]]
        utterances = [utt for utt, utt_len in zip(utterances, utt_lengths) if utt_len > DEFAULT_MIN_GENERATION_LEN]

        utts_finished = [tokenizer.eos_token_id in utt for utt in batch["utts"]]
        utterances = [utt for utt, utt_finished in zip(utterances, utts_finished) if utt_finished]

        speaker_code = '[CHI]'
        utterances = [speaker_code+u for u in utterances]
        print(f"computing scores for {len(utterances)} utterances")
        texts_encoded = value_model_tokenizer(utterances, padding=True, return_tensors="pt")
        texts_encoded = texts_encoded.to(device)
        with torch.no_grad():
            value_model_outputs = value_model(**texts_encoded)

        logits = value_model_outputs["logits"]
        scores = torch.argmax(logits, dim=1)
        scores = scores - 1
        return scores.cpu().numpy(), utterances

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 200)

    # sanity check
    # test_utts = ["I like this.", "like this.", "What is this?", "What this?", "He like that.", "He likes that.",
    #              "They like him.", "Do this now.", "She likes himself.", "She likes herself.", "This is an apple.",
    #              "This is a apple.", "Do you want an banana?", "Do you want a banana?"]
    # batch = {"utts_decoded": test_utts}
    # scores = compute_scores(batch, eval_model, eval_model_tokenizer)
    # df = pd.DataFrame.from_dict({"utterances": batch['utts_decoded'], "scores": scores})
    # print("Sanity check for eval model: ")
    # print(df.sort_values("scores"))

    scores_dict = {}
    for model_path in args.model_paths:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()

        all_scores = []
        sample_df = None
        for i in range(args.num_batches):
            batch = generate(model, tokenizer, args.batch_size, args.output_max_length)
            scores, utterances = compute_scores(batch, eval_model, eval_model_tokenizer, tokenizer)
            all_scores.extend(scores)
            if i == 0:
                sample_df = pd.DataFrame.from_dict({"utterances": utterances, "scores": scores})
        print("\n\n")
        print(sample_df.sort_values("scores"))

        print(f"Score for {model_path} (avg over {len(all_scores)} samples): {np.mean(all_scores):.3f}")
        scores_dict[model_path] = np.mean(all_scores)

    print(scores_dict)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_paths", type=str, nargs="+")
    parser.add_argument("--eval_model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_generations(args)
