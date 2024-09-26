import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from train_ppo import DEFAULT_MAX_GENERATION_LEN
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    hparams = yaml.safe_load(open(os.path.join(args.eval_model_path, "hparams.yaml")))
    eval_model_tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    checkpoints = list(glob.glob(os.path.join(args.eval_model_path, "checkpoints", "epoch*.ckpt")))
    assert len(checkpoints) == 1, f"No or multiple checkpoints found: {checkpoints}"
    checkpoint = checkpoints[0]
    print(f"Model checkpoint: {checkpoint}")
    eval_model = CHILDESGrammarModel.load_from_checkpoint(checkpoint,
                                                          strict=False)  # TODO upgrade huggingface/torch version?
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

    def compute_scores(batch, value_model, value_model_tokenizer):
        # TODO ignore sentences that don't contain EOS token?!

        texts_encoded = value_model_tokenizer(batch["utts_decoded"], padding=True, return_tensors="pt")
        texts_encoded = texts_encoded.to(device)
        with torch.no_grad():
            value_model_outputs = value_model(**texts_encoded)

        logits = value_model_outputs["logits"]
        scores = torch.argmax(logits, dim=1)
        scores = scores - 1
        return scores.cpu().numpy()

    all_scores = []
    for i in tqdm(range(args.num_batches)):
        batch = generate(model, tokenizer, args.batch_size, args.output_max_length)
        scores = compute_scores(batch, eval_model, eval_model_tokenizer)
        all_scores.extend(scores)
        if i == 0:
            df = pd.DataFrame.from_dict({"utterances": batch['utts_decoded'], "scores": scores})
            print(df.sort_values("scores"))

    print(f"Score for {args.model_path}: {np.mean(all_scores):.3f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--eval_model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval(args)
