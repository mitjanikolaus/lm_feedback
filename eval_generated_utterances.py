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


def eval(model_path, eval_model_path, batch_size=1024, num_batches=10, output_max_length=DEFAULT_MAX_GENERATION_LEN):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    hparams = yaml.safe_load(open(os.path.join(eval_model_path, "hparams.yaml")))
    eval_model_tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    checkpoints = list(glob.glob(os.path.join(eval_model_path, "checkpoints", "epoch*.ckpt")))
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
    for _ in tqdm(range(num_batches)):
        batch = generate(model, tokenizer, batch_size, output_max_length)
        scores = compute_scores(batch, eval_model, eval_model_tokenizer)
        all_scores.extend(scores)
        # df = pd.DataFrame.from_dict({"utterances": batch['utts_decoded'], "scores": scores})
        # print(df)

    print(f"Score for {model_path}: {np.mean(all_scores):.2f}")


if __name__ == "__main__":
    batch_size = 100
    num_batches = 10
    eval_model_path = os.path.expanduser('~/data/childes_grammaticality/lightning_logs/version_6/')

    eval(model_path='lightning_logs/4w7g7e0i/ckpt_huggingface_best',
         eval_model_path=eval_model_path,
         batch_size=batch_size)

    eval(model_path='ckpts_ppo/best_blimp/1e6_seed_2/',
         eval_model_path=eval_model_path,
         batch_size=batch_size)

    # eval(model_path='ckpts_ppo/best_zorro/1e6_seed_2/',
    #      eval_model_path=eval_model_path,
    #      batch_size=batch_size)

    eval(model_path='ckpts_ppo/1e6_seed_2/',
         eval_model_path=eval_model_path,
         batch_size=batch_size)
