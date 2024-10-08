import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

from lm_eval import evaluator
from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from train_lm import DEFAULT_EVAL_METRICS
from train_ppo import DEFAULT_MAX_GENERATION_LEN, DEFAULT_MIN_GENERATION_LEN
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer

from utilities import parse_babylm_metrics_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_babylm_metrics(ckpt_dir, eval_batch_size=1024):
    model_args = f"pretrained={ckpt_dir},add_bos_token=True"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=DEFAULT_EVAL_METRICS,
            batch_size=eval_batch_size,
            device=f"cuda:{device}",
            cache_requests=True,
        )

    results = parse_babylm_metrics_results(out)
    return results


def compute_scores_childes_grammaticality(utterances, value_model, value_model_tokenizer):
    speaker_code = '[CHI]'
    utterances = [speaker_code + u for u in utterances]
    texts_encoded = value_model_tokenizer(utterances, padding=True, return_tensors="pt")
    texts_encoded = texts_encoded.to(device)
    with torch.no_grad():
        value_model_outputs = value_model(**texts_encoded)

    logits = value_model_outputs["logits"]
    scores = torch.argmax(logits, dim=1)
    scores = scores - 1
    return scores.cpu().numpy()


def compute_scores_gec(utterances, gec_model, gec_model_tokenizer, max_length=DEFAULT_MAX_GENERATION_LEN*2, num_beams=5):
    utterances_gec = ['gec: ' + u for u in utterances]
    tokenized_sentences = gec_model_tokenizer(utterances_gec, max_length=max_length, truncation=True,
                                              padding='max_length',
                                              return_tensors='pt').to(device)
    with torch.no_grad():
        output = gec_model.generate(
            input_ids=tokenized_sentences.input_ids,
            attention_mask=tokenized_sentences.attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        corrected_sentences = gec_model_tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    scores = np.array([utt.lower().replace(",", "") == utt_corrected.lower().replace(",", "") for utt, utt_corrected in
                       zip(utterances, corrected_sentences)]).astype(int)
    return scores


def compute_scores(batch, childes_grammar_model, childes_grammar_model_tokenizer, gec_model, gec_model_tokenizer,
                   tokenizer):
    utterances = batch["utts_decoded"]
    utt_lengths = [(utt != torch.tensor(tokenizer.pad_token_id)).sum() - 1 for utt in batch["utts"]]
    utterances = [utt for utt, utt_len in zip(utterances, utt_lengths) if utt_len > DEFAULT_MIN_GENERATION_LEN]

    utts_finished = [tokenizer.eos_token_id in utt for utt in batch["utts"]]
    utterances = [utt for utt, utt_finished in zip(utterances, utts_finished) if utt_finished]

    if len(utterances) == 0:
        return [], []

    scores_gec = compute_scores_gec(utterances, gec_model, gec_model_tokenizer)
    scores_childes_grammar = compute_scores_childes_grammaticality(utterances, childes_grammar_model,
                                                                   childes_grammar_model_tokenizer)

    return scores_childes_grammar, scores_gec, utterances


def eval_models(args):
    hparams = yaml.safe_load(open(os.path.join(args.eval_model_path, "hparams.yaml")))
    childes_grammar_model_tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    eval_model_checkpoints = list(glob.glob(os.path.join(args.eval_model_path, "checkpoints", "epoch*.ckpt")))
    assert len(
        eval_model_checkpoints) == 1, f"No or multiple checkpoints found in dir {args.eval_model_path}: {eval_model_checkpoints}"
    eval_model_checkpoint = eval_model_checkpoints[0]
    print(f"Model checkpoint: {eval_model_checkpoint}")
    childes_grammar_model = CHILDESGrammarModel.load_from_checkpoint(eval_model_checkpoint).to(device)
    childes_grammar_model.eval()

    gec_model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small").to(device)
    gec_model_tokenizer = T5Tokenizer.from_pretrained('t5-small')

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

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 3)
    pd.set_option("expand_frame_repr", False)

    all_results = []
    for model_path in args.model_paths:
        results = eval_babylm_metrics(model_path)

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()

        # sanity check
        # test_utts = ["I like this.", "like this.", "What is this?", "What this?", "He like that.", "He likes that.",
        #              "They like him.", "Do this now.", "She likes himself.", "She likes herself.", "This is an apple.",
        #              "This is a apple.", "Do you want an banana?", "Do you want a banana?"]
        # batch = {"utts_decoded": test_utts}
        # scores, scores_gec, utterances = compute_scores(batch, childes_grammar_model, childes_grammar_model_tokenizer,
        #                                                 gec_model, gec_model_tokenizer, tokenizer)
        # df = pd.DataFrame.from_dict({"utterances": batch['utts_decoded'], "scores": scores, "scores_gec": scores_gec})
        # print("Sanity check for eval model: ")
        # print(df.sort_values("scores"))

        all_scores_childes_grammar = []
        all_scores_gec = []
        sample_df = None
        for i in range(args.num_batches):
            batch = generate(model, tokenizer, args.batch_size, args.output_max_length)
            scores_childes_grammar, scores_gec, utterances = compute_scores(batch, childes_grammar_model,
                                                            childes_grammar_model_tokenizer, gec_model,
                                                            gec_model_tokenizer, tokenizer)
            all_scores_childes_grammar.extend(scores_childes_grammar)
            all_scores_gec.extend(scores_gec)
            if i == 0:
                sample_df = pd.DataFrame.from_dict(
                    {"utterances": utterances, "scores_childes_grammaticality": scores_childes_grammar, "scores_gec": scores_gec})
        print("\n\n")
        print(sample_df.sort_values("scores_childes_grammaticality"))

        print(
            f"Childes grammaticality scores for {model_path} (avg over {len(all_scores_childes_grammar)} samples): {np.mean(all_scores_childes_grammar):.3f} | "
            f"scores_gec: {np.mean(all_scores_gec):.3f}"
        )
        results.update({"model": model_path, "scores_childes_grammar": np.mean(all_scores_childes_grammar), "scores_gec": np.mean(all_scores_gec)})
        all_results.append(results)

    all_results = pd.DataFrame(all_results)
    all_results.set_index("model", inplace=True)
    all_results.to_csv("results.csv", index=True, index_label="model")
    print(all_results[["zorro_filtered_childes", "blimp_filtered_childes", "scores_childes_grammar", "scores_gec"]])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--eval_model_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_models(args)
