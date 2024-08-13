import argparse
import math
import os
import warnings

import torch
from tqdm import tqdm
import pandas as pd

from utils import CHILDES_LM_DATA_FILE

tqdm.pandas()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from lm_eval import evaluator

os.environ["WANDB_PROJECT"] = "lm_feedback_ppo"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = "ckpts_trl"


def build_policy_trainer_dataset(tokenizer, query_data_path, input_min_text_length=1, input_max_text_length=4):
    assert tokenizer.pad_token == tokenizer.eos_token

    data_queries = pd.read_csv(query_data_path)
    # data_fb = data_fb.iloc[:1000]

    if "utt_transcript_clean" in data_queries.columns:
        data_queries["transcript_clean"] = data_queries["utt_transcript_clean"]
        del data_queries["utt_transcript_clean"]

    ds = Dataset.from_pandas(data_queries)

    ds.remove_columns(["response_is_clarification_request", "response_is_acknowledgement"])

    ds = ds.filter(lambda x: len(x["transcript_clean"]) > 10, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["transcript_clean"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=10)
    ds.set_format(type="torch")
    return ds


def eval_babylm(model, model_args, tasks, ppo_trainer, device, eval_batch_size=1024):
    print("Evaluating babylm metrics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=tasks,
            batch_size=eval_batch_size,
            device=f"cuda:{device}",
            cache_requests=True,
        )

    results = {key.replace("_", "/"): val["acc,none"] for key, val in out["results"].items() if "acc,none" in val}
    ppo_trainer.accelerator.log(results)


def main(args):
    config = PPOConfig(
        model_name="childes-gpt",
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        exp_name=args.exp_name,
        seed=args.seed,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_model)
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model)

    dataset = build_policy_trainer_dataset(tokenizer, query_data_path=args.query_data_path)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_model)

    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    value_model = AutoModelForSequenceClassification.from_pretrained(args.value_model)
    value_model_tokenizer = AutoTokenizer.from_pretrained(args.value_model)

    output_min_length = 4
    output_max_length = 20
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": args.generation_top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Get completion from gpt2
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        texts_encoded = value_model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=output_max_length+10)
        value_model_outputs = value_model(**texts_encoded)
        rewards = value_model_outputs.logits.squeeze()
        rewards = [torch.tensor(r.item()) for r in rewards]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if epoch % 25 == 0:
            model.save_pretrained(os.path.join(CKPT_DIR, args.exp_name))
            tokenizer.save_pretrained(os.path.join(CKPT_DIR, args.exp_name))

            eval_babylm(model="hf", model_args=f"pretrained={os.path.join(CKPT_DIR, args.exp_name)}", tasks=["zorro", "blimp_filtered"],
                        ppo_trainer=ppo_trainer, device=ppo_trainer.accelerator.device.index)

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--policy_model",
        type=str,
    )
    argparser.add_argument(
        "--value_model",
        type=str,
    )
    argparser.add_argument(
        "--query_data_path",
        type=str,
        default=CHILDES_LM_DATA_FILE
    )
    argparser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
    )
    argparser.add_argument(
        "--exp_name",
        type=str,
        default="test",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    argparser.add_argument(
        "--generation_top_p",
        type=float,
        default=1.0,
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    argparser.add_argument(
        "--mini_batch_size",
        type=int,
        default=512,
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1.41e-5,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    args = parse_args()
    main(args)
