import argparse
import os

import pandas as pd
import torch.cuda
from transformers import AutoTokenizer, AutoModelForMaskedLM

from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked import configs
from unmasked.utils import calc_accuracy_from_scores

LOWER_CASE = False

SCORING_METHOD = "mlm"  # holistic
TEST_SUITE_NAME = "zorro"

if TEST_SUITE_NAME == 'blimp':
    num_expected_scores = 2000
elif TEST_SUITE_NAME == 'zorro':
    num_expected_scores = 4000

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval():
    args = parse_args()

    if SCORING_METHOD == 'mlm':
        score_model_on_paradigm = mlm_score_model_on_paradigm
    elif SCORING_METHOD == 'holistic':
        score_model_on_paradigm = holistic_score_model_on_paradigm
    else:
        raise AttributeError('Invalid scoring_method.')

    # load from repo
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              add_prefix_space=True,  # this must be True for BabyBERTa
                                              )
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    model.eval()
    model.to(device)

    # for each paradigm in test suite
    accuracies = []
    for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):
        print(f"Scoring {path_paradigm.name:<60} with {args.model:<40} and method={SCORING_METHOD:30}", end="")
        scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=LOWER_CASE)

        assert len(scores) == num_expected_scores

        accuracy = calc_accuracy_from_scores(scores, SCORING_METHOD)
        print(f"{accuracy:.2f}")

        paradigm = path_paradigm.parts[-1]
        accuracies.append({"paradigm": paradigm, "acc": accuracy})

    acc_df = pd.DataFrame(accuracies)
    print(f'Overall accuracy: {acc_df.acc.mean():.2f}')

    out_dir = os.path.join("results", args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    acc_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="phueb/BabyBERTa-1",
    )

    args = argparser.parse_args()

    return args


if __name__ == '__main__':
    eval()

