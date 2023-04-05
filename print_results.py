import argparse
import json
import os.path


def gather_and_print_results(args):
    results = {}
    for task in ["anaphor_agreement", "argument_structure", "binding", "control_raising", "determiner_noun_agreement",
                 "ellipsis", "filler_gap", "irregular_forms", "island_effects", "npi_licensing", "quantifiers",
                 "subject_verb_agreement"]:
        res_file = f"{args.ckpt}/ckpt_huggingface/zeroshot/{task}/eval_results.json"
        if not os.path.isfile(res_file):
            print("result missing: ", task)
            continue
        result = json.load(open(res_file))
        results[task] = result["eval_accuracy"]
        print(f"{task:<25}: {result['eval_accuracy']:.2f}")

    print(results)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ckpt",
        type=str,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    gather_and_print_results(args)
