import argparse
import re

import numpy as np
import pandas as pd


def summarize_results(args):
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 3)
    pd.set_option("expand_frame_repr", False)

    results = pd.read_csv(args.results_file, index_col=0)
    metrics_base = ["zorro_filtered_childes", "blimp_filtered_childes", "scores_childes_grammar", "scores_gec"]
    # print(results[metrics])

    metrics_detailed = [c for c in results.columns if
               c.startswith("zorro_filtered_childes_phenomena/") or c.startswith("blimp_filtered_childes_phenomena/")]
    # metrics_detailed = [c for c in results.columns if
    #            c.startswith("zorro_filtered_childes/") or c.startswith("blimp_filtered_childes/")]

    for metrics in [metrics_detailed, metrics_base]:
        avg_results = []

        results_baseline = results[results.index.str.startswith("lightning_logs")]

        def create_avg_entry(df, name, metrics):
            std_values = [std if not np.isnan(std) else 0 for std in df[metrics].std(axis=0).values]
            keys = [k.replace("_filtered_childes", "") for k in df[metrics].columns]
            results_avg = {f"{key.replace('_', ' ')}": f"{mean:.3f} $\\pm$ {std:.3f}" for key, mean, std in
                           zip(keys, df[metrics].mean(axis=0).values, std_values)}

            if len(df) != 3:
                print(f"Expected 3 values, but got {len(df)} for {name}")

            item = {"model": name}
            item.update(results_avg)
            return item

        item = create_avg_entry(results_baseline, "baseline", metrics)
        avg_results.append(item)

        results_other = results[~results.index.str.startswith("lightning_logs")].copy()

        results_other["model_name"] = [re.sub(r'_seed_\d', '', x).replace("ckpts_ppo/1e6_", "").replace("/best_reward/", "")
                                       for x in results_other.index.values]

        for model_name in results_other.model_name.unique():
            results_model = results_other[results_other.model_name == model_name].copy()
            item = create_avg_entry(results_model, model_name, metrics)
            avg_results.append(item)

        avg_results = pd.DataFrame(avg_results)

        filter_models = ["baseline", "length_reward_0_entropy_001_lm_0", "length_reward_001_entropy_001_lm_001", "entropy_001"]
        avg_results = avg_results[avg_results.model.isin(filter_models)]

        avg_results.set_index("model", inplace=True)
        print(avg_results.T)
        print(avg_results.T.to_latex(escape=False))

        print("\n\n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=str, default="results.csv")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    summarize_results(args)
