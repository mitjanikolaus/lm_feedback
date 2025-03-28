import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from utilities import PPO_CKPTS_DIR, CKPT_DIR_BEST_REWARD


def summarize_results(args):
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 3)
    pd.set_option("expand_frame_repr", False)

    results = pd.read_csv(args.results_file, index_col=0)

    def clean_model_name(name):
        if name.startswith("lightning_logs/dkwnlvzm") or name.startswith(
                "lightning_logs/95f8k8zc") or name.startswith("lightning_logs/967ufsfk"):
            return '0.1M-baseline'
        if name.startswith("lightning_logs/lb86b69m") or name.startswith(
                "lightning_logs/he3nnzld") or name.startswith("lightning_logs/5z07yaqp"):
            return '1M-baseline'
        if name.startswith("lightning_logs/qpp61q7x") or name.startswith(
                "lightning_logs/m6s9vokb") or name.startswith("lightning_logs/uu5rtja8"):
            return '10M-baseline'

        name = name.replace("ckpts_ppo/", "").replace("/best_reward/", "").replace("/best_reward", "").replace("_", "-").replace("1e5", "0.1M").replace("1e6", "1M").replace("1e7", "10M")
        return "-".join(name.split("-")).replace("seed-1-", "").replace("seed-2-", "").replace("seed-3-", "")

    results["model_name"] = results.index.map(clean_model_name)

    results_files: list[str] = list(glob.glob(os.path.join(PPO_CKPTS_DIR, '*', CKPT_DIR_BEST_REWARD, "results.p")))

    for file in results_files:
        data = pickle.load(open(file, "rb"))
        key = os.path.dirname(file)
        data['model_name'] = clean_model_name(key)
        if key in results.index:
            print(f"skipping {data['model_name']} as it is already in the results data")
        else:
            results.loc[key] = data

    results["data_size"] = results.model_name.map(lambda x: x.split("-")[0] + " words")
    results["model_name"] = results.model_name.map(lambda x: "-".join(x.split("-")[1:]))
    results.rename(columns=lambda x: x.replace("_filtered_childes", ""), inplace=True)

    results.replace({"entropy-001-lm-loss-001": "finetuned"}, inplace=True)
    results.replace({"reward-topline-entropy-001-lm-loss-001": "topline"}, inplace=True)

    filter_models = ["baseline", "finetuned", "topline"]
    results = results[results.model_name.isin(filter_models)]

    metrics_base = ["Zorro", "Blimp", "Grammaticality", "Grammaticality (error correction)"]
    metrics_detailed = [c.replace("_phenomena", "") for c in results.columns if c.startswith("zorro_phenomena/") or c.startswith("blimp_phenomena/")]
    results.rename(columns=lambda x: x.replace("_phenomena", ""), inplace=True)
    results.model_name = results.model_name.apply(lambda x: x[0].upper()+x[1:])
    results.rename(
        columns={"grammaticality_childes": "Grammaticality", "grammaticality_gec": "Grammaticality (error correction)",
                 "zorro": "Zorro", "blimp": "Blimp"},
        inplace=True)

    # metrics_detailed = [c for c in results.columns if
    #            c.startswith("zorro_filtered_childes/") or c.startswith("blimp_filtered_childes/")]

    for metrics in [metrics_detailed, metrics_base]:  # metrics_detailed
        avg_results = []

        def create_avg_entry(df, name, data_size, metrics):
            std_values = [std if not np.isnan(std) else 0 for std in df[metrics].std(axis=0).values]
            keys = [k.replace('_', ' ') for k in df[metrics].columns]
            results_avg = {f"{key}": f"{mean:.2f} $\\pm$ {std:.2f}" for key, mean, std in
                           zip(keys, df[metrics].mean(axis=0).values, std_values)}

            if len(df) != 3:
                print(f"Expected 3 values, but got {len(df)} for {name}")

            # data_size = name.split("_")[0] + " words"
            # name = "-".join(name.split("_")[1:])

            item = {"model_name": name, "data_size": data_size}
            item.update(results_avg)
            return item

        for model_name in results.model_name.unique():
            results_model = results[results.model_name == model_name].copy()
            for data_size in results_model["data_size"].unique():
                results_model_ds = results_model[results_model["data_size"] == data_size].copy()
                item = create_avg_entry(results_model_ds, model_name, data_size, metrics)
                avg_results.append(item)

        avg_results = pd.DataFrame(avg_results)

        results.sort_values(by=["data_size", "model_name"], inplace=True)

        if metrics == metrics_base:
            print(avg_results.sort_values(by=["data_size", "model_name"]).set_index(["data_size", "model_name"]))
            print(avg_results.sort_values(by=["data_size", "model_name"]).set_index(
                ["data_size", "model_name"]).to_latex())

            results = results[["model_name", "data_size"] + metrics]

            results = results[results.model_name.isin([args.plot_comparison_model_1, args.plot_comparison_model_2])].copy()
            results_melted = results.melt(id_vars=["data_size", "model_name"], var_name="metric")

            plt.figure()
            metric_order = ["Grammaticality", "Grammaticality (error correction)", "Zorro", "Blimp"]
            data_size_order = ['0.1M words', '1M words', '10M words']
            g = sns.catplot(x="data_size", order=data_size_order, y="value", hue="model_name", data=results_melted,
                            col="metric", col_order=metric_order,
                            col_wrap=2, height=2.5, aspect=2, sharey=False,
                            kind="point", linewidth=1.5, errorbar="sd")

            print("t-tests:")
            for data_size_idx, data_size in enumerate(data_size_order):
                print(data_size)
                for metric in metrics:
                    baseline_scores = \
                    results[(results["data_size"] == data_size) & (results["model_name"] == args.plot_comparison_model_1)][metric].values
                    finetuned_scores = \
                    results[(results["data_size"] == data_size) & (results["model_name"] == args.plot_comparison_model_2)][metric].values
                    p_value = ttest_ind(baseline_scores, finetuned_scores).pvalue
                    print(f"{metric}: {p_value:5f}")

                    max_value = np.max(np.concatenate((finetuned_scores, baseline_scores)))
                    offset = 0.01
                    if p_value < 0.001:
                        g.axes_dict[metric].text(data_size_idx, max_value + offset, '***', ha='center')
                    elif p_value < 0.01:
                        g.axes_dict[metric].text(data_size_idx, max_value + offset, '**', ha='center')
                    elif p_value < 0.05:
                        g.axes_dict[metric].text(data_size_idx, max_value + offset, '*', ha='center')

            g.set_titles("{col_name}")
            g.set_axis_labels("Pretrainining data size", "")
            for metric, ax in zip(metric_order, g.axes):
                if metric == 'Grammaticality':
                    ax.set_ylim((-1, 1))
                else:
                    ax.set_ylim((0, 1))

            g._legend.remove()
            g.axes[-1].legend(loc='upper left', ncol=3, title="", bbox_to_anchor=(-0.3, 2.6))
            plt.subplots_adjust(left=0.05, right=1, top=0.8, bottom=0.1, hspace=0.3)
            # sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.55))
            plt.savefig("results/results.svg", dpi=300)
            plt.show()

            # plt.figure()
            # # g = sns.FacetGrid(results, col="metric", col_wrap=2, height=5)  # , ylim=(0, 10)
            # # g.map(sns.pointplot, "data_size", "value", "model_name", errorbar="sd", linestyle="none",
            # #       dodge=.3)  # order=[1, 2, 3]
            # g = sns.catplot(x="model_name", y="value", hue="data_size", data=results,
            #                 col="metric", col_wrap=2, height=2.5, aspect=2, sharey=True,
            #                 kind="point", linewidth=1.5, errorbar="sd")
            # g.set_titles("{col_name}")
            #
            # g.set_axis_labels("", "")
            #
            # g.set(ylim=(0, 1))
            # g._legend.remove()
            # g.axes[-1].legend(loc='upper left', ncol=3, title="Pretraining data_size", bbox_to_anchor=(-0.55, 2.8))
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
            # # sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.55))
            # plt.savefig("results_alt.png", dpi=300)
            # plt.show()

            # plt.figure()
            # # g = sns.FacetGrid(results, col="metric", col_wrap=2, height=5)  # , ylim=(0, 10)
            # # g.map(sns.pointplot, "data_size", "value", "model_name", errorbar="sd", linestyle="none",
            # #       dodge=.3)  # order=[1, 2, 3]
            # g = sns.catplot(x="data_size", y="value", hue="model_name", data=results,
            #                 col="metric", col_wrap=2, height=3,
            #                 dodge=.2, kind="point", linestyle="none", linewidth=2, errorbar="sd")
            # # g.set_xticklabels(rotation=80)
            # # plt.tight_layout()
            # plt.savefig("results.png", dpi=300)
            # plt.show()

        else:
            print(avg_results.sort_values(by=["data_size", "model_name"]).set_index(
                ["data_size", "model_name"]).T.to_latex())

        print("\n\n")

        # print(pd.read_csv("sample_utts_baseline.csv", index_col=0).sort_values(
        #     by="score_childes_grammaticality").to_latex(index=False))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=str, default="results.csv")

    parser.add_argument("--plot_comparison_model_1", type=str, default="Baseline")
    parser.add_argument("--plot_comparison_model_2", type=str, default="Finetuned")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    summarize_results(args)
