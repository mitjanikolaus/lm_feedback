import argparse
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utilities import PPO_CKPTS_DIR, CKPT_DIR_BEST_REWARD


def summarize_results(args):
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 3)
    pd.set_option("expand_frame_repr", False)

    results = pd.read_csv(args.results_file, index_col=0)

    results_files: list[str] = list(glob.glob(os.path.join(PPO_CKPTS_DIR, '*', CKPT_DIR_BEST_REWARD, "results.p")))

    for file in results_files:
        data = pickle.load(open(file, "rb"))
        model = os.path.dirname(file)
        if model in results.index:
            print(f"skipping {model} as it is already in the results data")
        else:
            results.loc[model] = data

    metrics_base = ["zorro", "blimp", "grammaticality_childes", "grammaticality_gec"]

    metrics_detailed = [c.replace("_filtered_childes", "").replace("phenomena", "") for c in results.columns if
                        c.startswith("zorro_filtered_childes_phenomena/") or c.startswith(
                            "blimp_filtered_childes_phenomena/")]
    # metrics_detailed = [c for c in results.columns if
    #            c.startswith("zorro_filtered_childes/") or c.startswith("blimp_filtered_childes/")]

    for metrics in [metrics_detailed, metrics_base]:  # metrics_detailed
        avg_results = []

        results.loc[~results.index.str.startswith("lightning_logs/"), "model_name"] = results.loc[
            ~results.index.str.startswith("lightning_logs/")].index.map(
            lambda x: x.replace("ckpts_ppo/", "").replace("/best_reward/", "").replace(
                "/best_reward", "").replace("_", "-"))

        results.loc[results.index.str.startswith("lightning_logs/dkwnlvzm") | results.index.str.startswith(
            "lightning_logs/95f8k8zc") | results.index.str.startswith(
            "lightning_logs/967ufsfk"), 'model_name'] = '1e5-baseline'

        results.loc[results.index.str.startswith("lightning_logs/lb86b69m") | results.index.str.startswith(
            "lightning_logs/he3nnzld") | results.index.str.startswith(
            "lightning_logs/5z07yaqp"), 'model_name'] = '1e6-baseline'

        results.loc[
            results.index.str.startswith("lightning_logs/qpp61q7x") | results.index.str.startswith(
                "lightning_logs/m6s9vokb") | results.index.str.startswith(
                "lightning_logs/uu5rtja8"), 'model_name'] = '1e7-baseline'

        results["data size"] = results.model_name.map(lambda x: x.split("-")[0] + " words")
        results["model_name"] = results.model_name.map(
            lambda x: "-".join(x.split("-")[1:]).replace("seed-1-", "").replace("seed-2-", "").replace("seed-3-", ""))

        results.rename(columns=lambda x: x.replace("_filtered_childes", "").replace("phenomena", ""), inplace=True)

        results.replace({"entropy-001-lm-loss-001": "finetuned"}, inplace=True)
        filter_models = ["baseline", "finetuned"]
        results = results[results.model_name.isin(filter_models)]

        def create_avg_entry(df, name, data_size, metrics):
            std_values = [std if not np.isnan(std) else 0 for std in df[metrics].std(axis=0).values]
            keys = [k.replace('_', ' ') for k in df[metrics].columns]
            results_avg = {f"{key}": f"{mean:.3f} $\\pm$ {std:.3f}" for key, mean, std in
                           zip(keys, df[metrics].mean(axis=0).values, std_values)}

            if len(df) != 3:
                print(f"Expected 3 values, but got {len(df)} for {name}")

            # data_size = name.split("_")[0] + " words"
            # name = "-".join(name.split("_")[1:])

            item = {"model_name": name, "data size": data_size}
            item.update(results_avg)
            return item

        for model_name in results.model_name.unique():
            results_model = results[results.model_name == model_name].copy()
            for data_size in results_model["data size"].unique():
                results_model_ds = results_model[results_model["data size"] == data_size].copy()
                item = create_avg_entry(results_model_ds, model_name, data_size, metrics)
                avg_results.append(item)

        avg_results = pd.DataFrame(avg_results)

        results.sort_values(by=["data size", "model_name"], inplace=True)

        if metrics == metrics_base:
            print(avg_results.sort_values(by=["data size", "model_name"]).set_index(["data size", "model_name"]))

            results = results[["model_name", "data size"] + metrics]

            # results.rename(columns={"grammaticality_childes": "grammaticality\nchildes", "grammaticality_gec": "grammaticality\ngec"}, inplace=True)
            results = results.melt(id_vars=["data size", "model_name"], var_name="metric")


            plt.figure()
            # g = sns.FacetGrid(results, col="metric", col_wrap=2, height=5)  # , ylim=(0, 10)
            # g.map(sns.pointplot, "data size", "value", "model_name", errorbar="sd", linestyle="none",
            #       dodge=.3)  # order=[1, 2, 3]
            g = sns.catplot(x="data size", y="value", hue="model_name", data=results,
                            col="metric", col_wrap=2, height=2.5, aspect=2, sharey=True,
                            kind="point", linewidth=1.5, errorbar="sd")
            g.set_titles("{col_name}")
            g.set_axis_labels("Pretrainining data size", "")
            g.set(ylim=(0, 1))
            g._legend.remove()
            g.axes[-1].legend(loc='upper left', ncol=3, title="", bbox_to_anchor=(-0.3, 2.5))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
            # sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.55))
            plt.savefig("results.png", dpi=300)
            plt.show()


            # plt.figure()
            # # g = sns.FacetGrid(results, col="metric", col_wrap=2, height=5)  # , ylim=(0, 10)
            # # g.map(sns.pointplot, "data size", "value", "model_name", errorbar="sd", linestyle="none",
            # #       dodge=.3)  # order=[1, 2, 3]
            # g = sns.catplot(x="model_name", y="value", hue="data size", data=results,
            #                 col="metric", col_wrap=2, height=2.5, aspect=2, sharey=True,
            #                 kind="point", linewidth=1.5, errorbar="sd")
            # g.set_titles("{col_name}")
            #
            # g.set_axis_labels("", "")
            #
            # g.set(ylim=(0, 1))
            # g._legend.remove()
            # g.axes[-1].legend(loc='upper left', ncol=3, title="Pretraining data size", bbox_to_anchor=(-0.55, 2.8))
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
            # # sns.move_legend(g, "center right", bbox_to_anchor=(0.95, 0.55))
            # plt.savefig("results_alt.png", dpi=300)
            # plt.show()


            # plt.figure()
            # # g = sns.FacetGrid(results, col="metric", col_wrap=2, height=5)  # , ylim=(0, 10)
            # # g.map(sns.pointplot, "data size", "value", "model_name", errorbar="sd", linestyle="none",
            # #       dodge=.3)  # order=[1, 2, 3]
            # g = sns.catplot(x="data size", y="value", hue="model_name", data=results,
            #                 col="metric", col_wrap=2, height=3,
            #                 dodge=.2, kind="point", linestyle="none", linewidth=2, errorbar="sd")
            # # g.set_xticklabels(rotation=80)
            # # plt.tight_layout()
            # plt.savefig("results.png", dpi=300)
            # plt.show()

        else:
            print(avg_results.sort_values(by=["data size", "model_name"]).set_index(["data size", "model_name"]).T.to_latex())


        print("\n\n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=str, default="results.csv")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    summarize_results(args)
