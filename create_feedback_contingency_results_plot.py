import argparse
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from pymer4.models import Lmer

from utilities import CONVERSATIONS_ANNOTATED_GRAMMAR_DATA_FILE
import seaborn as sns


def age_bin(age, num_months=3):
    return int((age + num_months / 2) / num_months) * num_months


def create_results_plot(args):
    print(f"\nStats for annotated data:")

    data = pd.read_csv(args.data_path)

    print("\n CR:")
    print("mean: ", data[data.is_cr == 1].is_grammatical.mean())
    print(data[data.is_cr == 1].is_grammatical.value_counts())

    print("\n Other:")
    print("mean: ", data[data.is_cr == 0].is_grammatical.mean())
    print(data[data.is_cr == 0].is_grammatical.value_counts())

    print(f"\nSamples of annotated data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(data.sample(50))

    data['is_cr'] = data.is_cr.astype(bool)
    assert data.is_grammatical.min() == -1 and data.is_grammatical.max() == 1
    assert data.is_cr.dtype == bool
    mod = Lmer('is_cr ~ is_grammatical + (1 | transcript_file)', family='binomial',
               data=data)
    print("=" * 50 + "\nGLM\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])

    data["age"] = data.age.apply(age_bin, num_months=6).astype(int)

    plt.figure()
    entries = []
    for transcript_file in tqdm(data.transcript_file.unique()):
        data_transcript = data[data.transcript_file == transcript_file]
        age = data_transcript.age.values[0]
        if len(data_transcript) > 100:
            for is_grammatical in [-1, 0, 1]:
                data_filtered = data_transcript[data_transcript.is_grammatical == is_grammatical]
                counts = data_filtered.is_cr.value_counts(normalize=True)
                for is_cr, count in zip(counts.index, counts):
                    grammaticality = "ungrammatical" if is_grammatical == -1 else "grammatical" if is_grammatical == 1 else "ambiguous"
                    entries.append(
                        {"is_cr": "clarification request" if is_cr else "other", "grammaticality": grammaticality,
                         "proportion": count, "transcript_file": transcript_file, "age": age})

    df = pd.DataFrame(entries)
    # df.sort_values(by='is_cr', inplace=True)
    # sns.set_palette("Set2")
    # fig = sns.barplot(df, x="grammaticality", y="proportion", hue="is_cr", errorbar="ci")
    # # fig = sns.barplot(df, x="is_cr", y="proportion", hue="grammaticality", errorbar="ci")
    #
    # plt.xlabel("Child utterance grammaticality")
    # plt.ylabel("Proportion")
    # fig.legend_.set_title("Caregiver response")
    # fig.get_figure().savefig("results/grammaticality_alt.svg", dpi=300)

    grammar_order = ['grammatical', 'ungrammatical']
    plt.figure(figsize=(6,5))
    data_filtered = df[df.grammaticality.isin(grammar_order)]
    data_filtered = data_filtered[data_filtered.is_cr == 'clarification request']
    data_filtered = data_filtered.rename(columns={"grammaticality": "Grammaticality"})
    sns.set_palette("Set2")
    fig = sns.barplot(data_filtered, x="Grammaticality", hue="Grammaticality", hue_order=grammar_order, y="proportion",
                      errorbar="ci")
    plt.ylabel("Proportion CRs")
    plt.ylim((0, 0.24))
    plt.tight_layout()
    fig.get_figure().savefig("results/grammaticality.png", dpi=300)

    plt.figure(figsize=(6,5))
    sns.set_palette("Set2")
    fig = sns.barplot(data_filtered, x="age", hue="Grammaticality", hue_order=grammar_order, y="proportion",
                      errorbar="ci")
    plt.ylabel("Proportion CRs")
    plt.xlabel("Age (months)")
    plt.ylim((0, 0.24))
    plt.tight_layout()
    fig.get_figure().savefig("results/grammaticality_by_age.png", dpi=300)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=CONVERSATIONS_ANNOTATED_GRAMMAR_DATA_FILE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results_plot(args)
