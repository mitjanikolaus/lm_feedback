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
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    fig = sns.barplot(df, x="grammaticality", y="proportion", hue="is_cr", errorbar="ci")
    # fig = sns.barplot(df, x="is_cr", y="proportion", hue="grammaticality", errorbar="ci")

    plt.xlabel("Child utterance grammaticality")
    plt.ylabel("Proportion")
    fig.legend_.set_title("Caregiver response")
    fig.get_figure().savefig("results/grammaticality.svg", dpi=300)

    grammar_order = ['grammatical', 'ambiguous', 'ungrammatical']
    plt.figure()
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    fig = sns.barplot(df[df.is_cr == 'clarification request'], x="grammaticality", hue="grammaticality", hue_order=grammar_order, y="proportion", errorbar="ci")

    plt.ylabel("Proportion CRs")
    fig.get_figure().savefig("results/grammaticality_alt.png", dpi=300)

    plt.figure()
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    fig = sns.barplot(df[df.is_cr == 'clarification request'], x="age", hue="grammaticality", hue_order=['grammatical', 'ambiguous', 'ungrammatical'], y="proportion",
                      errorbar="ci")
    plt.ylabel("Proportion CRs")
    fig.get_figure().savefig("results/grammaticality_by_age.png", dpi=300)


    plt.figure()
    entries = []
    for transcript_file in tqdm(data.transcript_file.unique()):
        data_transcript = data[data.transcript_file == transcript_file]
        if len(data_transcript) > 100:
            for is_cr in [0, 1]:
                data_filtered = data_transcript[data_transcript.is_cr == is_cr]
                counts = data_filtered.is_grammatical.value_counts(normalize=True)
                for is_grammatical, count in zip(counts.index, counts):
                    grammaticality = "ungrammatical" if is_grammatical == -1 else "grammatical" if is_grammatical == 1 else "ambiguous"
                    entries.append(
                        {"is_cr": "clarification request" if is_cr else "other", "grammaticality": grammaticality,
                         "proportion": count, "transcript_file": transcript_file})

    df = pd.DataFrame(entries)
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    fig = sns.barplot(df, x="grammaticality", y="proportion", hue="is_cr", errorbar="ci")
    plt.xlabel("Child utterance grammaticality")
    plt.ylabel("Proportion")
    fig.legend_.set_title("Caregiver response")
    fig.get_figure().savefig("grammaticality.svg", dpi=300)

    plt.figure()
    entries = []
    for transcript_file in tqdm(data.transcript_file.unique()):
        data_transcript = data[data.transcript_file == transcript_file]
        if len(data_transcript) > 100:
            for is_grammatical in [-1, 0, 1]:
                data_filtered = data_transcript[data_transcript.is_grammatical == is_grammatical]
                counts = data_filtered.is_cr.value_counts(normalize=True)
                for is_cr, count in zip(counts.index, counts):
                    grammaticality = "ungrammatical" if is_grammatical == -1 else "grammatical" if is_grammatical == 1 else "ambiguous"
                    entries.append({"is_cr": "clarification request" if is_cr else "other response", "grammaticality": grammaticality, "proportion": count, "transcript_file": transcript_file})

    df = pd.DataFrame(entries)
    df.sort_values(by='is_cr', inplace=True)
    sns.set_palette("Set2")
    print(df)
    fig = sns.barplot(df, x="is_cr", y="proportion", hue="grammaticality")
    plt.xlabel("Caregiver response")
    plt.xlabel("Proportion")
    fig.legend_.set_title("Child utterance")
    fig.get_figure().savefig("results/grammaticality2.svg", dpi=300)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=CONVERSATIONS_ANNOTATED_GRAMMAR_DATA_FILE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_results_plot(args)
