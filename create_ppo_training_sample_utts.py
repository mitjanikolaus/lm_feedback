import pandas as pd

BOLD = '\\textbf{'
UNDERLINE = '\\underline{'
END = '}'
RANDOM_STATE = 4


def print_table():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 3)
    pd.set_option("expand_frame_repr", False)

    results = pd.read_csv('data/ppo_sample_utts.csv')
    for step in results.step.unique():
        print(step)
        results_step = results[results['step'] == step]
        sample = results_step.sample(10, random_state=RANDOM_STATE)

        def assemble_utt(row):
            completion = row['utterance'].replace(row['query'], '', 1)
            assert completion != row['utterance']
            return UNDERLINE + row['query'] + END + completion

        sample['Utterance'] = sample.apply(assemble_utt, axis=1)
        sample['Reward'] = sample.reward.round(2).astype(str)
        print(sample[['Utterance', 'Reward']].to_latex(index=False, escape=False))


if __name__ == "__main__":
    print_table()
