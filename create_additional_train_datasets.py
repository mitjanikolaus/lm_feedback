import os
from collections import Counter

import pandas as pd
from utils import CHILDES_RL_DATA_FILE, DATA_DIR


if __name__ == '__main__':
    conversations = pd.read_csv(CHILDES_RL_DATA_FILE)

    convs_cr = conversations[conversations.response_is_clarification_request].copy()

    convs_cr["transcript_clean"] = convs_cr["utt_transcript_clean"]
    convs_cr = convs_cr[["transcript_clean"]]
    ref_num_words = Counter()
    for sent in convs_cr.transcript_clean.values:
        length = len(sent.split(" "))
        if length > 40:
            length = -1  # catch-all for long sentences
        ref_num_words.update([length])

    print(f"Number of utterances: {len(ref_num_words)}")
    print(f"Most common num of words: {ref_num_words.most_common(10)}")
    convs_cr.to_csv(os.path.join(DATA_DIR, "child_cr.csv"), index=False)

    convs_non_cr = conversations[~conversations.response_is_clarification_request].copy()
    convs_non_cr = convs_non_cr.sample(frac=1, random_state=1).reset_index(drop=True)   # Shuffle data
    convs_non_cr["num_words"] = convs_non_cr.utt_transcript_clean.apply(lambda x: len(x.split(" ")))
    convs_non_cr["num_words"] = convs_non_cr.num_words.apply(lambda x: -1 if x > 40 else x)
    convs_non_cr_filtered = []
    for num_words, count in ref_num_words.items():
        convs_non_cr_target_num_words = convs_non_cr[convs_non_cr.num_words == num_words]
        convs_non_cr_filtered.append(convs_non_cr_target_num_words.sample(count))
    convs_non_cr_filtered = pd.concat(convs_non_cr_filtered)
    convs_non_cr_filtered["transcript_clean"] = convs_non_cr_filtered["utt_transcript_clean"]
    convs_non_cr_filtered = convs_non_cr_filtered[["transcript_clean"]]

    print(f"Number of utts with CR response: {len(convs_cr)}")
    print(f"Number of utts without CR response: {len(convs_non_cr_filtered)}")

    convs_non_cr_filtered.to_csv(os.path.join(DATA_DIR, "child_non_cr.csv"), index=False)
