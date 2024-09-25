import os
from collections import Counter

import pandas as pd
from utilities import CHILDES_RL_DATA_FILE, DATA_DIR

LONG_SENTENCE_LENGTH = 30

if __name__ == '__main__':
    data_path = CHILDES_RL_DATA_FILE
    min_age = data_path.split("min_age_")[1].split(".")[0]
    conversations = pd.read_csv(data_path)

    convs_cr = conversations[conversations.response_is_clarification_request].copy()

    convs_cr["transcript_clean"] = convs_cr["utt_transcript_clean"]
    convs_cr = convs_cr[["transcript_clean"]]
    ref_num_words = Counter()
    for sent in convs_cr.transcript_clean.values:
        length = len(sent.split(" "))
        if length > LONG_SENTENCE_LENGTH:
            length = -1  # catch-all for long sentences
        ref_num_words.update([length])

    print(f"Number of utterances: {len(ref_num_words)}")
    print(f"Most common num of words: {ref_num_words.most_common(20)}")
    convs_cr.to_csv(os.path.join(DATA_DIR, f"child_cr_min_age_{min_age}.csv"), index=False)

    convs_non_cr = conversations[~conversations.response_is_clarification_request].copy()
    convs_non_cr = convs_non_cr.sample(frac=1, random_state=1).reset_index(drop=True)   # Shuffle data
    convs_non_cr["num_words"] = convs_non_cr.utt_transcript_clean.apply(lambda x: len(x.split(" ")))
    convs_non_cr["num_words"] = convs_non_cr.num_words.apply(lambda x: -1 if x > LONG_SENTENCE_LENGTH else x)
    convs_non_cr_filtered = []
    for num_words, count in ref_num_words.items():
        convs_non_cr_target_num_words = convs_non_cr[convs_non_cr.num_words == num_words]
        convs_non_cr_filtered.append(convs_non_cr_target_num_words.sample(count))
    convs_non_cr_filtered = pd.concat(convs_non_cr_filtered)
    convs_non_cr_filtered["transcript_clean"] = convs_non_cr_filtered["utt_transcript_clean"]
    convs_non_cr_filtered = convs_non_cr_filtered[["transcript_clean"]]

    print(f"Number of utts with CR response: {len(convs_cr)}")
    print(f"Number of utts without CR response: {len(convs_non_cr_filtered)}")

    convs_non_cr_filtered.to_csv(os.path.join(DATA_DIR, f"child_non_cr_min_age_{min_age}.csv"), index=False)
