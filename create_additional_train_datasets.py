import os
import pandas as pd
from utils import CHILDES_RL_DATA_FILE, DATA_DIR


if __name__ == '__main__':
    conversations = pd.read_csv(CHILDES_RL_DATA_FILE)

    convs_cr = conversations[conversations.response_is_clarification_request].copy()

    convs_cr["transcript_clean"] = convs_cr["utt_transcript_clean"]
    convs_cr = convs_cr[["transcript_clean"]]
    ref_num_words = 0
    for sent in convs_cr.transcript_clean.values:
        ref_num_words += len(sent.split(" "))

    print(f"Number of words: {ref_num_words}")
    convs_cr.to_csv(os.path.join(DATA_DIR, "child_cr.csv"), index=False)

    convs_non_cr = conversations[~conversations.response_is_clarification_request].copy()
    convs_non_cr = convs_non_cr.sample(frac=1, random_state=1).reset_index(drop=True)   # Shuffle data
    convs_non_cr["transcript_clean"] = convs_non_cr["utt_transcript_clean"]
    convs_non_cr = convs_non_cr[["transcript_clean"]]
    num_words = 0
    target_index = 0
    for target_index, row in convs_non_cr.iterrows():
        num_words += len(row.transcript_clean.split(" "))
        if num_words >= ref_num_words:
            break

    print(f"Number of utts with CR response: {len(convs_cr)}")
    print(f"Number of utts without CR response: {len(convs_non_cr)}")

    convs_non_cr = convs_non_cr.iloc[:target_index - 1]

    convs_non_cr.to_csv(os.path.join(DATA_DIR, "child_non_cr.csv"), index=False)


