import os.path

DATA_DIR = os.path.expanduser("~/data/lm_feedback/")
CHILDES_LM_DATA_FILE = os.path.join(DATA_DIR, "caregiver_utterances.csv")
CHILDES_RL_DATA_FILE = os.path.join(DATA_DIR, "conversations.csv")

BABYLM_DATA_DIR = os.path.expanduser("~/data/babylm_data/")
BABYLM_DATA_DIR_CLEAN = os.path.expanduser("~/data/babylm_data_clean/")

TRAIN_SET = "train"
DEV_SET = "dev"

BABYLM_DATA_PATH_DEV = os.path.join(BABYLM_DATA_DIR, DEV_SET)
BABYLM_DATA_PATH_DEV_CLEAN = os.path.join(BABYLM_DATA_DIR_CLEAN, DEV_SET)

TRAINING_TRACK_STRICT_SMALL = "train_10M"
TRAINING_TRACK_STRICT = "train_100M"


SPEAKER_CODE_CHILD = "CHI"

SPEAKER_CODES_CAREGIVER = [
    "MOT",
    "FAT",
    "DAD",
    "MOM",
    "GRA",
    "GRF",
    "GRM",
    "GMO",
    "GFA",
    "CAR",
]