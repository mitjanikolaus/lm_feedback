import os
from pathlib import Path
import re

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import sent_tokenize

import pandas as pd
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, \
    GPT2TokenizerFast, BertTokenizerFast, PreTrainedTokenizerFast, AutoTokenizer

from tokenizers import Tokenizer, normalizers
from utils import BABYLM_DATA_DIR, SPEAKER_CODES_CAREGIVER, BABYLM_DATA_DIR_CLEAN, BABYLM_DATA_PATH_DEV_CLEAN, \
    DEV_SET, TRAINING_TRACK_STRICT_SMALL, TRAIN_SET, CHILDES_LM_DATA_FILE, CHILDES_RL_DATA_FILE

DEV_SET_SIZE = 0.1
SPLIT_RANDOM_STATE = 1

SEQUENCE_START_TOKEN = "<|startoftext|>"
SEQUENCE_END_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<mask>"
SEP_TOKEN = "<|sep|>"
CLS_TOKEN = "<|cls|>"

SPECIAL_TOKENS = [PAD_TOKEN, SEQUENCE_END_TOKEN, SEQUENCE_START_TOKEN, UNK_TOKEN]

DATA_FILE_CHILDES = "childes"
DATA_FILE_BNC = "bnc_spoken"
DATA_FILE_GUTENBERG = "gutenberg"
DATA_FILE_OPEN_SUBTITLES = "open_subtitles"
DATA_FILE_WIKI = "simple_wiki"
DATA_FILE_SWITCHBOARD = "switchboard"

VOCAB_MIN_WORD_FREQ = 2

DATA_NAMES = [DATA_FILE_CHILDES, DATA_FILE_BNC, DATA_FILE_GUTENBERG, DATA_FILE_OPEN_SUBTITLES, DATA_FILE_WIKI,
              DATA_FILE_SWITCHBOARD]


def train_tokenizer(save_path, vocab_size, data_iterator=None, data_file_names=None, training_track=None,
                    tokenizer_type="word_level"):
    print(f"Training {tokenizer_type} tokenizer for vocab size {vocab_size} .. ")
    if tokenizer_type == "bpe":
        tokenizer = ByteLevelBPETokenizer()

        if data_iterator is not None:
            tokenizer.train_from_iterator(data_iterator, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        else:
            paths = [os.path.join(BABYLM_DATA_DIR_CLEAN, training_track, f"{name}.train") for name in data_file_names]
            tokenizer.train(files=paths, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(save_path)

    elif tokenizer_type == "word_level":
        if data_iterator is not None:
            tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase()])

            trainer = WordLevelTrainer(min_frequency=VOCAB_MIN_WORD_FREQ, special_tokens=SPECIAL_TOKENS, vocab_size=vocab_size)
            tokenizer.train_from_iterator(data_iterator, trainer=trainer)
            tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenizer_fast.save_pretrained(save_path)
        else:
            raise NotImplementedError()
    elif tokenizer_type == "word_piece":
        tokenizer = BertWordPieceTokenizer()
        if data_iterator is not None:
            tokenizer.train_from_iterator(data_iterator, vocab_size=vocab_size,
                                          special_tokens=SPECIAL_TOKENS + [SEP_TOKEN, CLS_TOKEN])
        else:
            paths = [os.path.join(BABYLM_DATA_DIR_CLEAN, training_track, f"{name}.train") for name in data_file_names]
            tokenizer.train(files=paths, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(save_path)
    else:
        raise RuntimeError(f"Unknown tokenizer type: {tokenizer_type}")

    print(f"Saved trained tokenizer to {save_path}")


CONTRACTIONS = {
    "i'm": "i am",
    "I'm": "I am",
    "it's": "it is",
    "she's": "she is",
    "he's": "he is",
    "one's": "one is",
    "who's": "who is",
    "what's": "what is",
    "how's": "how is",
    "when's": "when is",
    "there's": "there is",
    "that's": "that is",
    "where's": "where is",
    "here's": "here is",
    "why's": "why is",
    "we're": "they are",
    "you're": "they are",
    "yo're": "they are",
    "they're": "they are",
    "They're": "they are",
    "one're": "one are",
    "what're": "what are",
    "wha're": "what are",
    "where're": "where are",
    "here're": "here are",
    "there're": "there are",
    "why're": "why are",
    "how're": "how are",
    "who're": "who are",
    "when're": "when are",
    "those're": "those are",
    "these're": "these are",
    "that're": "that are",
    "I've": "I have",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "should've": "should have",
    "would've": "would have",
    "could've": "could have",
    "might've": "might have",
    "shouldn't've": "shouldn't have",
    "must've": "must have",
    "who've": "who have",
    "what've": "what have",
    "why've": "why have",
    "where've": "where have",
    "how've": "how have",
    "there've": "there have",
    "that've": "that have",
    "had've": "had have",
    "there'd": "there would",
    "what'd": "what would",
    "I'd": "I would",
    "i'd": "i would",
    "you'd": "you would",
    "she'd": "she would",
    "She'd": "She would",
    "he'd": "he would",
    "it'd": "it would",
    "one'd": "one would",
    "we'd": "we would",
    "they'd": "they would",
    "who'd": "who would",
    "why'd": "why would",
    "where'd": "where would",
    "how'd": "how would",
    "that'd": "that would",
    "this'd": "this would",
    "when'd": "when would",
    "hasta": "has to",
    "hafta": "have to",
    "hadta": "had to",
    "needta": "need to",
    "dat's": "that is",
    "dat": "that",
    "dis": "this",
    "dere": "there",
    "de": "the",
    "gonna": "going to",
    "wanna": "want to",
    "y'wanna": "you want to",
    "ywanna": "you want to",
    "dywanna": "do you want to",
    "dyawanna": "do you want to",
    "d'y'wanna": "do you want to",
    "anoder": "another",
    "dunno": "don't know",
    "'cause": "because",
    "wha'do": "what do",
    "wha'd'you": "what do you",
    "what'djou": "what do you",
    "what'dja": "what do you",
    "y'did": "you did",
    "whad'dya": "what did you",
    "d'ya": "do you",
    "I'll": "I will",
    "i'll": "i will",
    "you'll": "you will",
    "she'll": "she will",
    "he'll": "he will",
    "it'll": "it will",
    "one'll": "one will",
    "we'll": "we will",
    "they'll": "they will",
    "who'll": "who will",
    "when'll": "when will",
    "that'll": "that will",
    "there'll": "there will",
    "those'll": "those will",
    "this'll": "this will",
    "Mommy'll": "Mommy will",
    "mommy'll": "mommy will",
    "Mummy'll": "Mummy will",
    "Mama'll": "Mama will",
    "mummy'll": "mummy will",
    "mom'll": "mom will",
    "Mom'll": "Mom will",
    "mum'll": "mum will",
    "mummie'll": "mummie will",
    "Daddy'll": "Daddy will",
    "Dad'll": "Dad will",
    "Dada'll": "Dada will",
    "daddy'll": "daddy will",
    "what'll": "what will",
    "wait'll": "wait until",
    "y'haven't": "you haven't",
    "d'you": "do you",
    "what'r'ya": "what are you",
    "lookin'": "looking",
    "why'nt": "why don't",
    "c'mon": "come on",
    "c'mere": "come here",
    "com'ere": "come here",
    "come'ere": "come here",
    "wha'": "what",
    "wha'doyou": "what do you",
    "goin'": "going",
    "workin'": "working",
    "'em": "them",
    "cryin'": "crying",
    "peekin'": "peeking",
    "y'need": "you need",
    "y'know": "you know",
    "s'more": "some more",
    "s'pose": "suppose",
    "'kay": "okay",
    "di'jou": "did you",
    "combin'": "combining",
    "shakin'": "shaking",
    "makin'": "making",
    "doin'": "doing",
    "n'": "and",
    "g'head": "go ahead",
    "'member": "remember",
    "'bout": "about",
    "'round": "around",
    "jumpin'": "jumping",
    "put'em": "put them",
    "push'em": "push them",
    "whatd'ya": "what do you",
    "'ere": "here",
    "y'want": "you want",
    "y'can": "you can",
    "what'm": "what am",
    "who'm": "who am",
    "y'all": "you all",
    "s'that": "is that",
    "sh'we": "shall we",
    "y'have": "you have",
    "you'r": "you are",
    "darlin'": "darling",
    "what'ya": "what do you",
    "useta": "used to",
}


def replace_contractions(words):
    words = [
        word if word.replace(",", "") not in CONTRACTIONS.keys() else CONTRACTIONS[word.replace(",", "")]
        for word in words
    ]
    words = [word.replace("'ll", " will").replace("'ve", " have") for word in words]
    return words


def preprocess_childes_utterance(utt):
    utt = utt.strip()
    utt = utt.replace("   ", " ")
    utt = utt.replace("  ", " ")

    utt = utt.replace("Mommy's here", "Mommy is here")

    utt_without_punct = utt[:-1]
    words = utt_without_punct.split(" ")
    words = replace_contractions(words)
    cleaned_utterance = " ".join(words) + utt[-1]

    return cleaned_utterance


def load_data(data_path):
    data_df = pd.read_csv(data_path)

    data_df["transcript_clean"] = data_df["transcript_clean"].apply(preprocess_childes_utterance)

    data = data_df.transcript_clean.to_list()
    return data


class ChildesDataModule(LightningDataModule):
    def __init__(self, lm_data_path=CHILDES_LM_DATA_FILE, additional_train=None, fb=False, fb_data_path=CHILDES_RL_DATA_FILE, vocab_size=5000,
                 max_len=128, batch_size=256, num_workers=4, causal=True, max_num_words=-1,
                 tokenizer_type="word_level"):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.fb = fb
        self.tokenizer_type = tokenizer_type
        self.additional_train = additional_train

        self.save_hyperparameters()

        if ".csv" not in lm_data_path:
            raise RuntimeError(f"Unexpected file format (not .csv): {lm_data_path}")
        train_data_path = lm_data_path.replace(".csv", "_train.txt")
        val_data_path = lm_data_path.replace(".csv", "_val.txt")
        if max_num_words != -1:
            train_data_path = train_data_path.replace("_train.txt", f"_train_{max_num_words}_words.txt")
            val_data_path = val_data_path.replace("_val.txt", f"_val_{max_num_words}_words.txt")
        if not os.path.isfile(train_data_path) or not os.path.isfile(val_data_path):
            print("Creating train/val datasets")
            data = load_data(lm_data_path)
            if max_num_words != -1:
                num_words = 0
                target_index = 0
                for target_index, sent in enumerate(data):
                    num_words += len(sent.split(" "))
                    if num_words >= max_num_words:
                        break

                data = data[:target_index - 1]

            data_train, data_val = train_test_split(data, test_size=DEV_SET_SIZE, shuffle=True,
                                                    random_state=SPLIT_RANDOM_STATE)

            with open(train_data_path, "w") as file:
                file.write("\n".join(data_train))

            with open(val_data_path, "w") as file:
                file.write("\n".join(data_val))
        else:
            print(f"Loading train/val datasets from:\n{train_data_path} and\n{val_data_path}")
            with open(train_data_path, "r") as file:
                data_train = file.read().split("\n")
            with open(val_data_path, "r") as file:
                data_val = file.read().split("\n")

        tokenizer_dir = os.path.join("tokenizers", f"{tokenizer_type}_vocab_{vocab_size}")
        if max_num_words != -1:
            tokenizer_dir += f"_{max_num_words}"
        if not (os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")) or os.path.isfile(
            os.path.join(tokenizer_dir, 'vocab.txt')) or os.path.isfile(
            os.path.join(tokenizer_dir, 'tokenizer.json'))):
            os.makedirs(tokenizer_dir, exist_ok=True)
            train_tokenizer(tokenizer_dir, vocab_size, data_train, tokenizer_type=tokenizer_type)

        if tokenizer_type == "bpe":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                tokenizer_dir, return_token_type_ids=False, add_prefix_space=True, pad_token=PAD_TOKEN,
                bos_token=SEQUENCE_START_TOKEN, eos_token=SEQUENCE_END_TOKEN
            )
            self.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=self.tokenizer.bos_token + " $0 " + self.tokenizer.eos_token,
                special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                                (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],
            )
        elif tokenizer_type == "word_level":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                unk_token=UNK_TOKEN,
                                                eos_token=SEQUENCE_END_TOKEN, pad_token=PAD_TOKEN,
                                                bos_token=SEQUENCE_START_TOKEN,
                                                )
            self.tokenizer.add_bos_token = True
            self.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=self.tokenizer.bos_token + " $0 " + self.tokenizer.eos_token,
                special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                                (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],
            )
        elif tokenizer_type == "word_piece":
            self.tokenizer = BertTokenizerFast(os.path.join(tokenizer_dir, 'vocab.txt'), pad_token=PAD_TOKEN,
                                               sep_token=SEP_TOKEN, unk_token=UNK_TOKEN, cls_token=CLS_TOKEN,
                                               bos_token=SEQUENCE_START_TOKEN, eos_token=SEQUENCE_END_TOKEN)
            self.tokenizer.add_bos_token = True
            self.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=self.tokenizer.bos_token + " $0 " + self.tokenizer.eos_token,
                special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                                (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],
            )
        else:
            raise RuntimeError(f"Unknown tokenizer type: {tokenizer_type}")

        if self.additional_train is not None:
            data_add = load_data(self.additional_train)
            data_train = data_train + data_add

        self.dataset_dev = ChildesLMDataset(data_val, tokenizer=self.tokenizer, tokenizer_type=tokenizer_type,
                                            max_len=max_len)
        self.dataset_train = ChildesLMDataset(data_train, tokenizer=self.tokenizer, tokenizer_type=tokenizer_type,
                                              max_len=max_len)

        if self.fb:
            print("Loading FB data.. ", end="")
            data_fb = pd.read_csv(fb_data_path)
            data_fb["reward"] = data_fb.apply(compute_reward_value, axis=1)
            data_fb = data_fb[["utt_transcript_clean", "reward"]]

            data_fb_train, data_fb_dev = train_test_split(data_fb, test_size=DEV_SET_SIZE, shuffle=True,
                                                          random_state=SPLIT_RANDOM_STATE)
            print("Done.")
            self.dataset_fb_train = FeedbackDataset(data_fb_train, self.tokenizer, max_len)
            self.dataset_fb_dev = FeedbackDataset(data_fb_dev, self.tokenizer, max_len)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=not causal, mlm_probability=0.15 if not causal else None
        )
        self.collate_fn_fb = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                   shuffle=True, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True, collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader

    def val_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_dev, batch_size=self.batch_size, num_workers=self.num_workers,
                                       collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader


class ChildesLMDataset(Dataset):
    def __init__(self, data, tokenizer, tokenizer_type, max_len):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.max_len = max_len
        self.examples = data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        out = self.tokenizer(self.examples[i], add_special_tokens=True, truncation=True, max_length=self.max_len - 2,
                             return_token_type_ids=False)
        return out


class BabyLMDataModule(LightningDataModule):
    def __init__(self, training_track=TRAINING_TRACK_STRICT_SMALL, fb=False, fb_data_path=CHILDES_RL_DATA_FILE,
                 vocab_size=10000,
                 max_len=128, batch_size=128, num_workers=4, subset=None, causal=True):
        super().__init__()
        if subset is None:
            data_file_names = DATA_NAMES
        elif isinstance(subset, str):
            data_file_names = [subset]

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.fb = fb

        subset_name = ""
        if data_file_names != DATA_NAMES:
            subset_name = "_subset_" + "_".join(data_file_names)

        for split in [training_track, DEV_SET]:
            data_path = os.path.join(BABYLM_DATA_DIR, split)
            data_path_clean = os.path.join(BABYLM_DATA_DIR_CLEAN, split)

            for data_file_name in data_file_names:
                file_suffix = split if split == DEV_SET else TRAIN_SET
                if not os.path.isfile(os.path.join(data_path_clean, f"{data_file_name}.{file_suffix}")):
                    raw_src_file_path = os.path.join(data_path, f"{data_file_name}.{file_suffix}")
                    print("preprocessing: ", raw_src_file_path)
                    lines = Path(raw_src_file_path).read_text(encoding="utf-8").splitlines()
                    preprocessed = '\n'.join(preprocess(data_file_name, lines))
                    clean_src_file_path = os.path.join(data_path_clean, f"{data_file_name}.{file_suffix}")
                    os.makedirs(data_path_clean, exist_ok=True)
                    Path(clean_src_file_path).write_text(preprocessed, encoding="utf-8")

        tokenizer_dir = os.path.join("tokenizers", f"llama_{training_track}_vocab_{vocab_size}{subset_name}")
        os.makedirs(tokenizer_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
            # Train tokenizer if it doesn't exist yet
            train_tokenizer(tokenizer_dir, vocab_size, data_file_names=data_file_names, training_track=training_track)

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            tokenizer_dir, return_token_type_ids=False, add_prefix_space=True, pad_token=PAD_TOKEN,
            bos_token=SEQUENCE_START_TOKEN, eos_token=SEQUENCE_END_TOKEN,
        )
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=self.tokenizer.bos_token + " $0 " + self.tokenizer.eos_token,
            special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id),
                            (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],
        )

        self.dataset_dev = BabyLMDataset(BABYLM_DATA_PATH_DEV_CLEAN, tokenizer=self.tokenizer, max_len=max_len,
                                         data_file_names=data_file_names,
                                         split="dev")

        data_path_train = os.path.join(BABYLM_DATA_DIR_CLEAN, training_track)
        self.dataset_train = BabyLMDataset(data_path_train, tokenizer=self.tokenizer, max_len=max_len,
                                           data_file_names=data_file_names,
                                           split="train")

        if self.fb:
            self.dataset_fb_train = FeedbackDataset(fb_data_path, self.tokenizer, max_len)

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=not causal, mlm_probability=0.15 if not causal else None
        )
        self.collate_fn_fb = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        lm_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                   shuffle=True, collate_fn=self.collate_fn)
        if self.fb:
            fb_dataloader = DataLoader(self.dataset_fb_train, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True, collate_fn=self.collate_fn_fb)
            return {"lm": lm_dataloader, "fb": fb_dataloader}
        else:
            return lm_dataloader

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.dataset_dev, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=self.collate_fn)
        return validation_dataloader


SPEAKER_CODES_CAREGIVER_PREFIXES = ['*' + speaker_code + ':\t' for speaker_code in SPEAKER_CODES_CAREGIVER]


def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text


def preprocess(src_file_name, lines):
    lines = [l for l in lines if l]  # Discard empty lines

    if src_file_name == DATA_FILE_CHILDES:
        for line in lines:
            if line[:6] in SPEAKER_CODES_CAREGIVER_PREFIXES:
                yield cleanup_extra_spaces(line[6:])
    elif src_file_name == DATA_FILE_WIKI:
        for line in lines:
            if not line.startswith("= = ="):
                for sentence in sent_tokenize(line):
                    yield cleanup_extra_spaces(sentence)

    elif src_file_name == DATA_FILE_GUTENBERG:
        for line in lines:
            if not (line.startswith("*CHAPTER") or line.startswith("= = =")):
                line = line.replace("*", "")
                for sentence in sent_tokenize(line):
                    yield cleanup_extra_spaces(sentence)

    elif src_file_name == DATA_FILE_SWITCHBOARD:
        for line in lines:
            if line.startswith("A:\t") or line.startswith("B:\t"):
                yield cleanup_extra_spaces(line[3:])
            else:
                yield cleanup_extra_spaces(line)

    elif src_file_name in [DATA_FILE_BNC, DATA_FILE_OPEN_SUBTITLES]:
        for line in lines:
            yield cleanup_extra_spaces(line)
    else:
        raise RuntimeError("Unexpected src_file_name: " + src_file_name)


class BabyLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, data_file_names, split):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        print("Loading LM data: ")
        for src_file in data_file_names:
            src_file_path = os.path.join(data_path, f"{src_file}.{split}")
            print(src_file_path)
            lines = Path(src_file_path).read_text(encoding="utf-8").splitlines()
            self.examples += lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        out = self.tokenizer(self.examples[i], add_special_tokens=True, return_special_tokens_mask=True,
                             return_token_type_ids=True, truncation=True, max_length=self.max_len - 2)
        return out


def compute_reward_value(utt, reward_cr=0, reward_ack=1, reward_other=0.5):
    if utt.response_is_clarification_request:
        return reward_cr
    elif utt.response_is_acknowledgement:
        return reward_ack
    else:
        return reward_other


class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data.iloc[i]
        encoded_sample = self.tokenizer(item["utt_transcript_clean"], add_special_tokens=True,
                                        truncation=True, max_length=self.max_len - 2)
        encoded_sample.data["reward"] = item["reward"]
        encoded_sample.data["length"] = len(encoded_sample.input_ids)

        return encoded_sample
