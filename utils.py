import os.path

import numpy as np

BASE_DATA_DIR = os.getenv("DATA_DIR", default=os.path.expanduser("~/data"))
DATA_DIR = os.path.join(BASE_DATA_DIR, "lm_feedback")

CHILDES_LM_DATA_FILE = os.path.join(DATA_DIR, "caregiver_utterances.csv")
CHILDES_LM_TRAIN_DATA_FILE = os.path.join(DATA_DIR, "caregiver_utterances_train.txt")
CHILDES_LM_VAL_DATA_FILE = os.path.join(DATA_DIR, "caregiver_utterances_val.txt")

CHILDES_RL_DATA_FILE = os.path.join(DATA_DIR, "conversations.csv")

BABYLM_DATA_DIR = os.path.join(BASE_DATA_DIR, "babylm_data")
BABYLM_DATA_DIR_CLEAN = os.path.join(BASE_DATA_DIR, "babylm_data_clean")

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

BLIMP_METRIC_TO_PHENOMENON = {
    "anaphor_gender_agreement": "anaphor_agreement",
    "anaphor_number_agreement": "anaphor_agreement",
    "animate_subject_passive": "argument_structure",
    "animate_subject_trans": "argument_structure",
    "causative": "argument_structure",
    "drop_argument": "argument_structure",
    "inchoative": "argument_structure",
    "intransitive": "argument_structure",
    "passive_1": "argument_structure",
    "passive_2": "argument_structure",
    "transitive": "argument_structure",
    "principle_A_c_command": "binding",
    "principle_A_case_1": "binding",
    "principle_A_case_2": "binding",
    "principle_A_domain_1": "binding",
    "principle_A_domain_2": "binding",
    "principle_A_domain_3": "binding",
    "principle_A_reconstruction": "binding",
    "existential_there_object_raising": "control_raising",
    "existential_there_subject_raising": "control_raising",
    "expletive_it_object_raising": "control_raising",
    "tough_vs_raising_1": "control_raising",
    "tough_vs_raising_2": "control_raising",
    "determiner_noun_agreement_1": "determiner_noun_agr",
    "determiner_noun_agreement_2": "determiner_noun_agr",
    "determiner_noun_agreement_irregular_1": "determiner_noun_agr",
    "determiner_noun_agreement_irregular_2": "determiner_noun_agr",
    "determiner_noun_agreement_with_adjective_1": "determiner_noun_agr",
    "determiner_noun_agreement_with_adj_2": "determiner_noun_agr",
    "determiner_noun_agreement_with_adj_irregular_1": "determiner_noun_agr",
    "determiner_noun_agreement_with_adj_irregular_2": "determiner_noun_agr",
    "ellipsis_n_bar_1": "ellipsis",
    "ellipsis_n_bar_2": "ellipsis",
    "wh_questions_object_gap": "filler_gap",
    "wh_questions_subject_gap": "filler_gap",
    "wh_questions_subject_gap_long_distance": "filler_gap",
    "wh_vs_that_no_gap": "filler_gap",
    "wh_vs_that_no_gap_long_distance": "filler_gap",
    "wh_vs_that_with_gap": "filler_gap",
    "wh_vs_that_with_gap_long_distance": "filler_gap",
    "irregular_past_participle_adjectives": "irregular_forms",
    "irregular_past_participle_verbs": "irregular_forms",
    "adjunct_island": "island_effects",
    "complex_NP_island": "island_effects",
    "coordinate_structure_constraint_complex_left_branch": "island_effects",
    "coordinate_structure_constraint_object_extraction": "island_effects",
    "left_branch_island_echo_question": "island_effects",
    "left_branch_island_simple_question": "island_effects",
    "sentential_subject_island": "island_effects",
    "wh_island": "island_effects",
    "matrix_question_npi_licensor_present": "npi_licensing",
    "npi_present_1": "npi_licensing",
    "npi_present_2": "npi_licensing",
    "only_npi_licensor_present": "npi_licensing",
    "only_npi_scope": "npi_licensing",
    "sentential_negation_npi_licensor_present": "npi_licensing",
    "sentential_negation_npi_scope": "npi_licensing",
    "existential_there_quantifiers_1": "quantifiers",
    "existential_there_quantifiers_2": "quantifiers",
    "superlative_quantifiers_1": "quantifiers",
    "superlative_quantifiers_2": "quantifiers",
    "distractor_agreement_relational_noun": "subject_verb_agr",
    "distractor_agreement_relative_clause": "subject_verb_agr",
    "irregular_plural_subject_verb_agreement_1": "subject_verb_agr",
    "irregular_plural_subject_verb_agreement_2": "subject_verb_agr",
    "regular_plural_subject_verb_agreement_1": "subject_verb_agr",
    "regular_plural_subject_verb_agreement_2": "subject_verb_agr"
}


def parse_babylm_metrics_results(out):
    results = dict()
    if "zorro" in out["results"]:
        results["zorro"] = out["results"].pop("zorro")["acc,none"]
    if "zorro_filtered_childes" in out["results"]:
        results["zorro_filtered_childes"] = out["results"].pop("zorro_filtered_childes")["acc,none"]
    if "blimp_filtered" in out["results"]:
        results["blimp"] = out["results"].pop("blimp_filtered")["acc,none"]
    if "blimp_filtered_childes" in out["results"]:
        results["blimp_filtered_childes"] = out["results"].pop("blimp_filtered_childes")["acc,none"]

    phenomenon_results = dict()
    for key, val in out["results"].items():
        val = val["acc,none"]
        metric_category = key.split("_")[0]
        key = key[key.index("_") + 1:]
        if key.endswith("_filtered_childes"):
            metric_category += "_filtered_childes"
            key = key.replace("_filtered_childes", "")

        if metric_category.startswith("zorro"):
            phenomenon = key.split("-")[0]
            metric = key[key.index("-") + 1:]
        elif metric_category.startswith("blimp"):
            metric = key.replace("_filtered", "")
            phenomenon = BLIMP_METRIC_TO_PHENOMENON[metric]
        else:
            raise RuntimeError("Unknown metric key: ", key)
        prefix = metric_category + '/' + phenomenon
        results[prefix + '-' + metric] = val
        prefix_phen = metric_category + "_phenomena" + '/' + phenomenon
        if prefix_phen in phenomenon_results:
            phenomenon_results[prefix_phen].append(val)
        else:
            phenomenon_results[prefix_phen] = [val]
    phenomenon_results = {key: np.mean(values) for key, values in phenomenon_results.items()}
    results.update(phenomenon_results)
    return results
