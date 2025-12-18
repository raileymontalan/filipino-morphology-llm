"""Utility functions and constants for evaluation."""

from .constants import (
    ACCENTED_VOWELS,
    AFFIX_TYPES,
    DEFAULT_FREQ_WEIGHT,
    DEFAULT_FREQUENCY_FILE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RANK_FILLNA,
    DIACRITIC_MAP,
    DIACRITICS,
    LETTER_PAIRS,
    MABILIS,
    MALUMI,
    MARAGSA,
    MCQ_LABEL_MAP,
    MIN_WORD_LENGTH_COMPOSITION,
    MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING,
    MIN_WORD_LENGTH_MANIPULATION,
    MIN_WORD_LENGTH_SYLLABIFICATION,
    NUM_INCORRECT_OPTIONS,
    NUM_MCQ_OPTIONS,
    REVERSE_DIACRITIC_MAP,
    STRESS_PRONUNCIATION_MAP,
    UPPERCASE_DIACRITICS,
    UPPERCASE_LETTERS,
    VOWELS,
)

# Helper functions
from .helpers import prepare_gen_outputs, prepare_mcq_outputs

# String manipulation functions
from .strings import (
    chars_to_string,
    delete_char,
    diacritize,
    duplicate_char,
    get_random_char,
    insert_char,
    normalize_diacritic,
    permute_char,
    perturb_string,
    randomly_delete_char,
    randomly_diacritize,
    randomly_insert_char,
    randomly_merge_chars,
    same_string,
    shuffle_chars,
    spell_string,
    string_to_chars,
    substitute_char,
)

# Syllabification
from .syllabification import syllabify

__all__ = [
    # Constants
    "MCQ_LABEL_MAP",
    "NUM_MCQ_OPTIONS",
    "NUM_INCORRECT_OPTIONS",
    "MIN_WORD_LENGTH_COMPOSITION",
    "MIN_WORD_LENGTH_MANIPULATION",
    "MIN_WORD_LENGTH_SYLLABIFICATION",
    "MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING",
    "AFFIX_TYPES",
    "VOWELS",
    "DIACRITICS",
    "ACCENTED_VOWELS",
    "UPPERCASE_LETTERS",
    "UPPERCASE_DIACRITICS",
    "MABILIS",
    "MALUMI",
    "MARAGSA",
    "DIACRITIC_MAP",
    "REVERSE_DIACRITIC_MAP",
    "LETTER_PAIRS",
    "STRESS_PRONUNCIATION_MAP",
    "DEFAULT_FREQUENCY_FILE",
    "DEFAULT_RANK_FILLNA",
    "DEFAULT_FREQ_WEIGHT",
    "DEFAULT_RANDOM_STATE",
    # Helper functions
    "prepare_mcq_outputs",
    "prepare_gen_outputs",
    # String functions
    "string_to_chars",
    "chars_to_string",
    "get_random_char",
    "same_string",
    "delete_char",
    "insert_char",
    "substitute_char",
    "permute_char",
    "duplicate_char",
    "normalize_diacritic",
    "diacritize",
    "randomly_diacritize",
    "spell_string",
    "shuffle_chars",
    "randomly_merge_chars",
    "randomly_insert_char",
    "randomly_delete_char",
    "perturb_string",
    # Syllabification
    "syllabify",
]
