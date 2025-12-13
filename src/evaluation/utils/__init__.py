"""
Utility functions and constants for evaluation.
"""

# Constants
from .constants import (
    MCQ_LABEL_MAP, NUM_MCQ_OPTIONS, NUM_INCORRECT_OPTIONS,
    MIN_WORD_LENGTH_COMPOSITION, MIN_WORD_LENGTH_MANIPULATION,
    MIN_WORD_LENGTH_SYLLABIFICATION, MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING,
    AFFIX_TYPES,
    VOWELS, DIACRITICS, ACCENTED_VOWELS, UPPERCASE_LETTERS, UPPERCASE_DIACRITICS,
    MABILIS, MALUMI, MARAGSA,
    DIACRITIC_MAP, REVERSE_DIACRITIC_MAP,
    LETTER_PAIRS, STRESS_PRONUNCIATION_MAP,
    DEFAULT_FREQUENCY_FILE, DEFAULT_RANK_FILLNA, DEFAULT_FREQ_WEIGHT, DEFAULT_RANDOM_STATE,
)

# Helper functions
from .helpers import prepare_mcq_outputs, prepare_gen_outputs

# String manipulation functions
from .strings import (
    string_to_chars, chars_to_string,
    get_random_char, same_string,
    delete_char, insert_char, substitute_char, permute_char, duplicate_char,
    normalize_diacritic, diacritize, randomly_diacritize,
    spell_string,
    shuffle_chars, randomly_merge_chars, randomly_insert_char, randomly_delete_char,
    perturb_string,
)

# Syllabification
from .syllabification import syllabify

__all__ = [
    # Constants
    "MCQ_LABEL_MAP", "NUM_MCQ_OPTIONS", "NUM_INCORRECT_OPTIONS",
    "MIN_WORD_LENGTH_COMPOSITION", "MIN_WORD_LENGTH_MANIPULATION",
    "MIN_WORD_LENGTH_SYLLABIFICATION", "MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING",
    "AFFIX_TYPES",
    "VOWELS", "DIACRITICS", "ACCENTED_VOWELS", "UPPERCASE_LETTERS", "UPPERCASE_DIACRITICS",
    "MABILIS", "MALUMI", "MARAGSA",
    "DIACRITIC_MAP", "REVERSE_DIACRITIC_MAP",
    "LETTER_PAIRS", "STRESS_PRONUNCIATION_MAP",
    "DEFAULT_FREQUENCY_FILE", "DEFAULT_RANK_FILLNA", "DEFAULT_FREQ_WEIGHT", "DEFAULT_RANDOM_STATE",
    # Helper functions
    "prepare_mcq_outputs", "prepare_gen_outputs",
    # String functions
    "string_to_chars", "chars_to_string",
    "get_random_char", "same_string",
    "delete_char", "insert_char", "substitute_char", "permute_char", "duplicate_char",
    "normalize_diacritic", "diacritize", "randomly_diacritize",
    "spell_string",
    "shuffle_chars", "randomly_merge_chars", "randomly_insert_char", "randomly_delete_char",
    "perturb_string",
    # Syllabification
    "syllabify",
]
