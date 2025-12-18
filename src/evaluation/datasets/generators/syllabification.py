"""
Syllabification Dataset Generation Module.

This module provides functionality for creating linguistic datasets focused on
Filipino syllabification tasks, including stress classification, reduplication
detection, and syllable counting with 'ng' digraph awareness. Supports both
multiple-choice questions (MCQ) and generative (GEN) formats.
"""

import random
from typing import Any, Dict, List, Optional

import pandas as pd

from ...utils.constants import (
    MCQ_LABEL_MAP,
    MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING,
    MIN_WORD_LENGTH_SYLLABIFICATION,
    NUM_MCQ_OPTIONS,
    STRESS_PRONUNCIATION_MAP,
)
from ...utils.helpers import prepare_gen_outputs, prepare_mcq_outputs

# ============================================================================
# Helper Functions
# ============================================================================


def prepare_options(words: List[Any], correct_word: Any) -> Dict[str, Any]:
    """
    Prepare MCQ options from a list of words, with one correct answer.

    Args:
        words: List of all candidate words
        correct_word: The correct answer to be used

    Returns:
        Dictionary with 'correct' and three 'incorrect' options
    """
    incorrect_words = [word for word in words if word != correct_word]

    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_words[0] if len(incorrect_words) > 0 else correct_word,
        "incorrect2": incorrect_words[1] if len(incorrect_words) > 1 else correct_word,
        "incorrect3": incorrect_words[2] if len(incorrect_words) > 2 else correct_word,
    }

    return mcq_options


# ============================================================================
# Task Creation Functions - Stress Classification
# ============================================================================


def create_mcq_stress_classification(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an MCQ task for identifying syllable stress type.

    Tests knowledge of Filipino stress patterns on the last syllable:
    - mabilis (acute: á): rising tone
    - malumi (grave: à): falling tone
    - maragsa (circumflex: â): glottal stop
    - malumay (unmarked): neutral

    Args:
        row: Dictionary containing word data with 'last_syllable_pronunciation' key

    Returns:
        Dictionary with formatted MCQ prompts and options
    """
    text_en = 'What is the stress type of the last syllable in "{word}"?'
    text_tl = 'Ano ang uri ng diin ng huling pantig sa "{word}"?'

    correct_pronunciation = row["last_syllable_pronunciation"]
    correct_label = STRESS_PRONUNCIATION_MAP[correct_pronunciation]

    other_pronunciations = [p for p in STRESS_PRONUNCIATION_MAP.keys() if p != correct_pronunciation]
    random.shuffle(other_pronunciations)

    mcq_options = {
        "correct": correct_label,
        "incorrect1": STRESS_PRONUNCIATION_MAP[other_pronunciations[0]],
        "incorrect2": STRESS_PRONUNCIATION_MAP[other_pronunciations[1]],
        "incorrect3": STRESS_PRONUNCIATION_MAP[other_pronunciations[2]],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_gen_stress_classification(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a generative task for identifying syllable stress type.

    Asks students to identify the stress pattern on the last syllable of a word.

    Args:
        row: Dictionary containing word data with 'last_syllable_pronunciation' key

    Returns:
        Dictionary with formatted prompts and label
    """
    text_en = 'What is the stress type of the last syllable in "{word}"?'
    text_tl = 'Ano ang uri ng diin ng huling pantig sa "{word}"?'

    correct_pronunciation = row["last_syllable_pronunciation"]
    label = STRESS_PRONUNCIATION_MAP[correct_pronunciation]

    outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
    return outputs


# ============================================================================
# Helper Functions - Reduplication Detection
# ============================================================================


def is_reduplicated(word: str, syllables: List[str]) -> bool:
    """
    Check if a word contains CV-reduplication pattern.

    Detects if the first two syllables share the same consonant-vowel (CV)
    pattern at their start, which is a common reduplication pattern in Filipino.

    Args:
        word: The complete word string
        syllables: List of syllables in the word

    Returns:
        True if CV-reduplication is detected, False otherwise

    Examples:
        >>> is_reduplicated("maganda", ["ma", "gan", "da"])
        False
        >>> is_reduplicated("babae", ["ba", "ba", "e"])
        True
    """
    if len(syllables) < 2:
        return False

    first_syll = syllables[0].lower()
    second_syll = syllables[1].lower()

    # Check for CV reduplication (consonant-vowel pattern)
    # Extract CV from first syllable
    if len(first_syll) >= 2:
        # Get first consonant and first vowel
        cv_pattern = first_syll[:2]

        # Check if second syllable starts with same CV pattern
        if len(second_syll) >= 2:
            if second_syll[:2] == cv_pattern:
                return True

        # Also check single character CV reduplication (e.g., ma-ma)
        if len(first_syll) >= 1 and len(second_syll) >= 1:
            if first_syll[0] == second_syll[0]:
                # Check if it's a simple reduplication (ma-ma, ba-ba)
                if len(first_syll) == len(second_syll) and first_syll == second_syll:
                    return True

    return False


# ============================================================================
# Task Creation Functions - Reduplication
# ============================================================================


def create_mcq_reduplication_detection(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Create an MCQ task for detecting which word has CV-reduplication.

    Generates a multiple-choice question where students identify which word
    contains a consonant-vowel reduplication pattern (e.g., ba-ba-e, ta-ta-k).

    Args:
        rows: List of word dictionaries, each containing 'normalized_word' and
              'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted MCQ prompts and options, or None if insufficient
        reduplicated/non-reduplicated words are available
    """
    text_en = "Which word has CV-reduplication (first consonant-vowel repeated)?"
    text_tl = "Alin ang may uulit-pantig (inuulit ang unang katinig-patinig)?"

    reduplicated_words = []
    non_reduplicated_words = []

    for row in rows:
        syllables = row["normalized_syllable_list"]
        word = row["normalized_word"]
        if is_reduplicated(word, syllables):
            reduplicated_words.append(word)
        else:
            non_reduplicated_words.append(word)

    if len(reduplicated_words) > 0 and len(non_reduplicated_words) >= 3:
        correct_word = reduplicated_words[0]
        incorrect_options = non_reduplicated_words[:3]
    else:
        return None

    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_options[0],
        "incorrect2": incorrect_options[1],
        "incorrect3": incorrect_options[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options)
    return outputs


def create_gen_reduplication_identification(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Create a generative task for identifying the reduplicated syllable.

    Asks students to identify which syllable pattern is being repeated in a
    word with reduplication.

    Args:
        row: Dictionary containing 'normalized_word' and 'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted prompts and label, or None if no reduplication
        is detected
    """
    text_en = 'What is the reduplicated syllable in "{normalized_word}"?'
    text_tl = 'Ano ang inuulit na pantig sa "{normalized_word}"?'

    syllables = row["normalized_syllable_list"]

    if len(syllables) >= 2:
        first_syll = syllables[0].lower()
        second_syll = syllables[1].lower()

        # Check for full syllable reduplication first
        if first_syll == second_syll:
            label = first_syll
            outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
            return outputs

        # Then check for CV reduplication
        if len(first_syll) >= 2 and len(second_syll) >= 2:
            if first_syll[:2] == second_syll[:2]:
                label = first_syll[:2]
                outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
                return outputs

    return None


# ============================================================================
# Task Creation Functions - Syllable Counting with 'ng'
# ============================================================================


def count_syllables_with_ng(word: str, syllables: List[str]) -> int:
    """
    Count syllables in a word (helper function for 'ng' awareness tasks).

    Args:
        word: The complete word string
        syllables: List of syllables in the word

    Returns:
        Number of syllables
    """
    return len(syllables)


def create_mcq_ng_syllable_count(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an MCQ task for counting syllables in words containing 'ng'.

    Tests awareness that 'ng' is a single consonant digraph in Filipino, not
    two separate sounds. This affects syllable counting (e.g., "ngayon" is
    2 syllables, not 3).

    Args:
        row: Dictionary containing 'normalized_word' and 'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted MCQ prompts and options
    """
    text_en = 'How many syllables in "{normalized_word}"?'
    text_tl = 'Ilang pantig ang "{normalized_word}"?'

    correct_count = len(row["normalized_syllable_list"])

    options = [correct_count - 1, correct_count, correct_count + 1, correct_count + 2]
    options = [o for o in options if o > 0]
    random.shuffle(options[1:])

    mcq_options = prepare_options(options, correct_count)

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_gen_ng_syllable_count(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a generative task for counting syllables in words containing 'ng'.

    Asks students to count syllables, testing awareness that 'ng' is a single
    consonant digraph in Filipino.

    Args:
        row: Dictionary containing 'normalized_word' and 'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted prompts and label (syllable count as string)
    """
    text_en = 'How many syllables in "{normalized_word}"?'
    text_tl = 'Ilang pantig ang "{normalized_word}"?'

    correct_count = len(row["normalized_syllable_list"])
    label = str(correct_count)

    outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
    return outputs


# ============================================================================
# Task Creation Functions - General Syllable Counting
# ============================================================================


def create_mcq_general_syllable_count(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an MCQ task for counting syllables in words (general syllable counting).

    Tests basic syllable counting skills without any specific focus on digraphs
    or other linguistic features.

    Args:
        row: Dictionary containing 'normalized_word' and 'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted MCQ prompts and options
    """
    text_en = 'How many syllables in "{normalized_word}"?'
    text_tl = 'Ilang pantig ang "{normalized_word}"?'

    correct_count = len(row["normalized_syllable_list"])

    options = [correct_count - 1, correct_count, correct_count + 1, correct_count + 2]
    options = [o for o in options if o > 0]
    random.shuffle(options[1:])

    mcq_options = prepare_options(options, correct_count)

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_gen_general_syllable_count(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a generative task for counting syllables in words (general syllable counting).

    Asks students to count syllables in longer words, testing basic syllabification skills.

    Args:
        row: Dictionary containing 'normalized_word' and 'normalized_syllable_list' keys

    Returns:
        Dictionary with formatted prompts and label (syllable count as string)
    """
    text_en = 'How many syllables in "{normalized_word}"?'
    text_tl = 'Ilang pantig ang "{normalized_word}"?'

    correct_count = len(row["normalized_syllable_list"])
    label = str(correct_count)

    outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
    return outputs


# ============================================================================
# Helper Functions for Dataset Creation
# ============================================================================


def _filter_by_word_length(syllables_df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """
    Filter syllables DataFrame to only include words of sufficient length.

    Args:
        syllables_df: Input DataFrame containing syllable/word data
        min_length: Minimum word length (inclusive)

    Returns:
        Filtered DataFrame containing only words with length >= min_length
    """
    return syllables_df[syllables_df["normalized_word"].str.len() >= min_length].reset_index(drop=True)


def _apply_frequency_weighting(syllables_df: pd.DataFrame, freq_weight: float, random_seed: int) -> pd.DataFrame:
    """
    Apply frequency-based sampling to the syllables dataframe.

    Args:
        syllables_df: Input DataFrame containing syllable/word data
        freq_weight: Weight for frequency-based sampling (0.0 to 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with frequency rankings and sampling applied
    """
    from evaluation.utils.sampling import (
        add_frequency_ranks,
        load_frequency_data,
        sample_by_frequency,
    )

    freq_df = load_frequency_data()
    syllables_df = add_frequency_ranks(syllables_df, freq_df)
    syllables_df = sample_by_frequency(
        syllables_df,
        n_samples=len(syllables_df),
        freq_weight=freq_weight,
        random_state=random_seed,
    )
    return syllables_df


def _shuffle_mcq_options(dataset: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """
    Shuffle MCQ options and assign labels (A, B, C, D) to the dataset.

    Takes MCQ options from the 'mcq_options' field, randomly shuffles them,
    and assigns the correct label based on where the correct answer ends up.

    Args:
        dataset: DataFrame with MCQ prompts containing 'mcq_options' field
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with shuffled options and assigned labels
    """
    random.seed(random_seed)

    for i in range(len(dataset)):
        label_index = i % NUM_MCQ_OPTIONS
        correct = dataset.iloc[i]["prompts"][0]["mcq_options"]["correct"]
        options = [
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect1"],
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect2"],
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect3"],
        ]
        random.shuffle(options)
        options.insert(label_index, correct)
        choices = {
            "choice1": options[0],
            "choice2": options[1],
            "choice3": options[2],
            "choice4": options[3],
        }
        label = MCQ_LABEL_MAP[label_index]
        dataset.at[i, "prompts"][0].update(choices)
        dataset.at[i, "label"] = label

    return dataset


# ============================================================================
# Main Dataset Creation Function
# ============================================================================


def create_syllabification_dataset(
    syllables_df: pd.DataFrame,
    num_samples: int,
    mode: str = "mcq",
    random_seed: int = 100,
    freq_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Create a complete syllabification dataset with various Filipino linguistic tasks.

    This function generates a comprehensive dataset for testing Filipino syllabification
    and phonological awareness skills. It creates multiple types of tasks including:
    - Stress classification: Identify stress patterns on final syllables
    - Reduplication detection: Identify words with CV-reduplication
    - Syllable counting with 'ng': Count syllables correctly treating 'ng' as single sound
    - General syllable counting: Count syllables in longer words (9+ characters)

    Args:
        syllables_df: DataFrame containing syllable/word data with columns:
            - word: The original word (may contain diacritics/uppercase)
            - normalized_word: Normalized version of the word
            - normalized_syllable_list: List of syllables
            - last_syllable_pronunciation: Stress type of final syllable
        num_samples: Number of samples to generate per task type
        mode: Question format - 'mcq' for multiple-choice or 'gen' for generative (default: 'mcq')
        random_seed: Random seed for reproducibility (default: 100)
        freq_weight: Weight for frequency-based sampling, 0.0 to 1.0 (default: 0.0)
            Higher values prioritize more common words

    Returns:
        DataFrame with columns:
            - category: Task category (always "syllabification")
            - subcategory: Specific task type (stress_classification, reduplication_detection, etc.)
            - prompts: List containing question prompts in English and Filipino
            - label: Correct answer (A/B/C/D for MCQ, actual answer for GEN)

    Raises:
        ValueError: If mode is not 'mcq' or 'gen'

    Examples:
        >>> syllables = pd.read_json("syllables.jsonl", lines=True)
        >>> mcq_dataset = create_syllabification_dataset(syllables, num_samples=100, mode='mcq')
        >>> gen_dataset = create_syllabification_dataset(syllables, num_samples=100, mode='gen')
    """
    random.seed(random_seed)

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    # Filter for words of sufficient length
    syllables_df = _filter_by_word_length(syllables_df, MIN_WORD_LENGTH_SYLLABIFICATION)

    # Apply frequency weighting if requested
    if freq_weight > 0:
        syllables_df = _apply_frequency_weighting(syllables_df, freq_weight, random_seed)

    # Filter for words containing 'ng' for ng-awareness tasks
    syllables_with_ng = syllables_df[syllables_df["normalized_word"].str.contains("ng", na=False)]

    if mode == "mcq":
        # Stress classification tasks - use frequency-weighted samples
        sample_rows = (
            syllables_df.head(num_samples)
            if freq_weight > 0
            else syllables_df.sample(num_samples, random_state=random_seed)
        )
        for _, row in sample_rows.iterrows():
            mcq_row = create_mcq_stress_classification(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["stress_classification"],
                            "prompts": [mcq_row["prompts"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Reduplication detection tasks - use frequency-ordered data if weighted
        if freq_weight > 0:
            # Use already frequency-weighted and sorted data
            shuffled_dataset = syllables_df.reset_index(drop=True)
        else:
            # Use uniform random sampling
            shuffled_dataset = syllables_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        processed_count = 0
        for _, rows in shuffled_dataset.groupby(lambda x: x // NUM_MCQ_OPTIONS):
            if processed_count >= num_samples:
                break

            samples = rows.to_dict(orient="records")
            mcq_row = create_mcq_reduplication_detection(samples)

            if mcq_row is not None:
                dataset = pd.concat(
                    [
                        dataset,
                        pd.DataFrame(
                            {
                                "category": ["syllabification"],
                                "subcategory": ["reduplication_detection"],
                                "prompts": [mcq_row["prompts"]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                processed_count += 1

        # Syllable counting with 'ng' tasks
        # Prefer words with 'ng', but fall back to all words if insufficient
        if len(syllables_with_ng) >= num_samples:
            ng_sample_rows = (
                syllables_with_ng.head(num_samples)
                if freq_weight > 0
                else syllables_with_ng.sample(num_samples, random_state=random_seed)
            )
        else:
            # Use all available ng words plus additional non-ng words to reach num_samples
            ng_count = len(syllables_with_ng)
            remaining_needed = num_samples - ng_count

            if freq_weight > 0:
                ng_rows = syllables_with_ng.head(ng_count)
                # Get additional non-ng words from the frequency-weighted dataset
                non_ng_rows = syllables_df[~syllables_df["normalized_word"].str.contains("ng", na=False)].head(
                    remaining_needed
                )
            else:
                ng_rows = (
                    syllables_with_ng.sample(ng_count, random_state=random_seed) if ng_count > 0 else pd.DataFrame()
                )
                non_ng_rows = syllables_df[~syllables_df["normalized_word"].str.contains("ng", na=False)].sample(
                    remaining_needed, random_state=random_seed
                )

            ng_sample_rows = pd.concat([ng_rows, non_ng_rows], ignore_index=True)

        for _, row in ng_sample_rows.iterrows():
            mcq_row = create_mcq_ng_syllable_count(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["ng_aware_syllable_counting"],
                            "prompts": [mcq_row["prompts"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # General syllable counting tasks (longer words, minimum 9 characters)
        syllables_long = _filter_by_word_length(syllables_df, MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING)
        general_sample_rows = (
            syllables_long.head(num_samples)
            if freq_weight > 0
            else syllables_long.sample(num_samples, random_state=random_seed)
        )
        for _, row in general_sample_rows.iterrows():
            mcq_row = create_mcq_general_syllable_count(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["general_syllable_counting"],
                            "prompts": [mcq_row["prompts"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Shuffle options and assign labels
        dataset = _shuffle_mcq_options(dataset, random_seed)

    elif mode == "gen":
        # Stress classification tasks (GEN)
        sample_rows = (
            syllables_df.head(num_samples)
            if freq_weight > 0
            else syllables_df.sample(num_samples, random_state=random_seed)
        )
        for _, row in sample_rows.iterrows():
            gen_row = create_gen_stress_classification(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["stress_classification"],
                            "prompts": [gen_row["prompts"]],
                            "label": [gen_row["label"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Reduplication identification tasks (GEN) - use frequency-ordered data if weighted
        if freq_weight > 0:
            # Use already frequency-weighted and sorted data
            redup_dataset = syllables_df.reset_index(drop=True)
        else:
            # Use uniform random sampling
            redup_dataset = syllables_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        processed_count = 0
        for _, row in redup_dataset.iterrows():
            if processed_count >= num_samples:
                break

            gen_row = create_gen_reduplication_identification(row)
            if gen_row is not None:
                dataset = pd.concat(
                    [
                        dataset,
                        pd.DataFrame(
                            {
                                "category": ["syllabification"],
                                "subcategory": ["reduplication_identification"],
                                "prompts": [gen_row["prompts"]],
                                "label": [gen_row["label"]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                processed_count += 1

        # Syllable counting with 'ng' tasks (GEN)
        # Prefer words with 'ng', but fall back to all words if insufficient
        if len(syllables_with_ng) >= num_samples:
            ng_sample_rows = (
                syllables_with_ng.head(num_samples)
                if freq_weight > 0
                else syllables_with_ng.sample(num_samples, random_state=random_seed)
            )
        else:
            # Use all available ng words plus additional non-ng words to reach num_samples
            ng_count = len(syllables_with_ng)
            remaining_needed = num_samples - ng_count
            if freq_weight > 0:
                ng_rows = syllables_with_ng.head(ng_count)
                # Get additional non-ng words from the frequency-weighted dataset
                non_ng_rows = syllables_df[~syllables_df["normalized_word"].str.contains("ng", na=False)].head(
                    remaining_needed
                )
            else:
                ng_rows = (
                    syllables_with_ng.sample(ng_count, random_state=random_seed) if ng_count > 0 else pd.DataFrame()
                )
                non_ng_rows = syllables_df[~syllables_df["normalized_word"].str.contains("ng", na=False)].sample(
                    remaining_needed, random_state=random_seed
                )

            ng_sample_rows = pd.concat([ng_rows, non_ng_rows], ignore_index=True)

        for _, row in ng_sample_rows.iterrows():
            gen_row = create_gen_ng_syllable_count(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["ng_aware_syllable_counting"],
                            "prompts": [gen_row["prompts"]],
                            "label": [gen_row["label"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # General syllable counting tasks (GEN) - longer words, minimum 9 characters
        syllables_long = _filter_by_word_length(syllables_df, MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING)
        general_sample_rows = (
            syllables_long.head(num_samples)
            if freq_weight > 0
            else syllables_long.sample(num_samples, random_state=random_seed)
        )
        for _, row in general_sample_rows.iterrows():
            gen_row = create_gen_general_syllable_count(row)
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame(
                        {
                            "category": ["syllabification"],
                            "subcategory": ["general_syllable_counting"],
                            "prompts": [gen_row["prompts"]],
                            "label": [gen_row["label"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'mcq' or 'gen'.")

    return dataset
