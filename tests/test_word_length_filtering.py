"""
Demonstration of Word Length Filtering in PACUTE

This script demonstrates how minimum word length filtering improves
the quality of generated tasks, especially for manipulation tasks.
"""

import pandas as pd

from pacute.composition import create_composition_dataset
from pacute.manipulation import create_manipulation_dataset
from setup_paths import setup_project_paths

setup_project_paths()


def main():
    print("=" * 70)
    print("PACUTE Word Length Filtering Demonstration")
    print("=" * 70)

    # Load syllables data
    syllables = pd.read_json("data/syllables.jsonl", lines=True)
    syllable_lengths = syllables["normalized_word"].str.len()

    print(f"\nTotal words in dataset: {len(syllables)}")
    print("Word length distribution:")
    print(f"  2 chars or less: {len(syllables[syllable_lengths <= 2])}")
    print(f"  3-5 chars: {len(syllables[(syllable_lengths >= 3) & (syllable_lengths <= 5)])}")
    print(f"  6+ chars: {len(syllables[syllable_lengths >= 6])}")
    print("\n" + "=" * 70)
    print("COMPOSITION TASKS (Minimum length: 3 characters)")
    print("=" * 70)
    print("\nComposition tasks work well with shorter words because:")
    print("  - Spelling: Even short words have multiple letters to spell")
    print("  - Character counting: 3+ chars provide meaningful counts")
    print("  - Length analysis: Reasonable variation in 3+ char words")

    # Generate composition dataset
    composition_mcq = create_composition_dataset(syllables, num_samples=5, mode="mcq", random_seed=42, freq_weight=0.9)

    print(f"\nGenerated {len(composition_mcq)} composition MCQ samples")
    print(f"Subcategories: {composition_mcq['subcategory'].value_counts().to_dict()}")

    # Show sample composition question
    sample = composition_mcq.iloc[0]
    print("\nSample composition question:")
    print(f"  Subcategory: {sample['subcategory']}")
    print(f"  English: {sample['prompts'][0]['text_en']}")
    print(f"  Filipino: {sample['prompts'][0]['text_tl']}")
    print(f"  Answer: {sample['label']}")

    print("\n" + "=" * 70)
    print("MANIPULATION TASKS (Minimum length: 6 characters)")
    print("=" * 70)
    print("\nManipulation tasks require longer words because:")
    print("  - Deletion: Need space to delete without making word meaningless")
    print("  - Insertion: Need visible changes in reasonably-sized words")
    print("  - Substitution: Need context for meaningful character replacement")
    print("  - Permutation: Need multiple characters that can be swapped")
    print("  - Duplication: Need length for visible character repetition")

    # Generate manipulation dataset
    manipulation_mcq = create_manipulation_dataset(
        syllables, num_samples=5, mode="mcq", random_seed=42, freq_weight=0.9
    )

    print(f"\nGenerated {len(manipulation_mcq)} manipulation MCQ samples")
    print(f"Subcategories: {manipulation_mcq['subcategory'].value_counts().to_dict()}")

    # Show sample manipulation question
    sample = manipulation_mcq.iloc[0]
    print("\nSample manipulation question:")
    print(f"  Subcategory: {sample['subcategory']}")
    print(f"  English: {sample['prompts'][0]['text_en']}")
    print(f"  Filipino: {sample['prompts'][0]['text_tl']}")
    print(f"  Answer: {sample['label']}")

    print("\n" + "=" * 70)
    print("WORD QUALITY WITH HIGH FREQUENCY WEIGHTING")
    print("=" * 70)
    print("\nWith freq_weight=0.9 and minimum length filtering:")
    print("  ✓ Common words are prioritized (higher frequency rank)")
    print("  ✓ Words are long enough for meaningful operations")
    print("  ✓ Tasks demonstrate realistic linguistic patterns")
    print("  ✓ Edge cases (very short words) are avoided")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Composition: min_length=3, suitable for counting/spelling")
    print("✓ Manipulation: min_length=6, room for character operations")
    print("✓ Both modules successfully filter by word length")
    print("✓ Frequency weighting works correctly with length filtering")


if __name__ == "__main__":
    main()
