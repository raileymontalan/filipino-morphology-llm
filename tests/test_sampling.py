import pandas as pd

from pacute.sampling import (
    add_frequency_ranks,
    load_frequency_data,
    sample_by_frequency,
)
from setup_paths import setup_project_paths

setup_project_paths()

# Load sample data
syllables_df = pd.read_json("data/syllables.jsonl", lines=True).head(1000)
print(f"Total syllables loaded: {len(syllables_df)}\n")

# Load frequency data
freq_df = load_frequency_data()
print(f"Frequency data loaded: {len(freq_df)} words\n")

# Add frequency ranks
syllables_with_ranks = add_frequency_ranks(syllables_df, freq_df)
print("Sample of data with ranks:")
print(syllables_with_ranks[["normalized_word", "rank"]].head(10))
print()

# Demonstrate different freq_weight values
print("=" * 80)
print("SAMPLING DEMONSTRATION: How freq_weight affects word selection")
print("=" * 80)
print()

for freq_weight in [0.0, 0.3, 0.7, 1.0]:
    print(f"\n{'=' * 80}")
    print(f"freq_weight = {freq_weight}")
    print(f"{'=' * 80}")

    if freq_weight == 0.0:
        print("📌 Pure random sampling (ignores frequency)")
    elif freq_weight < 0.5:
        print("📌 Mostly random, slightly favoring common words")
    elif freq_weight < 0.8:
        print("📌 Balanced: favoring common words but allowing some rare ones")
    elif freq_weight < 1.0:
        print("📌 Strongly favoring common words")
    else:
        print("📌 Pure frequency-based sampling (heavily favors common words)")

    sampled = sample_by_frequency(syllables_with_ranks, n_samples=20, freq_weight=freq_weight, random_state=42)

    # Show the sampled words with their ranks
    result = sampled[["normalized_word", "rank"]].sort_values("rank")
    print("\nTop 10 sampled words (sorted by frequency rank):")
    print(result.head(10).to_string(index=False))

    # Statistics
    avg_rank = result["rank"].mean()
    median_rank = result["rank"].median()
    min_rank = result["rank"].min()

    print("\nStatistics:")
    print(f"  Average rank: {avg_rank:.0f}")
    print(f"  Median rank:  {median_rank:.0f}")
    print(f"  Best rank:    {min_rank:.0f}")
    print("  (Lower rank = more common word)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    """
How freq_weight works:
- freq_weight = 0.0: Pure random sampling (all words equally likely)
- freq_weight = 0.5: Balanced (50% frequency-based, 50% random)
- freq_weight = 1.0: Pure frequency sampling (common words highly preferred)

Higher freq_weight → More common/frequent words in your dataset
Lower freq_weight → More uniform distribution across all words
"""
)
