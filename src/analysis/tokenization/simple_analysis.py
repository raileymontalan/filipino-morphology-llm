"""
Simplified Tokenization Analysis.

Quick comparison of GPT-2 vs morphologically-aware tokenization
on Filipino words with affix annotations.
"""

import json
import sys

sys.path.insert(0, ".")

from typing import Dict, List

import tiktoken


def load_annotations(file_path: str, limit: int = 100) -> List[Dict]:
    """Load morpheme annotations."""
    annotations = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            annotations.append(json.loads(line))
    return annotations


def tokenize_word(word: str, tokenizer) -> List[str]:
    """Tokenize word and return tokens as strings."""
    token_ids = tokenizer.encode(word)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return tokens


def find_token_boundaries(word: str, tokens: List[str]) -> List[int]:
    """Find character positions of token boundaries."""
    boundaries = []
    pos = 0
    for token in tokens:
        pos += len(token)
        if pos < len(word):
            boundaries.append(pos)
    return boundaries


def compute_morph_score(annotations: List[Dict], tokenizer) -> Dict:
    """Compute MorphScore: alignment between token and morpheme boundaries."""
    total_aligned = 0
    total_morpheme_boundaries = 0
    word_scores = []

    for ann in annotations:
        word = ann["word"]
        morpheme_boundaries = set(ann.get("boundaries", []))

        if not morpheme_boundaries:
            continue

        # Tokenize and get boundaries
        tokens = tokenize_word(word, tokenizer)
        token_boundaries = set(find_token_boundaries(word, tokens))

        # Count aligned boundaries
        aligned = len(token_boundaries & morpheme_boundaries)

        total_aligned += aligned
        total_morpheme_boundaries += len(morpheme_boundaries)

        word_scores.append(
            {
                "word": word,
                "morphemes": ann["morphemes"],
                "tokens": tokens,
                "morph_boundaries": sorted(morpheme_boundaries),
                "token_boundaries": sorted(token_boundaries),
                "aligned": aligned,
                "total_morph": len(morpheme_boundaries),
                "score": (aligned / len(morpheme_boundaries) if morpheme_boundaries else 0),
            }
        )

    morph_score = total_aligned / total_morpheme_boundaries if total_morpheme_boundaries > 0 else 0

    return {
        "morph_score": morph_score,
        "total_aligned": total_aligned,
        "total_boundaries": total_morpheme_boundaries,
        "word_scores": word_scores,
    }


def compute_fragmentation(annotations: List[Dict], tokenizer) -> Dict:
    """Compute fragmentation: tokens per morpheme."""
    total_tokens = 0
    total_morphemes = 0
    frag_scores = []

    for ann in annotations:
        word = ann["word"]
        morphemes = ann["morphemes"]
        tokens = tokenize_word(word, tokenizer)

        num_morphemes = len(morphemes)
        num_tokens = len(tokens)

        total_tokens += num_tokens
        total_morphemes += num_morphemes

        frag_scores.append(
            {
                "word": word,
                "num_morphemes": num_morphemes,
                "num_tokens": num_tokens,
                "fragmentation": num_tokens / num_morphemes if num_morphemes > 0 else 0,
            }
        )

    avg_fragmentation = total_tokens / total_morphemes if total_morphemes > 0 else 0

    return {
        "fragmentation": avg_fragmentation,
        "total_tokens": total_tokens,
        "total_morphemes": total_morphemes,
        "scores": frag_scores,
    }


def main():
    """Run simple tokenization analysis."""
    print("=" * 70)
    print("TOKENIZATION ANALYSIS: GPT-2 BASELINE")
    print("=" * 70)
    print()

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations("data/corpora/affix_annotations.jsonl", limit=100)
    print(f"Loaded {len(annotations)} annotated words")
    print()

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Using tokenizer: GPT-2 (vocab size: {tokenizer.n_vocab})")
    print()

    # Compute metrics
    print("Computing metrics...")
    morph_results = compute_morph_score(annotations, tokenizer)
    frag_results = compute_fragmentation(annotations, tokenizer)

    print("  ✓ MorphScore computed")
    print("  ✓ Fragmentation computed")
    print()

    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(f"MorphScore:          {morph_results['morph_score']:.3f}")
    print(f"  Aligned boundaries: {morph_results['total_aligned']}/{morph_results['total_boundaries']}")
    print()

    print(f"Fragmentation:       {frag_results['fragmentation']:.3f} tokens/morpheme")
    print(f"  Total tokens:       {frag_results['total_tokens']}")
    print(f"  Total morphemes:    {frag_results['total_morphemes']}")
    print()

    # Examples
    print("Example Tokenizations:")
    print("-" * 70)
    for i, word_score in enumerate(morph_results["word_scores"][:10]):
        word = word_score["word"]
        morphemes = word_score["morphemes"]
        tokens = word_score["tokens"]
        score = word_score["score"]

        print(f"{i + 1}. {word}")
        print(f"   Morphemes: {' + '.join(morphemes)}")
        print(f"   Tokens:    {' | '.join(tokens)}")
        print(f"   MorphScore: {score:.2f}")
        print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if morph_results["morph_score"] < 0.3:
        print("⚠ Low MorphScore (<0.3): Token boundaries rarely align with morpheme boundaries")
        print("  → Standard BPE tokenization does not respect Filipino morphology")
    elif morph_results["morph_score"] < 0.6:
        print("→ Moderate MorphScore (0.3-0.6): Some alignment with morpheme boundaries")
    else:
        print("✓ Good MorphScore (>0.6): Strong alignment with morpheme boundaries")

    print()

    if frag_results["fragmentation"] > 1.5:
        print("⚠ High fragmentation (>1.5): Morphemes split across multiple tokens")
        print("  → May hinder morphological understanding")
    elif frag_results["fragmentation"] > 1.2:
        print("→ Moderate fragmentation (1.2-1.5): Some morpheme splitting")
    else:
        print("✓ Low fragmentation (<1.2): Morphemes mostly intact")

    print()

    # Save results
    print("Saving results...")
    results = {
        "tokenizer": "gpt2",
        "num_words": len(annotations),
        "morph_score": morph_results["morph_score"],
        "fragmentation": frag_results["fragmentation"],
        "examples": morph_results["word_scores"][:20],
    }

    with open("results/tokenization_baseline.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✓ Results saved to results/tokenization_baseline.json")
    print()
    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
