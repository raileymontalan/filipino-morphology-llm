"""
Tokenization-Only Analysis Script.

Compares tokenization approaches on Filipino morphology:
1. GPT-2 baseline (standard BPE)
2. Patok (affix-aware expand-contract tokenization)

Metrics:
- MorphScore: Alignment between token and morpheme boundaries
- Boundary F1: Precision and recall of morpheme boundary detection
- Fragmentation: Average tokens per morpheme
- Information-theoretic metrics: Mutual information I(M;T)

No model training required - pure tokenization comparison.
"""

import json
import sys

sys.path.insert(0, ".")

from typing import Dict, List, Tuple

import tiktoken

from src.analysis.information_theory import (
    InformationTheoreticAnalysis,
    MorphemeTokenAlignment,
    generate_information_theoretic_report,
)
from src.analysis.morphological_metrics import (
    MorphologicalAnnotation,
    MorphologicalMetrics,
    generate_morphological_report,
)


def load_annotations(file_path: str) -> List[Dict]:
    """Load morpheme annotations."""
    annotations = []
    with open(file_path) as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations


def tokenize_with_gpt2(word: str) -> Tuple[List[str], List[int]]:
    """Tokenize word with GPT-2 BPE tokenizer."""
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(word)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return tokens, token_ids


def tokenize_with_patok(
    word: str, morpheme_boundaries: List[int], base_tokenizer: str = "gpt2"
) -> Tuple[List[str], List[int]]:
    """
    Simulate Patok tokenization.

    Patok uses expand-contract cycles to respect morpheme boundaries.
    For this analysis, we simulate it by:
    1. Splitting word at morpheme boundaries
    2. Tokenizing each morpheme independently
    3. Concatenating results

    In practice, Patok would use stochastic expansion, but for analysis
    we use this deterministic approximation.
    """
    tokenizer = tiktoken.get_encoding(base_tokenizer)

    # Split word at morpheme boundaries
    morphemes = []
    prev_boundary = 0
    for boundary in morpheme_boundaries + [len(word)]:
        morpheme = word[prev_boundary:boundary]
        if morpheme:
            morphemes.append(morpheme)
        prev_boundary = boundary

    # Tokenize each morpheme
    all_tokens = []
    all_token_ids = []

    for morpheme in morphemes:
        token_ids = tokenizer.encode(morpheme)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        all_tokens.extend(tokens)
        all_token_ids.extend(token_ids)

    return all_tokens, all_token_ids


def find_token_boundaries(word: str, tokens: List[str]) -> List[int]:
    """
    Find character positions where tokens boundaries occur.

    Example:
        word = "tumakbo"
        tokens = ["tum", "ak", "bo"]
        boundaries = [3, 5, 7]
    """
    boundaries = []
    pos = 0

    for token in tokens:
        pos += len(token)
        if pos < len(word):
            boundaries.append(pos)

    return boundaries


def create_morphological_annotation(
    word: str,
    morpheme_boundaries: List[int],
    token_boundaries: List[int],
    morphemes: List[str],
) -> MorphologicalAnnotation:
    """Create annotation for morphological metrics."""
    return MorphologicalAnnotation(
        word=word,
        morphemes=morphemes,
        morpheme_boundaries=set(morpheme_boundaries),
        tokens=[],  # Not needed for boundary analysis
        token_boundaries=set(token_boundaries),
    )


def analyze_tokenization(annotations: List[Dict], tokenizer_name: str):
    """Analyze tokenization for all annotated words."""
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {tokenizer_name.upper()}")
    print(f"{'=' * 70}\n")

    results = []
    morph_annotations = []

    for i, ann in enumerate(annotations[:100]):  # Analyze first 100 for speed
        word = ann["word"]
        morphemes = ann["morphemes"]
        morpheme_boundaries = ann.get("boundaries", [])

        # Tokenize
        if tokenizer_name == "gpt2":
            tokens, token_ids = tokenize_with_gpt2(word)
        elif tokenizer_name == "patok":
            tokens, token_ids = tokenize_with_patok(word, morpheme_boundaries)
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

        # Find token boundaries
        token_boundaries = find_token_boundaries(word, tokens)

        # Create annotation
        morph_ann = create_morphological_annotation(word, morpheme_boundaries, token_boundaries, morphemes)
        morph_annotations.append(morph_ann)

        # Record results
        results.append(
            {
                "word": word,
                "num_morphemes": len(morphemes),
                "num_tokens": len(tokens),
                "morpheme_boundaries": morpheme_boundaries,
                "token_boundaries": token_boundaries,
                "tokens": tokens,
                "morphemes": morphemes,
                "fragmentation": len(tokens) / len(morphemes) if morphemes else 0,
            }
        )

    # Compute morphological metrics
    print("Computing morphological metrics...")
    metrics = MorphologicalMetrics()

    morph_score = metrics.compute_morph_score(morph_annotations)
    boundary_f1 = metrics.compute_boundary_f1(morph_annotations)
    fragmentation = metrics.compute_fragmentation(morph_annotations)

    print(f"\n{generate_morphological_report(morph_annotations)}")

    # Compute information-theoretic metrics
    print("\nComputing information-theoretic metrics...")
    alignments = []
    for ann, res in zip(morph_annotations, results):
        alignment = MorphemeTokenAlignment(morphemes=tuple(res["morphemes"]), tokens=tuple(res["tokens"]))
        alignments.append(alignment)

    InformationTheoreticAnalysis(alignments)

    print(f"\n{generate_information_theoretic_report(alignments)}")

    return {
        "tokenizer": tokenizer_name,
        "results": results,
        "morph_score": morph_score,
        "boundary_f1": boundary_f1,
        "fragmentation": fragmentation,
        "morphological_annotations": morph_annotations,
        "it_alignments": alignments,
    }


def compare_tokenizers(gpt2_analysis, patok_analysis):
    """Generate comparison report."""
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}\n")

    metrics = [
        ("MorphScore", "morph_score"),
        ("Boundary F1", "boundary_f1"),
        ("Fragmentation", "fragmentation"),
    ]

    print(f"{'Metric':<20} {'GPT-2':<15} {'Patok':<15} {'Δ':<15}")
    print("-" * 70)

    for metric_name, metric_key in metrics:
        gpt2_val = gpt2_analysis[metric_key]
        patok_val = patok_analysis[metric_key]

        if isinstance(gpt2_val, dict):
            # Handle F1 dict
            gpt2_val = gpt2_val.get("f1", 0)
            patok_val = patok_val.get("f1", 0)

        delta = patok_val - gpt2_val
        delta_str = f"{delta:+.3f}"

        print(f"{metric_name:<20} {gpt2_val:<15.3f} {patok_val:<15.3f} {delta_str:<15}")

    print()

    # Interpretation
    print("Interpretation:")
    print("-" * 70)

    morph_delta = patok_analysis["morph_score"] - gpt2_analysis["morph_score"]
    if morph_delta > 0.1:
        print("✓ Patok shows significantly better morpheme boundary alignment")
    elif morph_delta > 0.05:
        print("→ Patok shows moderate improvement in boundary alignment")
    else:
        print("⚠ Little difference in morpheme boundary alignment")

    frag_delta = patok_analysis["fragmentation"] - gpt2_analysis["fragmentation"]
    if frag_delta < -0.2:
        print("✓ Patok produces less fragmented representations")
    elif frag_delta < -0.1:
        print("→ Patok shows moderate reduction in fragmentation")
    else:
        print("⚠ Similar fragmentation levels")

    print()


def main():
    """Run comprehensive tokenization analysis."""
    print("=" * 70)
    print("TOKENIZATION-ONLY ANALYSIS")
    print("=" * 70)

    # Load annotations
    print("\nLoading annotations...")
    annotations = load_annotations("data/corpora/affix_annotations.jsonl")
    print(f"Loaded {len(annotations)} annotated words")

    # Analyze GPT-2
    gpt2_analysis = analyze_tokenization(annotations, "gpt2")

    # Analyze Patok
    patok_analysis = analyze_tokenization(annotations, "patok")

    # Compare
    compare_tokenizers(gpt2_analysis, patok_analysis)

    # Save results
    print("\nSaving results...")
    results = {
        "gpt2": {
            "morph_score": gpt2_analysis["morph_score"],
            "boundary_f1": gpt2_analysis["boundary_f1"],
            "fragmentation": gpt2_analysis["fragmentation"],
        },
        "patok": {
            "morph_score": patok_analysis["morph_score"],
            "boundary_f1": patok_analysis["boundary_f1"],
            "fragmentation": patok_analysis["fragmentation"],
        },
    }

    with open("results/tokenization_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("✓ Results saved to results/tokenization_analysis.json")

    print("\n" + "=" * 70)
    print("✅ TOKENIZATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
