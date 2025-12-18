"""
Compare Tokenization Approaches.

Comprehensive comparison of GPT-2 baseline vs morphologically-aware tokenization
on Filipino words with affix annotations.

Metrics:
- MorphScore: Morpheme boundary alignment
- Fragmentation: Tokens per morpheme
- Boundary F1: Precision and recall
- Example-level analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import tiktoken

from src.tokenization.patok_morphology import MorphologyAwarePatokProcessor

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


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


def tokenize_patok_style(word: str, morpheme_boundaries: List[int], tokenizer) -> List[str]:
    """
    Morphologically-aware tokenization (Patok-style).

    Splits at morpheme boundaries before applying BPE.
    This respects morphological structure.
    """
    # Split at morpheme boundaries
    morphemes = []
    prev = 0
    for boundary in morpheme_boundaries + [len(word)]:
        if prev < boundary:
            morphemes.append(word[prev:boundary])
        prev = boundary

    # Tokenize each morpheme independently
    all_tokens = []
    for morpheme in morphemes:
        tokens = tokenize_word(morpheme, tokenizer)
        all_tokens.extend(tokens)

    return all_tokens


def find_token_boundaries(word: str, tokens: List[str]) -> List[int]:
    """Find character positions of token boundaries."""
    boundaries = []
    pos = 0
    for token in tokens:
        pos += len(token)
        if pos < len(word):
            boundaries.append(pos)
    return boundaries


def compute_boundary_f1(token_boundaries: set, morpheme_boundaries: set) -> Dict[str, float]:
    """Compute precision, recall, and F1 for boundary detection."""
    if not morpheme_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not token_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(token_boundaries & morpheme_boundaries)
    precision = true_positives / len(token_boundaries) if token_boundaries else 0.0
    recall = true_positives / len(morpheme_boundaries) if morpheme_boundaries else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def analyze_tokenizer(annotations: List[Dict], tokenizer, mode: str = "baseline", patok_processor=None) -> Dict:
    """
    Analyze tokenization for all annotated words.

    Args:
        annotations: List of word annotations
        tokenizer: The tokenizer to use
        mode: "baseline", "patok_oracle", or "patok_real"
        patok_processor: MorphologyAwarePatokProcessor instance (for patok_real mode)
    """
    results = []
    total_aligned = 0
    total_morpheme_boundaries = 0
    total_tokens = 0
    total_morphemes = 0

    all_precision = []
    all_recall = []
    all_f1 = []

    for ann in annotations:
        word = ann["word"]
        morphemes = ann["morphemes"]
        morpheme_boundaries = set(ann.get("boundaries", []))

        if not morpheme_boundaries:
            continue

        # Tokenize based on mode
        if mode == "baseline":
            tokens = tokenize_word(word, tokenizer)
        elif mode == "patok_oracle":
            tokens = tokenize_patok_style(word, sorted(morpheme_boundaries), tokenizer)
        elif mode == "patok_real":
            if patok_processor is None:
                raise ValueError("patok_processor required for patok_real mode")
            token_ids = patok_processor.process(word, disable_tqdm=True)
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        token_boundaries = set(find_token_boundaries(word, tokens))

        # Compute metrics
        aligned = len(token_boundaries & morpheme_boundaries)
        f1_scores = compute_boundary_f1(token_boundaries, morpheme_boundaries)

        total_aligned += aligned
        total_morpheme_boundaries += len(morpheme_boundaries)
        total_tokens += len(tokens)
        total_morphemes += len(morphemes)

        all_precision.append(f1_scores["precision"])
        all_recall.append(f1_scores["recall"])
        all_f1.append(f1_scores["f1"])

        results.append(
            {
                "word": word,
                "morphemes": morphemes,
                "tokens": tokens,
                "morph_boundaries": sorted(morpheme_boundaries),
                "token_boundaries": sorted(token_boundaries),
                "aligned": aligned,
                "total_morph": len(morpheme_boundaries),
                "score": (aligned / len(morpheme_boundaries) if morpheme_boundaries else 0),
                "f1": f1_scores["f1"],
                "fragmentation": len(tokens) / len(morphemes) if morphemes else 0,
            }
        )

    morph_score = total_aligned / total_morpheme_boundaries if total_morpheme_boundaries > 0 else 0
    fragmentation = total_tokens / total_morphemes if total_morphemes > 0 else 0

    avg_precision = sum(all_precision) / len(all_precision) if all_precision else 0
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

    return {
        "mode": mode,
        "morph_score": morph_score,
        "fragmentation": fragmentation,
        "boundary_f1": {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1},
        "total_aligned": total_aligned,
        "total_boundaries": total_morpheme_boundaries,
        "total_tokens": total_tokens,
        "total_morphemes": total_morphemes,
        "results": results,
    }


def print_comparison(gpt2_analysis: Dict, patok_analysis: Dict):
    """Print side-by-side comparison."""
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    metrics = [
        ("MorphScore", "morph_score", "Higher is better", 1),
        ("Fragmentation", "fragmentation", "Lower is better", -1),
        ("Boundary Precision", "boundary_f1.precision", "Higher is better", 1),
        ("Boundary Recall", "boundary_f1.recall", "Higher is better", 1),
        ("Boundary F1", "boundary_f1.f1", "Higher is better", 1),
    ]

    print(f"{'Metric':<25} {'GPT-2':<15} {'Patok':<15} {'Δ':<15} {'Winner':<10}")
    print("-" * 80)

    for metric_name, metric_path, direction, sign in metrics:
        # Get values
        gpt2_val = gpt2_analysis
        patok_val = patok_analysis
        for key in metric_path.split("."):
            gpt2_val = gpt2_val[key]
            patok_val = patok_val[key]

        delta = patok_val - gpt2_val
        delta_pct = (delta / gpt2_val * 100) if gpt2_val != 0 else 0
        delta_str = f"{delta:+.3f} ({delta_pct:+.1f}%)"

        # Determine winner
        if abs(delta) < 0.01:
            winner = "Tie"
        elif sign * delta > 0:
            winner = "Patok ✓"
        else:
            winner = "GPT-2"

        print(f"{metric_name:<25} {gpt2_val:<15.3f} {patok_val:<15.3f} {delta_str:<15} {winner:<10}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    morph_delta = patok_analysis["morph_score"] - gpt2_analysis["morph_score"]
    if morph_delta > 0.2:
        print("✓ Patok shows SIGNIFICANT improvement in morpheme boundary alignment")
        print(f"  → {morph_delta:.1%} absolute improvement in MorphScore")
    elif morph_delta > 0.1:
        print("✓ Patok shows MODERATE improvement in morpheme boundary alignment")
        print(f"  → {morph_delta:.1%} absolute improvement")
    elif morph_delta > 0:
        print("→ Patok shows slight improvement in morpheme boundary alignment")
    else:
        print("⚠ No improvement in morpheme boundary alignment")

    print()

    frag_delta = patok_analysis["fragmentation"] - gpt2_analysis["fragmentation"]
    if frag_delta < -0.3:
        print("✓ Patok produces SIGNIFICANTLY less fragmented representations")
        print(f"  → {abs(frag_delta):.2f} fewer tokens per morpheme")
    elif frag_delta < -0.1:
        print("✓ Patok produces MODERATELY less fragmented representations")
        print(f"  → {abs(frag_delta):.2f} fewer tokens per morpheme")
    elif frag_delta < 0:
        print("→ Patok produces slightly less fragmented representations")
    else:
        print("⚠ No improvement in fragmentation")

    print()

    f1_delta = patok_analysis["boundary_f1"]["f1"] - gpt2_analysis["boundary_f1"]["f1"]
    if f1_delta > 0.2:
        print("✓ Patok shows STRONG improvement in boundary detection (F1)")
        print(f"  → {f1_delta:.1%} absolute improvement")
    elif f1_delta > 0.1:
        print("✓ Patok shows MODERATE improvement in boundary detection")
    elif f1_delta > 0:
        print("→ Patok shows slight improvement in boundary detection")

    print()


def print_examples(gpt2_analysis: Dict, patok_analysis: Dict, n: int = 10):
    """Print side-by-side examples."""
    print("=" * 80)
    print("EXAMPLE COMPARISONS")
    print("=" * 80)
    print()

    for i in range(min(n, len(gpt2_analysis["results"]))):
        gpt2_ex = gpt2_analysis["results"][i]
        patok_ex = patok_analysis["results"][i]

        word = gpt2_ex["word"]
        morphemes = " + ".join(gpt2_ex["morphemes"])

        print(f"{i + 1}. {word}")
        print(f"   Morphemes:  {morphemes}")
        print(f"   GPT-2:      {' | '.join(gpt2_ex['tokens'])}")
        print(f"        MorphScore={gpt2_ex['score']:.2f}, F1={gpt2_ex['f1']:.2f}, Frag={gpt2_ex['fragmentation']:.2f}")
        print(f"   Patok:      {' | '.join(patok_ex['tokens'])}")
        print(
            f"        MorphScore={patok_ex['score']:.2f}, F1={patok_ex['f1']:.2f}, Frag={patok_ex['fragmentation']:.2f}"
        )

        # Indicate improvement
        if patok_ex["score"] > gpt2_ex["score"]:
            print("               ✓ Better boundary alignment")
        if patok_ex["fragmentation"] < gpt2_ex["fragmentation"]:
            print("               ✓ Less fragmented")

        print()


def main():
    """Compare tokenization approaches on Filipino words."""
    print("=" * 80)
    print("TOKENIZATION COMPARISON: GPT-2 BASELINE VS MORPHOLOGICALLY-AWARE")
    print("=" * 80)
    print()

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations("data/corpora/affix_annotations.jsonl", limit=100)
    print(f"  ✓ Loaded {len(annotations)} annotated words")
    print()

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: GPT-2 (vocab size: {tokenizer.n_vocab})")
    print()

    # Analyze GPT-2 baseline
    print("Analyzing GPT-2 baseline...")
    gpt2_analysis = analyze_tokenizer(annotations, tokenizer, mode="baseline")
    print("  ✓ GPT-2 analysis complete")
    print()

    # Analyze Oracle (split at known boundaries)
    print("Analyzing Oracle tokenization (split at known boundaries)...")
    oracle_analysis = analyze_tokenizer(annotations, tokenizer, mode="patok_oracle")
    print("  ✓ Oracle analysis complete")
    print()

    # Initialize and analyze real Patok
    print("Initializing Patok processor...")
    patok_processor = MorphologyAwarePatokProcessor(tokenizer)
    print("  ✓ Patok processor initialized")
    print()

    print("Analyzing real Patok (morphology-aware)...")
    patok_analysis = analyze_tokenizer(annotations, tokenizer, mode="patok_real", patok_processor=patok_processor)
    print("  ✓ Patok analysis complete")
    print()

    # Print results
    print_comparison(gpt2_analysis, oracle_analysis)
    print()
    print("=" * 80)
    print("REAL PATOK VS BASELINE")
    print("=" * 80)
    print()
    print_comparison(gpt2_analysis, patok_analysis)
    print_examples(gpt2_analysis, patok_analysis, n=15)

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    results = {
        "gpt2": {
            "morph_score": gpt2_analysis["morph_score"],
            "fragmentation": gpt2_analysis["fragmentation"],
            "boundary_f1": gpt2_analysis["boundary_f1"],
        },
        "oracle": {
            "morph_score": oracle_analysis["morph_score"],
            "fragmentation": oracle_analysis["fragmentation"],
            "boundary_f1": oracle_analysis["boundary_f1"],
        },
        "patok": {
            "morph_score": patok_analysis["morph_score"],
            "fragmentation": patok_analysis["fragmentation"],
            "boundary_f1": patok_analysis["boundary_f1"],
        },
        "oracle_vs_baseline": {
            "morph_score": oracle_analysis["morph_score"] - gpt2_analysis["morph_score"],
            "fragmentation": oracle_analysis["fragmentation"] - gpt2_analysis["fragmentation"],
            "boundary_f1": oracle_analysis["boundary_f1"]["f1"] - gpt2_analysis["boundary_f1"]["f1"],
        },
        "patok_vs_baseline": {
            "morph_score": patok_analysis["morph_score"] - gpt2_analysis["morph_score"],
            "fragmentation": patok_analysis["fragmentation"] - gpt2_analysis["fragmentation"],
            "boundary_f1": patok_analysis["boundary_f1"]["f1"] - gpt2_analysis["boundary_f1"]["f1"],
        },
    }

    with open("results/tokenization_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to results/tokenization_comparison.json")
    print()
    print("=" * 80)
    print("✅ TOKENIZATION COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
