"""
Morphological Alignment Metrics

Quantitative measures of how well tokenizations align with morphological structure.

Key Metrics:
1. MorphScore: Alignment between token boundaries and morpheme boundaries
2. Affix Preservation Score: How often affixes appear as complete tokens
3. Affix Consistency Entropy: Consistency of affix tokenization across words
4. Boundary Alignment F1: Precision/recall of morpheme boundary detection

References:
- "Why do language models perform worse for morphologically complex languages?"
- "Rethinking Tokenization for Rich Morphology" (MorphScore definition)
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class MorphologicalAnnotation:
    """Morphological annotation for a word."""

    word: str
    morphemes: List[str]  # e.g., ["nag", "luto"] for "nagluto"
    morpheme_boundaries: List[int]  # Character positions of boundaries
    affix_types: List[str]  # e.g., ["prefix", "root"] for each morpheme


class MorphologicalMetrics:
    """Compute morphological alignment metrics for tokenizations."""

    def __init__(self, tokenizer):
        """
        Initialize metrics calculator.

        Args:
            tokenizer: Tokenizer object with encode() and decode() methods
        """
        self.tokenizer = tokenizer

    def compute_morph_score(self, annotations: List[MorphologicalAnnotation], normalize: bool = True) -> float:
        """
        Compute MorphScore: alignment between token and morpheme boundaries.

        MorphScore = (# token boundaries that align with morpheme boundaries) / (# total morpheme boundaries)  # noqa: E501

        Higher score = better alignment

        Args:
            annotations: List of morphological annotations
            normalize: Whether to normalize by number of morpheme boundaries

        Returns:
            MorphScore (0.0 to 1.0 if normalized)
        """
        total_aligned = 0
        total_morpheme_boundaries = 0

        for ann in annotations:
            # Get token boundaries
            tokens = self._tokenize_word(ann.word)
            token_boundaries = self._get_token_boundaries(ann.word, tokens)

            # Get morpheme boundaries
            morpheme_boundaries = set(ann.morpheme_boundaries)

            # Count alignments
            aligned = len(token_boundaries & morpheme_boundaries)

            total_aligned += aligned
            total_morpheme_boundaries += len(morpheme_boundaries)

        if normalize and total_morpheme_boundaries > 0:
            return total_aligned / total_morpheme_boundaries
        return total_aligned

    def compute_affix_preservation_score(
        self,
        annotations: List[MorphologicalAnnotation],
        affix_types: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute how often affixes appear as complete tokens.

        Args:
            annotations: List of morphological annotations
            affix_types: Which affix types to consider (e.g., {"prefix", "suffix"})
                        If None, considers all non-root morphemes

        Returns:
            Dictionary with:
            - "overall": Overall preservation rate
            - "by_type": Preservation rate by affix type (prefix, suffix, etc.)
            - "by_affix": Preservation rate for each specific affix
        """
        if affix_types is None:
            affix_types = {"prefix", "suffix", "infix", "circumfix"}

        affix_counts = defaultdict(int)  # affix -> total occurrences
        affix_preserved = defaultdict(int)  # affix -> times preserved
        type_counts = defaultdict(int)  # type -> total occurrences
        type_preserved = defaultdict(int)  # type -> times preserved

        for ann in annotations:
            tokens = self._tokenize_word(ann.word)

            for morpheme, mtype in zip(ann.morphemes, ann.affix_types):
                if mtype not in affix_types:
                    continue

                # Count occurrence
                affix_counts[morpheme] += 1
                type_counts[mtype] += 1

                # Check if morpheme appears as a complete token
                if morpheme in tokens:
                    affix_preserved[morpheme] += 1
                    type_preserved[mtype] += 1

        # Compute scores
        total_affixes = sum(affix_counts.values())
        total_preserved = sum(affix_preserved.values())

        results = {
            "overall": total_preserved / total_affixes if total_affixes > 0 else 0.0,
            "by_type": {
                mtype: (type_preserved[mtype] / type_counts[mtype] if type_counts[mtype] > 0 else 0.0)
                for mtype in affix_types
            },
            "by_affix": {
                affix: (affix_preserved[affix] / affix_counts[affix] if affix_counts[affix] > 0 else 0.0)
                for affix in affix_counts.keys()
            },
            "counts": {
                "total_affixes": total_affixes,
                "total_preserved": total_preserved,
            },
        }

        return results

    def compute_affix_consistency_entropy(
        self,
        words_with_affixes: List[Tuple[str, str]],
    ) -> Dict[str, float]:
        """
        Compute entropy of affix tokenization patterns.

        Lower entropy = more consistent tokenization (good)
        Higher entropy = affix tokenized differently each time (bad)

        Args:
            words_with_affixes: List of (word, affix) pairs
                                e.g., [("nagluto", "nag"), ("naglaba", "nag")]

        Returns:
            Dictionary: {affix: entropy}
        """
        affix_tokenizations = defaultdict(list)

        for word, affix in words_with_affixes:
            # Find how the affix is tokenized in this word
            tokens = self._tokenize_word(word)
            affix_pattern = self._find_affix_pattern(word, affix, tokens)
            if affix_pattern:
                affix_tokenizations[affix].append(affix_pattern)

        # Compute entropy for each affix
        entropies = {}
        for affix, patterns in affix_tokenizations.items():
            if len(patterns) > 0:
                # Count pattern frequencies
                pattern_counts = Counter(patterns)
                total = len(patterns)

                # Compute entropy: -sum(p * log(p))
                entropy = 0.0
                for count in pattern_counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * np.log2(p)

                entropies[affix] = entropy
            else:
                entropies[affix] = 0.0

        return entropies

    def compute_boundary_alignment_f1(self, annotations: List[MorphologicalAnnotation]) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 for morpheme boundary detection.

        Treats tokenizer boundaries as "predictions" and morpheme boundaries as "gold".

        Returns:
            Dictionary with "precision", "recall", "f1"
        """
        total_tp = 0  # True positives: token boundary = morpheme boundary
        total_fp = 0  # False positives: token boundary ≠ morpheme boundary
        total_fn = 0  # False negatives: morpheme boundary missed by tokenizer

        for ann in annotations:
            tokens = self._tokenize_word(ann.word)
            token_boundaries = self._get_token_boundaries(ann.word, tokens)
            morpheme_boundaries = set(ann.morpheme_boundaries)

            # All possible boundary positions
            all_positions = set(range(1, len(ann.word)))

            for pos in all_positions:
                token_has_boundary = pos in token_boundaries
                morpheme_has_boundary = pos in morpheme_boundaries

                if token_has_boundary and morpheme_has_boundary:
                    total_tp += 1
                elif token_has_boundary and not morpheme_has_boundary:
                    total_fp += 1
                elif not token_has_boundary and morpheme_has_boundary:
                    total_fn += 1

        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

    def compute_morpheme_fragmentation(self, annotations: List[MorphologicalAnnotation]) -> Dict[str, float]:
        """
        Measure how fragmented morphemes are in tokenization.

        Fragmentation = avg # of tokens per morpheme

        Lower is better (ideally 1.0 = one token per morpheme)

        Returns:
            Dictionary with overall and per-affix-type fragmentation
        """
        fragmentations = []
        type_fragmentations = defaultdict(list)

        for ann in annotations:
            tokens = self._tokenize_word(ann.word)

            for morpheme, mtype in zip(ann.morphemes, ann.affix_types):
                # Count how many tokens overlap with this morpheme
                n_tokens = self._count_tokens_for_morpheme(ann.word, morpheme, tokens)
                fragmentations.append(n_tokens)
                type_fragmentations[mtype].append(n_tokens)

        results = {
            "overall": np.mean(fragmentations) if fragmentations else 0.0,
            "std": np.std(fragmentations) if fragmentations else 0.0,
            "by_type": {mtype: np.mean(frags) if frags else 0.0 for mtype, frags in type_fragmentations.items()},
        }

        return results

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a word and return token strings."""
        # Assuming tokenizer has encode method that returns token IDs
        token_ids = self.tokenizer.encode(word)

        # Decode each token separately
        tokens = []
        for token_id in token_ids:
            token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
            try:
                token_str = token_bytes.decode("utf-8")
                tokens.append(token_str)
            except Exception:
                tokens.append(token_bytes.decode("utf-8", errors="replace"))

        return tokens

    def _get_token_boundaries(self, word: str, tokens: List[str]) -> Set[int]:
        """
        Get character positions of token boundaries.

        Returns: Set of positions (0 to len(word))
                 Position i means boundary before character i
        """
        boundaries = set()
        pos = 0

        for token in tokens:
            pos += len(token)
            if pos < len(word):
                boundaries.add(pos)

        return boundaries

    def _find_affix_pattern(self, word: str, affix: str, tokens: List[str]) -> Optional[str]:
        """
        Find how an affix is tokenized within a word.

        Returns:
            String representation of tokenization pattern
            e.g., "nag" might be "nag" (single token) or "na+g" (two tokens)
        """
        # Find where affix appears in word
        start_idx = word.find(affix)
        if start_idx == -1:
            return None

        end_idx = start_idx + len(affix)

        # Find which tokens cover this span
        pattern_tokens = []
        pos = 0

        for token in tokens:
            token_start = pos
            token_end = pos + len(token)

            # Check if this token overlaps with affix
            if token_end > start_idx and token_start < end_idx:
                # Compute overlap
                overlap_start = max(token_start, start_idx)
                overlap_end = min(token_end, end_idx)
                overlap = word[overlap_start:overlap_end]
                pattern_tokens.append(overlap)

            pos = token_end

        if pattern_tokens:
            return "+".join(pattern_tokens)
        return None

    def _count_tokens_for_morpheme(self, word: str, morpheme: str, tokens: List[str]) -> int:
        """Count how many tokens overlap with a morpheme."""
        # Find where morpheme appears in word
        start_idx = word.find(morpheme)
        if start_idx == -1:
            return 0

        end_idx = start_idx + len(morpheme)

        # Count overlapping tokens
        count = 0
        pos = 0

        for token in tokens:
            token_start = pos
            token_end = pos + len(token)

            # Check overlap
            if token_end > start_idx and token_start < end_idx:
                count += 1

            pos = token_end

        return count


def compare_tokenizers_morphologically(
    tokenizer_dict: Dict[str, any], annotations: List[MorphologicalAnnotation]
) -> pd.DataFrame:
    """
    Compare multiple tokenizers on morphological metrics.

    Args:
        tokenizer_dict: {name: tokenizer} dictionary
        annotations: List of morphological annotations

    Returns:
        DataFrame comparing tokenizers
    """
    results = []

    for name, tokenizer in tokenizer_dict.items():
        metrics = MorphologicalMetrics(tokenizer)

        morph_score = metrics.compute_morph_score(annotations)
        affix_preservation = metrics.compute_affix_preservation_score(annotations)
        boundary_f1 = metrics.compute_boundary_alignment_f1(annotations)
        fragmentation = metrics.compute_morpheme_fragmentation(annotations)

        results.append(
            {
                "tokenizer": name,
                "morph_score": morph_score,
                "affix_preservation": affix_preservation["overall"],
                "boundary_f1": boundary_f1["f1"],
                "boundary_precision": boundary_f1["precision"],
                "boundary_recall": boundary_f1["recall"],
                "fragmentation": fragmentation["overall"],
            }
        )

    return pd.DataFrame(results).sort_values("morph_score", ascending=False)


def generate_morphological_report(
    tokenizer,
    annotations: List[MorphologicalAnnotation],
    tokenizer_name: str = "Unknown",
) -> str:
    """Generate a human-readable report on morphological alignment."""
    metrics = MorphologicalMetrics(tokenizer)

    morph_score = metrics.compute_morph_score(annotations)
    affix_preservation = metrics.compute_affix_preservation_score(annotations)
    boundary_f1 = metrics.compute_boundary_alignment_f1(annotations)
    fragmentation = metrics.compute_morpheme_fragmentation(annotations)

    report = []
    report.append("=" * 70)
    report.append(f"MORPHOLOGICAL ALIGNMENT REPORT - {tokenizer_name}")
    report.append("=" * 70)
    report.append("")

    # Overall scores
    report.append("Overall Metrics:")
    report.append("-" * 70)
    report.append(f"  MorphScore:           {morph_score:.3f}")
    report.append(f"  Affix Preservation:   {affix_preservation['overall']:.3f}")
    report.append(f"  Boundary F1:          {boundary_f1['f1']:.3f}")
    report.append(f"    Precision:          {boundary_f1['precision']:.3f}")
    report.append(f"    Recall:             {boundary_f1['recall']:.3f}")
    report.append(f"  Fragmentation:        {fragmentation['overall']:.2f} tokens/morpheme")
    report.append("")

    # Affix preservation by type
    report.append("Affix Preservation by Type:")
    report.append("-" * 70)
    for affix_type, score in affix_preservation["by_type"].items():
        if score > 0:
            report.append(f"  {affix_type:15s}: {score:.3f}")
    report.append("")

    # Fragmentation by type
    report.append("Fragmentation by Type:")
    report.append("-" * 70)
    for affix_type, frag in fragmentation["by_type"].items():
        if frag > 0:
            report.append(f"  {affix_type:15s}: {frag:.2f} tokens/morpheme")
    report.append("")

    # Interpretation
    report.append("Interpretation:")
    report.append("-" * 70)
    if morph_score > 0.7:
        report.append("  ✓ Excellent morphological alignment (>0.7)")
    elif morph_score > 0.5:
        report.append("  ✓ Good morphological alignment (0.5-0.7)")
    elif morph_score > 0.3:
        report.append("  → Moderate morphological alignment (0.3-0.5)")
    else:
        report.append("  ✗ Poor morphological alignment (<0.3)")

    if affix_preservation["overall"] > 0.5:
        report.append("  ✓ Most affixes preserved as complete tokens")
    else:
        report.append("  → Many affixes split across multiple tokens")

    if fragmentation["overall"] < 1.5:
        report.append("  ✓ Low fragmentation - morphemes mostly intact")
    elif fragmentation["overall"] < 2.5:
        report.append("  → Moderate fragmentation")
    else:
        report.append("  ✗ High fragmentation - morphemes heavily split")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)
