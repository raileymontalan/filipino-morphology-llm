"""
Information-Theoretic Analysis of Tokenizations.

Quantifies morphological information using information theory metrics:
1. Morpheme-Token Mutual Information: How much does tokenization tell us about morphology?
2. Morphological Perplexity: How surprising are morphologically complex words?
3. Morphological Conditional Entropy: Uncertainty about morphemes given tokens

These metrics provide theoretical grounding for why affix-aware tokenization helps.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MorphemeTokenAlignment:
    """Alignment between morphemes and tokens for a word."""

    word: str
    morphemes: List[str]
    tokens: List[str]
    morpheme_boundaries: List[int]  # Character positions
    token_boundaries: List[int]  # Character positions


class InformationTheoreticAnalysis:
    """Compute information-theoretic metrics for tokenization."""

    def __init__(self, tokenizer):
        """
        Initialize analyzer.

        Args:
            tokenizer: Tokenizer with encode() and decode() methods
        """
        self.tokenizer = tokenizer

    def compute_morpheme_token_mutual_information(self, alignments: List[MorphemeTokenAlignment]) -> float:
        """
        Compute mutual information between morpheme and token boundaries.

        I(Morphemes; Tokens) = H(Morphemes) - H(Morphemes | Tokens)

        Higher MI = tokenization provides more information about morphology

        Args:
            alignments: List of morpheme-token alignments

        Returns:
            Mutual information in bits
        """
        # Compute H(M): entropy of morpheme boundaries
        h_morphemes = self._compute_boundary_entropy(
            [a.morpheme_boundaries for a in alignments],
            [len(a.word) for a in alignments],
        )

        # Compute H(M|T): conditional entropy of morpheme boundaries given token boundaries
        h_morphemes_given_tokens = self._compute_conditional_boundary_entropy(
            [a.morpheme_boundaries for a in alignments],
            [a.token_boundaries for a in alignments],
            [len(a.word) for a in alignments],
        )

        # MI = H(M) - H(M|T)
        mutual_information = h_morphemes - h_morphemes_given_tokens

        return mutual_information

    def compute_morphological_perplexity(
        self, model, words_by_complexity: Dict[str, List[str]], context_fn=None
    ) -> Dict[str, float]:
        """
        Compute perplexity on words of different morphological complexity.

        Lower perplexity = model finds these words less surprising

        Args:
            model: Language model with get_log_prob(word) method
            words_by_complexity: Dictionary mapping complexity level to words
                                e.g., {"simple": [...], "affixed": [...], "complex": [...]}
            context_fn: Optional function to provide context for each word

        Returns:
            Dictionary: {complexity_level: perplexity}
        """
        perplexities = {}

        for complexity, words in words_by_complexity.items():
            log_probs = []

            for word in words:
                # Get context if available
                context = context_fn(word) if context_fn else ""

                # Get log probability from model
                log_prob = model.get_log_prob(word, context)
                log_probs.append(log_prob)

            # Compute perplexity: exp(-mean(log_prob))
            if log_probs:
                mean_log_prob = np.mean(log_probs)
                perplexity = np.exp(-mean_log_prob)
                perplexities[complexity] = perplexity
            else:
                perplexities[complexity] = float("in")

        return perplexities

    def compute_affix_consistency_information(
        self, words_with_affixes: List[Tuple[str, List[str]]]
    ) -> Dict[str, float]:
        """
        Measure information content in affix tokenizations.

        Lower entropy = more consistent (predictable) tokenization

        Args:
            words_with_affixes: List of (word, [affixes]) pairs
                               e.g., [("nagluto", ["nag"]), ("naglaba", ["nag"])]

        Returns:
            Dictionary: {affix: entropy_in_bits}
        """
        affix_tokenizations = defaultdict(list)

        for word, affixes in words_with_affixes:
            tokens = self._tokenize_word(word)

            for affix in affixes:
                # Find how this affix is tokenized
                affix_tokens = self._extract_affix_tokens(word, affix, tokens)
                if affix_tokens:
                    # Convert to string signature
                    signature = "|".join(affix_tokens)
                    affix_tokenizations[affix].append(signature)

        # Compute entropy for each affix
        entropies = {}
        for affix, signatures in affix_tokenizations.items():
            if signatures:
                entropies[affix] = self._compute_entropy(signatures)
            else:
                entropies[affix] = 0.0

        return entropies

    def compute_morphological_information_content(self, alignments: List[MorphemeTokenAlignment]) -> Dict[str, float]:
        """
        Compute information content at different levels.

        Returns:
            Dictionary with:
            - "token_entropy": Entropy of token distribution
            - "morpheme_entropy": Entropy of morpheme distribution
            - "token_given_morpheme_entropy": H(T|M)
            - "morpheme_given_token_entropy": H(M|T)
        """
        # Collect all tokens and morphemes
        all_tokens = []
        all_morphemes = []
        morpheme_to_tokens = defaultdict(list)
        token_to_morphemes = defaultdict(list)

        for alignment in alignments:
            all_tokens.extend(alignment.tokens)
            all_morphemes.extend(alignment.morphemes)

            # Build conditional distributions
            for morpheme in alignment.morphemes:
                for token in alignment.tokens:
                    # Check if they overlap
                    if self._tokens_overlap_morpheme(alignment, morpheme, token):
                        morpheme_to_tokens[morpheme].append(token)
                        token_to_morphemes[token].append(morpheme)

        # Compute entropies
        h_tokens = self._compute_entropy(all_tokens)
        h_morphemes = self._compute_entropy(all_morphemes)

        # H(T|M): Given morpheme, how uncertain about tokens?
        h_tokens_given_morphemes = 0.0
        morpheme_counts = Counter(all_morphemes)
        for morpheme, tokens in morpheme_to_tokens.items():
            p_morpheme = morpheme_counts[morpheme] / len(all_morphemes)
            h_tokens_for_morpheme = self._compute_entropy(tokens)
            h_tokens_given_morphemes += p_morpheme * h_tokens_for_morpheme

        # H(M|T): Given token, how uncertain about morphemes?
        h_morphemes_given_tokens = 0.0
        token_counts = Counter(all_tokens)
        for token, morphemes in token_to_morphemes.items():
            p_token = token_counts[token] / len(all_tokens)
            h_morphemes_for_token = self._compute_entropy(morphemes)
            h_morphemes_given_tokens += p_token * h_morphemes_for_token

        return {
            "token_entropy": h_tokens,
            "morpheme_entropy": h_morphemes,
            "token_given_morpheme_entropy": h_tokens_given_morphemes,
            "morpheme_given_token_entropy": h_morphemes_given_tokens,
            "mutual_information": h_morphemes - h_morphemes_given_tokens,
        }

    def compare_tokenization_information(
        self,
        tokenizer1,
        tokenizer2,
        alignments1: List[MorphemeTokenAlignment],
        alignments2: List[MorphemeTokenAlignment],
        tokenizer1_name: str = "Tokenizer 1",
        tokenizer2_name: str = "Tokenizer 2",
    ) -> pd.DataFrame:
        """
        Compare information-theoretic properties of two tokenizations.

        Args:
            tokenizer1, tokenizer2: Tokenizers to compare
            alignments1, alignments2: Alignments for each tokenizer
            tokenizer1_name, tokenizer2_name: Names for display

        Returns:
            DataFrame comparing metrics
        """
        analyzer1 = InformationTheoreticAnalysis(tokenizer1)
        analyzer2 = InformationTheoreticAnalysis(tokenizer2)

        mi1 = analyzer1.compute_morpheme_token_mutual_information(alignments1)
        mi2 = analyzer2.compute_morpheme_token_mutual_information(alignments2)

        info1 = analyzer1.compute_morphological_information_content(alignments1)
        info2 = analyzer2.compute_morphological_information_content(alignments2)

        comparison = pd.DataFrame(
            [
                {
                    "tokenizer": tokenizer1_name,
                    "mutual_information": mi1,
                    "token_entropy": info1["token_entropy"],
                    "morpheme_given_token_entropy": info1["morpheme_given_token_entropy"],
                },
                {
                    "tokenizer": tokenizer2_name,
                    "mutual_information": mi2,
                    "token_entropy": info2["token_entropy"],
                    "morpheme_given_token_entropy": info2["morpheme_given_token_entropy"],
                },
            ]
        )

        return comparison

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _compute_entropy(self, items: List[str]) -> float:
        """
        Compute Shannon entropy: H(X) = -sum(p(x) * log2(p(x))).

        Args:
            items: List of items (tokens, morphemes, etc.)

        Returns:
            Entropy in bits
        """
        if not items:
            return 0.0

        counts = Counter(items)
        total = len(items)

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _compute_boundary_entropy(self, boundary_lists: List[List[int]], word_lengths: List[int]) -> float:
        """
        Compute entropy of boundary positions.

        Args:
            boundary_lists: List of boundary position lists for each word
            word_lengths: Lengths of each word

        Returns:
            Entropy in bits
        """
        # Normalize boundaries to relative positions [0, 1]
        all_normalized_boundaries = []

        for boundaries, length in zip(boundary_lists, word_lengths):
            for boundary in boundaries:
                # Normalize to [0, 1]
                normalized = boundary / length
                # Discretize to bins (e.g., 10 bins)
                bin_idx = int(normalized * 10)
                all_normalized_boundaries.append(bin_idx)

        return self._compute_entropy([str(b) for b in all_normalized_boundaries])

    def _compute_conditional_boundary_entropy(
        self,
        morpheme_boundaries_list: List[List[int]],
        token_boundaries_list: List[List[int]],
        word_lengths: List[int],
    ) -> float:
        """
        Compute H(Morpheme Boundaries | Token Boundaries).

        Returns:
            Conditional entropy in bits
        """
        # For each word, compute local conditional entropy
        local_entropies = []
        weights = []

        for morpheme_boundaries, token_boundaries, length in zip(
            morpheme_boundaries_list, token_boundaries_list, word_lengths
        ):
            # For simplicity, compute: are morpheme boundaries aligned with token boundaries?
            # H(M|T) ≈ H(alignment_status)

            alignment_status = []
            for m_boundary in morpheme_boundaries:
                # Check if this morpheme boundary aligns with a token boundary
                aligned = any(abs(m_boundary - t_boundary) < 1 for t_boundary in token_boundaries)
                alignment_status.append("aligned" if aligned else "unaligned")

            if alignment_status:
                local_entropy = self._compute_entropy(alignment_status)
                local_entropies.append(local_entropy)
                weights.append(len(morpheme_boundaries))

        # Weighted average
        if local_entropies and weights:
            total_weight = sum(weights)
            weighted_entropy = sum(e * w for e, w in zip(local_entropies, weights)) / total_weight
            return weighted_entropy

        return 0.0

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a word and return token strings."""
        token_ids = self.tokenizer.encode(word)
        tokens = []

        for token_id in token_ids:
            token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
            try:
                tokens.append(token_bytes.decode("utf-8"))
            except Exception:
                tokens.append(token_bytes.decode("utf-8", errors="replace"))

        return tokens

    def _extract_affix_tokens(self, word: str, affix: str, tokens: List[str]) -> Optional[List[str]]:
        """Extract tokens that cover a specific affix in the word."""
        start_idx = word.find(affix)
        if start_idx == -1:
            return None

        end_idx = start_idx + len(affix)

        affix_tokens = []
        pos = 0

        for token in tokens:
            token_start = pos
            token_end = pos + len(token)

            # Check overlap with affix span
            if token_end > start_idx and token_start < end_idx:
                affix_tokens.append(token)

            pos = token_end

        return affix_tokens if affix_tokens else None

    def _tokens_overlap_morpheme(self, alignment: MorphemeTokenAlignment, morpheme: str, token: str) -> bool:
        """Check if a token overlaps with a morpheme in the word."""
        # Find morpheme position
        word = alignment.word
        morpheme_start = word.find(morpheme)
        if morpheme_start == -1:
            return False
        morpheme_end = morpheme_start + len(morpheme)

        # Find token position
        token_start = word.find(token)
        if token_start == -1:
            return False
        token_end = token_start + len(token)

        # Check overlap
        return token_end > morpheme_start and token_start < morpheme_end


def generate_information_theoretic_report(
    tokenizer, alignments: List[MorphemeTokenAlignment], tokenizer_name: str = "Unknown"
) -> str:
    """Generate a human-readable information-theoretic report."""
    analyzer = InformationTheoreticAnalysis(tokenizer)

    mi = analyzer.compute_morpheme_token_mutual_information(alignments)
    info_content = analyzer.compute_morphological_information_content(alignments)

    report = []
    report.append("=" * 70)
    report.append(f"INFORMATION-THEORETIC ANALYSIS - {tokenizer_name}")
    report.append("=" * 70)
    report.append("")

    report.append("Mutual Information:")
    report.append("-" * 70)
    report.append(f"  I(Morphemes; Tokens) = {mi:.3f} bits")
    report.append("")
    if mi > 1.0:
        report.append("  → High MI: Tokenization provides substantial information about morphology")
    elif mi > 0.5:
        report.append("  → Moderate MI: Some morphological information captured")
    else:
        report.append("  → Low MI: Tokenization largely independent of morphology")
    report.append("")

    report.append("Entropy Analysis:")
    report.append("-" * 70)
    report.append(f"  H(Tokens)               = {info_content['token_entropy']:.3f} bits")
    report.append(f"  H(Morphemes)            = {info_content['morpheme_entropy']:.3f} bits")
    report.append(f"  H(Morphemes | Tokens)   = {info_content['morpheme_given_token_entropy']:.3f} bits")
    report.append(f"  H(Tokens | Morphemes)   = {info_content['token_given_morpheme_entropy']:.3f} bits")
    report.append("")

    report.append("Interpretation:")
    report.append("-" * 70)
    report.append(f"  Knowing tokens reduces morpheme uncertainty by {mi:.3f} bits")
    reduction_pct = (mi / info_content["morpheme_entropy"] * 100) if info_content["morpheme_entropy"] > 0 else 0
    report.append(f"  ({reduction_pct:.1f}% of morpheme entropy)")
    report.append("")

    report.append("=" * 70)

    return "\n".join(report)
