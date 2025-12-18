"""
Affix Decomposition Algorithm (Option 3).

For tokenizers that don't have Filipino affixes in their vocabulary,
this module finds optimal ways to represent affixes using existing tokens.

Key Idea:
- If "ikina-" is not in vocabulary, represent as "i + ki + na"
- Choose decomposition that best preserves morpheme boundaries
- Rank options by linguistic validity

Example:
    ikina- can be tokenized as:
    1. "i" + "ki" + "na"  (3 tokens, preserves structure)
    2. "ik" + "ina"       (2 tokens, splits affix oddly)
    3. "ikin" + "a"       (2 tokens, very bad split)

    We choose option 1 based on linguistic criteria.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken


@dataclass
class AffixDecomposition:
    """A decomposition of an affix into sub-tokens."""

    affix: str  # Original affix (e.g., "ikina")
    tokens: List[str]  # Decomposition (e.g., ["i", "ki", "na"])
    token_ids: List[int]  # Token IDs in vocabulary
    score: float  # Quality score (higher = better)
    decomposition_type: str  # "exact_match", "optimal", "fallback"

    def __str__(self):
        return f"{self.affix} → {' + '.join(self.tokens)} (score: {self.score:.2f})"


class AffixDecomposer:
    """Finds optimal decompositions for affixes using tokenizer vocabulary."""

    def __init__(self, tokenizer_name: str = "gpt2", affixes_file: Optional[str] = None):
        """
        Initialize decomposer.

        Args:
            tokenizer_name: Name of tiktoken encoding ("gpt2", "cl100k_base", etc.)
            affixes_file: Path to file with Filipino affixes (one per line)
        """
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab = self.tokenizer._mergeable_ranks
        self.vocab_tokens = set(self.vocab.keys())

        # Load affixes
        self.affixes = self._load_affixes(affixes_file)

        # Cache for decompositions
        self._decomposition_cache: Dict[str, AffixDecomposition] = {}

    def _load_affixes(self, affixes_file: Optional[str]) -> List[str]:
        """Load affixes from file, removing prefix/suffix markers."""
        if affixes_file is None:
            # Use default common affixes
            return [
                "um",
                "in",
                "mag",
                "nag",
                "pag",
                "ka",
                "ma",
                "pa",
                "an",
                "han",
                "nan",
            ]

        affixes = []
        with open(affixes_file) as f:
            for line in f:
                affix = line.strip()
                if affix:
                    # Remove markers: "mag-" → "mag", "-an" → "an"
                    affix = affix.replace("-", "")
                    affixes.append(affix)
        return affixes

    def is_in_vocabulary(self, affix: str) -> bool:
        """Check if affix exists as a token in vocabulary."""
        token_bytes = affix.encode("utf-8")
        return token_bytes in self.vocab_tokens

    def find_all_decompositions(self, affix: str, max_tokens: int = 5) -> List[AffixDecomposition]:
        """
        Find all possible ways to decompose affix using vocabulary tokens.

        Args:
            affix: The affix to decompose
            max_tokens: Maximum number of tokens in decomposition

        Returns:
            List of possible decompositions, sorted by score (best first)
        """
        # Check for exact match first
        if self.is_in_vocabulary(affix):
            token_bytes = affix.encode("utf-8")
            token_id = self.vocab[token_bytes]
            return [
                AffixDecomposition(
                    affix=affix,
                    tokens=[affix],
                    token_ids=[token_id],
                    score=10.0,  # Perfect score for exact match
                    decomposition_type="exact_match",
                )
            ]

        # Find all possible decompositions via BPE
        decompositions = []

        # Try all possible split points
        for n_tokens in range(2, min(len(affix) + 1, max_tokens + 1)):
            splits = self._generate_splits(affix, n_tokens)
            for split in splits:
                if self._all_in_vocabulary(split):
                    token_ids = [self.vocab[s.encode("utf-8")] for s in split]
                    score = self._score_decomposition(affix, split)
                    decompositions.append(
                        AffixDecomposition(
                            affix=affix,
                            tokens=split,
                            token_ids=token_ids,
                            score=score,
                            decomposition_type="optimal",
                        )
                    )

        # Sort by score
        decompositions.sort(key=lambda d: d.score, reverse=True)

        # If no decomposition found, use character-level fallback
        if not decompositions:
            char_decomp = self._character_level_decomposition(affix)
            if char_decomp:
                decompositions = [char_decomp]

        return decompositions

    def _generate_splits(self, text: str, n_parts: int) -> List[List[str]]:
        """
        Generate all ways to split text into n_parts.

        Example: "abc", 2 → [["a", "bc"], ["ab", "c"]]
        """
        if n_parts == 1:
            return [[text]]
        if n_parts >= len(text):
            return [[c for c in text]]

        splits = []
        for i in range(1, len(text) - n_parts + 2):
            first_part = text[:i]
            for rest in self._generate_splits(text[i:], n_parts - 1):
                splits.append([first_part] + rest)

        return splits

    def _all_in_vocabulary(self, tokens: List[str]) -> bool:
        """Check if all tokens exist in vocabulary."""
        return all(t.encode("utf-8") in self.vocab_tokens for t in tokens)

    def _score_decomposition(self, affix: str, tokens: List[str]) -> float:
        """
        Score a decomposition based on linguistic criteria.

        Higher score = better decomposition

        Criteria:
        1. Fewer tokens is better (1-3 tokens good, 4+ tokens bad)
        2. Balanced token lengths preferred
        3. Preserve common morpheme patterns (e.g., CV structure)
        4. Penalize very short tokens (single consonants usually bad)
        5. Bonus for linguistically meaningful splits
        """
        score = 5.0  # Base score

        # Criterion 1: Token count (fewer is better)
        if len(tokens) == 1:
            score += 5.0  # Exact match (shouldn't happen here, but just in case)
        elif len(tokens) == 2:
            score += 3.0  # Good
        elif len(tokens) == 3:
            score += 1.5  # OK
        else:
            score -= (len(tokens) - 3) * 1.0  # Penalize many tokens

        # Criterion 2: Token length balance
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        length_variance = sum((len(t) - avg_len) ** 2 for t in tokens) / len(tokens)
        if length_variance < 1.0:
            score += 1.0  # Balanced lengths

        # Criterion 3: Penalize very short tokens
        for token in tokens:
            if len(token) == 1:
                # Single consonant is usually bad
                if token.lower() not in "aeiou":
                    score -= 1.0
                # Single vowel is OK
                else:
                    score += 0.5

        # Criterion 4: Preserve CV structure
        # Filipino morphemes often have consonant-vowel patterns
        for token in tokens:
            if len(token) >= 2:
                # Check if starts with consonant and contains vowel
                if token[0].lower() not in "aeiou" and any(c in "aeiou" for c in token.lower()):
                    score += 0.5

        # Criterion 5: Bonus for linguistically meaningful splits
        # Common morpheme patterns in Filipino
        meaningful_units = {
            "um",
            "in",
            "an",
            "ka",
            "ma",
            "pa",
            "na",
            "mag",
            "nag",
            "pag",
        }
        for token in tokens:
            if token.lower() in meaningful_units:
                score += 1.5

        return max(score, 0.0)  # Ensure non-negative

    def _character_level_decomposition(self, affix: str) -> Optional[AffixDecomposition]:
        """
        Fallback: decompose into individual characters.

        This should always work since all characters should be in vocabulary.
        """
        tokens = []
        token_ids = []

        for char in affix:
            char_bytes = char.encode("utf-8")
            if char_bytes in self.vocab_tokens:
                tokens.append(char)
                token_ids.append(self.vocab[char_bytes])
            else:
                # This character isn't in vocabulary - very rare
                return None

        if not tokens:
            return None

        return AffixDecomposition(
            affix=affix,
            tokens=tokens,
            token_ids=token_ids,
            score=0.5,  # Low score for character-level fallback
            decomposition_type="fallback",
        )

    def get_best_decomposition(self, affix: str, use_cache: bool = True) -> Optional[AffixDecomposition]:
        """
        Get the best decomposition for an affix.

        Args:
            affix: The affix to decompose
            use_cache: Whether to use cached result

        Returns:
            Best decomposition, or None if affix cannot be decomposed
        """
        if use_cache and affix in self._decomposition_cache:
            return self._decomposition_cache[affix]

        decompositions = self.find_all_decompositions(affix)
        if decompositions:
            best = decompositions[0]
            self._decomposition_cache[affix] = best
            return best

        return None

    def analyze_vocabulary_coverage(self) -> Dict[str, any]:
        """
        Analyze which affixes are in vocabulary and which need decomposition.

        Returns:
            Dictionary with coverage statistics
        """
        in_vocab = []
        need_decomposition = []
        cannot_decompose = []

        for affix in self.affixes:
            if self.is_in_vocabulary(affix):
                in_vocab.append(affix)
            else:
                decomp = self.get_best_decomposition(affix)
                if decomp:
                    need_decomposition.append((affix, decomp))
                else:
                    cannot_decompose.append(affix)

        return {
            "total_affixes": len(self.affixes),
            "in_vocabulary": len(in_vocab),
            "need_decomposition": len(need_decomposition),
            "cannot_decompose": len(cannot_decompose),
            "coverage_rate": len(in_vocab) / len(self.affixes) if self.affixes else 0,
            "in_vocab_list": in_vocab,
            "decomposition_list": need_decomposition,
            "cannot_decompose_list": cannot_decompose,
        }

    def build_decomposition_table(self) -> Dict[str, List[int]]:
        """
        Build a lookup table mapping affixes to token ID sequences.

        Returns:
            Dictionary: {affix: [token_id_1, token_id_2, ...]}

        This table can be used by Patok to preferentially form these token sequences.
        """
        table = {}

        for affix in self.affixes:
            decomp = self.get_best_decomposition(affix)
            if decomp:
                table[affix] = decomp.token_ids

        return table

    def export_analysis(self, output_file: str):
        """
        Export detailed analysis to JSON file.

        Includes:
        - Vocabulary coverage statistics
        - Best decomposition for each affix
        - Alternative decompositions
        """
        analysis = self.analyze_vocabulary_coverage()

        # Add detailed decomposition info
        decomposition_details = {}
        for affix in self.affixes:
            decompositions = self.find_all_decompositions(affix)
            if decompositions:
                decomposition_details[affix] = [
                    {
                        "tokens": d.tokens,
                        "token_ids": d.token_ids,
                        "score": d.score,
                        "type": d.decomposition_type,
                    }
                    for d in decompositions[:3]  # Top 3 options
                ]

        export_data = {
            "tokenizer": self.tokenizer.name,
            "coverage_summary": {
                "total_affixes": analysis["total_affixes"],
                "in_vocabulary": analysis["in_vocabulary"],
                "need_decomposition": analysis["need_decomposition"],
                "coverage_rate": analysis["coverage_rate"],
            },
            "in_vocabulary": analysis["in_vocab_list"],
            "decompositions": decomposition_details,
            "cannot_decompose": analysis["cannot_decompose_list"],
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported affix analysis to {output_file}")

    def generate_report(self) -> str:
        """Generate a human-readable report on affix coverage."""
        analysis = self.analyze_vocabulary_coverage()

        report = []
        report.append("=" * 70)
        report.append(f"AFFIX VOCABULARY ANALYSIS - {self.tokenizer.name}")
        report.append("=" * 70)
        report.append("")

        # Summary
        report.append("Summary:")
        report.append(f"  Total affixes: {analysis['total_affixes']}")
        report.append(f"  In vocabulary: {analysis['in_vocabulary']} ({analysis['coverage_rate']:.1%})")
        report.append(f"  Need decomposition: {analysis['need_decomposition']}")
        report.append(f"  Cannot decompose: {analysis['cannot_decompose']}")
        report.append("")

        # Affixes in vocabulary
        if analysis["in_vocab_list"]:
            report.append("Affixes in Vocabulary (can use directly):")
            report.append("-" * 70)
            for affix in sorted(analysis["in_vocab_list"])[:20]:  # Show first 20
                report.append(f"  ✓ {affix}")
            if len(analysis["in_vocab_list"]) > 20:
                report.append(f"  ... and {len(analysis['in_vocab_list']) - 20} more")
            report.append("")

        # Decompositions needed
        if analysis["decomposition_list"]:
            report.append("Affixes Needing Decomposition:")
            report.append("-" * 70)
            for affix, decomp in sorted(analysis["decomposition_list"], key=lambda x: x[1].score, reverse=True)[:15]:
                report.append(f"  {decomp}")
            if len(analysis["decomposition_list"]) > 15:
                report.append(f"  ... and {len(analysis['decomposition_list']) - 15} more")
            report.append("")

        # Cannot decompose
        if analysis["cannot_decompose_list"]:
            report.append("⚠ Affixes That Cannot Be Decomposed:")
            report.append("-" * 70)
            for affix in analysis["cannot_decompose_list"]:
                report.append(f"  ✗ {affix}")
            report.append("")

        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 70)
        if analysis["coverage_rate"] < 0.3:
            report.append("  ⚠ Low coverage (<30%) - consider training custom tokenizer")
        elif analysis["coverage_rate"] < 0.6:
            report.append("  → Moderate coverage - Option 3 (decomposition) recommended")
        else:
            report.append("  ✓ Good coverage (>60%) - can work with existing vocabulary")

        if analysis["need_decomposition"] > 0:
            report.append(f"  → Use decomposition table for {analysis['need_decomposition']} affixes")

        if analysis["cannot_decompose"] > 0:
            report.append(f"  ⚠ {analysis['cannot_decompose']} affixes cannot be represented")
            report.append("     Consider adding to vocabulary (Option 2)")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def compare_tokenizers(tokenizer_names: List[str], affixes_file: str):
    """
    Compare affix coverage across multiple tokenizers.

    Args:
        tokenizer_names: List of tiktoken encoding names
        affixes_file: Path to affixes file

    Returns:
        DataFrame comparing coverage statistics
    """
    import pandas as pd

    results = []
    for tokenizer_name in tokenizer_names:
        decomposer = AffixDecomposer(tokenizer_name, affixes_file)
        analysis = decomposer.analyze_vocabulary_coverage()

        results.append(
            {
                "tokenizer": tokenizer_name,
                "total_affixes": analysis["total_affixes"],
                "in_vocabulary": analysis["in_vocabulary"],
                "coverage_rate": analysis["coverage_rate"],
                "need_decomposition": analysis["need_decomposition"],
                "cannot_decompose": analysis["cannot_decompose"],
            }
        )

    return pd.DataFrame(results)
