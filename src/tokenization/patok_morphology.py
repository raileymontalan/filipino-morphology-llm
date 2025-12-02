# -*- coding: utf-8 -*-
"""
Morphology-aware Patok tokenization for Filipino.

This module implements affix-aware and reduplication-aware tokenization
using Aho-Corasick automaton for efficient affix detection.
"""

import os
import random
import numpy as np
import ahocorasick
from tqdm import tqdm
from typing import List, Tuple, Optional


class MorphologyAwarePatokProcessor:
    """
    Patok processor with morphological awareness for Filipino.

    Features:
    - Aho-Corasick automaton for fast affix detection
    - Affix-aware contraction (avoids breaking known affixes)
    - Morphological re-expansion (splits off affixes and duplications)
    - Configurable affix awareness probability
    """

    def __init__(
        self,
        tokenizer,
        affixes: Optional[List[str]] = None,
        affix_awareness: float = 0.95,
        affix_awareness_if_overlap: float = 0.75,
        expand_prop: float = 0.1,
        contract_prop: float = 0.9,
    ):
        """
        Initialize morphology-aware Patok processor.

        Args:
            tokenizer: Tokenizer with encode/decode methods
            affixes: List of Filipino affixes. If None, loads from default file
            affix_awareness: Probability of skipping contraction if token is affix
            affix_awareness_if_overlap: Affix awareness if multiple affixes at same position
            expand_prop: Default proportion of tokens to expand
            contract_prop: Default proportion of tokens to contract
        """
        self.tokenizer = tokenizer
        self.affix_awareness = affix_awareness
        self.affix_awareness_if_overlap = affix_awareness_if_overlap
        self.expand_prop = expand_prop
        self.contract_prop = contract_prop

        # Load affixes and build automaton
        self.affixes = self._load_affixes(affixes)
        self.affix_finder = self._build_affix_finder(self.affixes)

        # Get affix token IDs
        self.affix_ids = self._get_affix_token_ids()

        # Build expansion dictionary
        self.expansions = self._build_expansions()

        print(f"Initialized MorphologyAwarePatokProcessor:")
        print(f"  - {len(self.affixes)} affixes loaded")
        print(f"  - {len(self.affix_ids)} affix token IDs")
        print(f"  - {len(self.expansions)} expandable tokens")

    def _load_affixes(self, affixes: Optional[List[str]]) -> List[str]:
        """Load affixes from list or file."""
        if affixes is not None:
            return affixes

        # Try to load from default location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        affixes_path = os.path.join(project_root, "data", "affixes", "filipino_affixes.txt")

        if os.path.exists(affixes_path):
            with open(affixes_path, 'r', encoding='utf-8') as f:
                affixes = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(affixes)} affixes from {affixes_path}")
            return affixes
        else:
            print(f"Warning: Affixes file not found at {affixes_path}")
            return []

    def _build_affix_finder(self, affixes: List[str]) -> ahocorasick.Automaton:
        """
        Build Aho-Corasick automaton for efficient affix detection.

        Args:
            affixes: List of affix strings

        Returns:
            Aho-Corasick automaton
        """
        affix_finder = ahocorasick.Automaton()

        for affix in affixes:
            affix_finder.add_word(affix, affix)

        affix_finder.make_automaton()
        return affix_finder

    def _get_affix_token_ids(self) -> set:
        """Convert affixes to their token IDs (only single-token affixes)."""
        affix_ids = set()
        for affix in self.affixes:
            try:
                token_ids = self.tokenizer.encode(affix)
                if len(token_ids) == 1:
                    affix_ids.add(token_ids[0])
            except:
                pass
        return affix_ids

    def find_affixes(self, s: str) -> List[Tuple[int, str]]:
        """
        Find all affixes in string s.

        Args:
            s: String to search

        Returns:
            List of (start_index, affix) tuples
        """
        matches = []
        for end_index, aff in self.affix_finder.iter(s):
            start_index = end_index - len(aff) + 1
            matches.append((start_index, aff))
        return matches

    def _build_expansions(self) -> dict:
        """
        Build expansions dictionary from tokenizer vocabulary.

        Returns:
            Dict mapping token_id -> list of (token_id1, token_id2) tuples
        """
        print("Building token expansions...")

        tokenizer_vocab = self.tokenizer._mergeable_ranks
        tokens_as_tuples = [tuple(token) for token in tokenizer_vocab.keys()]

        # Build merges first
        merges = {}
        for i, (token_bytes, token_id) in enumerate(tqdm(
            tokenizer_vocab.items(),
            desc="Building merges",
            disable=False
        )):
            if i < 256:  # Skip byte tokens
                continue

            # Find all valid splits
            for j in range(1, len(token_bytes)):
                first_part = token_bytes[:j]
                second_part = token_bytes[j:]

                if (tuple(first_part) in tokens_as_tuples and
                    tuple(second_part) in tokens_as_tuples):
                    first_id = tokenizer_vocab[first_part]
                    second_id = tokenizer_vocab[second_part]
                    merges[(first_id, second_id)] = token_id

        # Invert merges to get expansions
        expansions = {}
        for (id1, id2), merged_id in merges.items():
            if merged_id not in expansions:
                expansions[merged_id] = []
            expansions[merged_id].append((id1, id2))

        print(f"Built {len(expansions)} expansions from {len(merges)} merges")
        return expansions

    def contract_randomly(
        self,
        token_ids: List[int],
        num_tokens: List[int] = [2, 3],
        contract_prob: List[float] = [0.5, 0.5]
    ) -> Tuple[str, int, int]:
        """
        Randomly contract 2-3 adjacent tokens, avoiding affixes.

        Uses affix awareness: if contracted token contains affix,
        may skip based on probability.

        Args:
            token_ids: List of token IDs
            num_tokens: Possible numbers of tokens to contract
            contract_prob: Probability weights for num_tokens choices

        Returns:
            (contracted_string, start_idx, end_idx)
        """
        # Choose number of tokens to contract
        n = random.choices(num_tokens, weights=contract_prob)[0]

        # Keep trying until we find a valid contraction
        max_attempts = 100
        for _ in range(max_attempts):
            # Pick random starting index
            start_idx = random.randint(0, len(token_ids) - n)

            # Get tokens to contract
            for_contraction = token_ids[start_idx:start_idx + n]

            # Check if any token is already an affix
            affix_in_tokens = any(tok_id in self.affix_ids for tok_id in for_contraction)

            # Contract the tokens
            contracted = self.tokenizer.decode(for_contraction)

            # Find all affixes in contracted token
            affix_matches = self.find_affixes(contracted)
            affix_indices = [idx for idx, _ in affix_matches]

            # Check for overlapping affixes (multiple affixes at same position)
            has_overlap = len(affix_indices) != len(set(affix_indices))

            # Determine affix awareness threshold
            awareness = (self.affix_awareness_if_overlap if has_overlap
                        else self.affix_awareness)

            # Random test for affix awareness
            if not affix_in_tokens or random.random() >= awareness:
                return contracted, start_idx, start_idx + n

        # If we couldn't find valid contraction, just return first n tokens
        start_idx = 0
        contracted = self.tokenizer.decode(token_ids[:n])
        return contracted, start_idx, n

    def affix_aware_expand(self, token: str) -> List[str]:
        """
        Expand token by splitting off known affixes.

        If multiple affixes found, randomly chooses one.

        Args:
            token: String to expand

        Returns:
            List of token pieces (may contain affix and remainder)
        """
        affix_matches = self.find_affixes(token)

        if len(affix_matches) == 0:
            return [token]

        # Randomly choose an affix if multiple found
        idx, affix = random.choice(affix_matches)

        # Split token at affix boundary
        pieces = [
            token[:idx],
            affix,
            token[idx + len(affix):]
        ]

        # Remove empty strings
        return [p for p in pieces if p]

    def dup_aware_expand(self, token_list: List[str]) -> List[str]:
        """
        Expand tokens by splitting off syllable duplications.

        Looks for 2-letter syllable repetitions (e.g., "gaganda" → "ga" + "ganda")

        Args:
            token_list: List of token strings

        Returns:
            List with duplications split
        """
        result = []
        for token in token_list:
            split_tokens = self._split_repeating_pairs(token)
            result.extend(split_tokens)
        return result

    def _split_repeating_pairs(self, s: str) -> List[str]:
        """Split off first instance of any 2-letter repetition."""
        for i in range(len(s)):
            if i + 4 <= len(s):
                chunk = s[i:i+4]
                first_pair = chunk[:2]
                second_pair = chunk[2:4]

                if first_pair == second_pair:
                    tokens = []
                    if i > 0:
                        tokens.append(s[:i])
                    tokens.append(first_pair)
                    # Remove first instance of repetition
                    tokens.append(s[i:].replace(first_pair, '', 1))
                    return [t for t in tokens if t]

        return [s]

    def tokenizer_expand(self, token_list: List[str]) -> List[int]:
        """
        Re-tokenize list of strings using base tokenizer.

        Args:
            token_list: List of strings

        Returns:
            Flattened list of token IDs
        """
        token_ids = [self.tokenizer.encode(s) for s in token_list]
        # Flatten
        return [tid for group in token_ids for tid in group]

    def stochastok_expand_nonaffs(
        self,
        token_ids: List[int],
        expand_prop: Optional[float] = None,
        max_num_to_expand: Optional[int] = None,
        disable_tqdm: bool = True
    ) -> List[int]:
        """
        Stochastically expand non-affix tokens.

        Args:
            token_ids: List of token IDs
            expand_prop: Proportion of tokens to expand
            max_num_to_expand: Maximum number to expand
            disable_tqdm: Disable progress bar

        Returns:
            List of token IDs after expansion
        """
        if expand_prop is None:
            expand_prop = self.expand_prop

        num_to_expand = int(len(token_ids) * expand_prop)
        num_expanded = 0

        for _ in tqdm(range(num_to_expand), disable=disable_tqdm, desc="Expanding"):
            if max_num_to_expand and num_expanded >= max_num_to_expand:
                break

            if len(token_ids) == 0:
                break

            idx = np.random.randint(len(token_ids))
            token_id = token_ids[idx]

            # Skip affixes
            while token_id in self.affix_ids and len(token_ids) > 0:
                idx = np.random.randint(len(token_ids))
                token_id = token_ids[idx]

            # Expand if possible
            if token_id in self.expansions:
                chosen_expansion = random.choice(self.expansions[token_id])
                token_ids = (token_ids[:idx] +
                           list(chosen_expansion) +
                           token_ids[idx+1:])
                num_expanded += 1

        return token_ids

    def contract_expand(
        self,
        token_ids: List[int],
        contract_prop: Optional[float] = None,
        expand_prop: Optional[float] = None,
        tok_to_contract: List[int] = [2, 3],
        contract_prob: List[float] = [0.5, 0.5],
        disable_tqdm: bool = False
    ) -> List[int]:
        """
        Main Patok pipeline: contract-expand with morphological awareness.

        Process:
        1. Contract random tokens (avoiding affixes)
        2. Re-expand contracted token by:
           a. Splitting off known affixes
           b. Splitting off syllable duplications
           c. Re-tokenizing with base tokenizer
        3. Stochastically expand remaining non-affix tokens

        Args:
            token_ids: List of token IDs
            contract_prop: Proportion of tokens to contract
            expand_prop: Proportion to expand in final step
            tok_to_contract: Options for number of tokens to contract
            contract_prob: Probability weights for tok_to_contract
            disable_tqdm: Disable progress bar

        Returns:
            List of token IDs after processing
        """
        if contract_prop is None:
            contract_prop = self.contract_prop
        if expand_prop is None:
            expand_prop = self.expand_prop

        num_to_contract = int(len(token_ids) * contract_prop)

        for _ in tqdm(range(num_to_contract),
                     desc='Contracting and expanding w/ morphology-awareness',
                     disable=disable_tqdm):

            # Contract random tokens
            contracted, start_idx, end_idx = self.contract_randomly(
                token_ids, tok_to_contract, contract_prob
            )

            # Affix-aware expansion
            aff_aware = self.affix_aware_expand(contracted)

            # Duplication-aware expansion
            dup_aware = self.dup_aware_expand(aff_aware)

            # Re-tokenize
            new_token_ids = self.tokenizer_expand(dup_aware)

            # Replace in original sequence
            token_ids[start_idx:end_idx] = new_token_ids

        # Final stochastic expansion of non-affixes
        token_ids = self.stochastok_expand_nonaffs(
            token_ids, expand_prop, disable_tqdm=disable_tqdm
        )

        return token_ids

    def process(
        self,
        text: str,
        contract_prop: Optional[float] = None,
        expand_prop: Optional[float] = None,
        disable_tqdm: bool = True
    ) -> List[int]:
        """
        Tokenize text with morphology-aware Patok processing.

        Args:
            text: Input text
            contract_prop: Proportion of tokens to contract
            expand_prop: Proportion to expand
            disable_tqdm: Disable progress bar

        Returns:
            List of token IDs
        """
        # Initial tokenization
        token_ids = self.tokenizer.encode(text)

        # Apply Patok processing
        token_ids = self.contract_expand(
            token_ids,
            contract_prop=contract_prop,
            expand_prop=expand_prop,
            disable_tqdm=disable_tqdm
        )

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids)

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to list of token strings."""
        return [
            self.tokenizer.decode_single_token_bytes(tid).decode("utf-8", "replace")
            for tid in token_ids
        ]
