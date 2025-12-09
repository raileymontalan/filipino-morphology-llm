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
import json
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
        prefix_file: Optional[str] = None,
        infix_file: Optional[str] = None,
        suffix_file: Optional[str] = None,
        num_toks_to_cont: List[int] = [2, 3, 4],
        contract_prob: List[float] = [0.35, 0.35, 0.3],
        affix_awareness: float = 0.95,
        affix_awareness_if_overlap: float = 0.75,
        save_expansions_to: Optional[str] = None,
        expansions_file: Optional[str] = None,
        expand_prop: float = 0.1,
        contract_prop: float = 0.9,
    ):
        """
        Initialize morphology-aware Patok processor.

        Args:
            tokenizer: Tokenizer with encode/decode methods
            affixes: List of Filipino affixes. If None, loads from default file
            prefix_file: path to file containing one prefix per line
            infix_file: path to file containing one infix per line
            suffix_file: path to file containing one suffix per line
            num_toks_to_cont (list of int): list of possibilities for number of tokens to merge
            contract_prob (list of float, sum = 1): probability weights of choosing number of tokens in num_toks_to_cont
            affix_awareness: Probability of skipping contraction if token is affix
            affix_awareness_if_overlap: Affix awareness if multiple affixes at same position
            save_expansions_to: json file to which expansions will be saved
            expansions_file: path to json file containing expansions
            expand_prop: Default proportion of tokens to expand
            contract_prop: Default proportion of tokens to contract
        """
        self.tokenizer = tokenizer
        self.prefix_file = prefix_file
        self.infix_file = infix_file
        self.suffix_file = suffix_file
        self.num_toks_to_cont = num_toks_to_cont
        self.contract_prob = contract_prob
        self.affix_awareness = affix_awareness
        self.affix_awareness_if_overlap = affix_awareness_if_overlap
        self.expand_prop = expand_prop
        self.contract_prop = contract_prop
        self.save_expansions_to = save_expansions_to
        self.expansions_file = expansions_file

        # Load affixes and build automaton
        self.affixes = self._build_affix_list()
        self.affix_finder = self._build_affix_finder(self.affixes)

        # Get affix token IDs
        self.affix_ids = self._generate_affix_ids(self.affixes)

        # Build expansion dictionary
        self.expansions = self._build_expansions()

        print(f"Initialized MorphologyAwarePatokProcessor:")
        print(f"  - {len(self.affixes)} affix versions loaded")
        print(f"  - {len(self.affix_ids)} affix token IDs")
        print(f"  - {len(self.expansions)} expandable tokens")

    def _build_affix_list(self) -> List[str]:
        """
        Given a prefix file, generate four versions of each prefix: original ('ma'), space-prepended (' ma'),
        capitalized ('Ma'), space-prepeneded and capitalized (' Ma'). Then, generate two versions of each
        suffix: original ('an') and space-appended ('an '). Combine all expanded prefixes + infixes +
        expanded suffixes into one list.
        Args:
            prefix_file (string): path to prefix file if any
            infix_file (string): path to infix file if any
            suffix_file (string): path to suffix file if any
        Returns:
            list: containing unique prefixes (and each of the prefix's four versions), infixes, and suffixes
        """

        def read_affix_file(path):
            """
            Read a text file where each line is one affix.
            Args:
                path (string): path to file containing affixes
            Returns:
                list: list of affixes from file
            """
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        # read affix files into lists
        prefix = read_affix_file(self.prefix_file)
        infix = read_affix_file(self.infix_file)
        suffix = read_affix_file(self.suffix_file)

        # Combine all affixes
        all_affixes = prefix + infix + suffix

        # initialize empty list for the prefix versions
        expanded_prefixes = []

        # iterate through each prefix
        for p in prefix:
            # generate each version of the prefix and add to expanded_prefixes
            expanded_prefixes.extend([
                p,            # original
                " " + p,      # space-prepended
                p.capitalize(),      # capitalized
                " " + p.capitalize() # space + capital
            ])

        # initialize empty list for the suffix versions
        expanded_suffixes = []

        # iterate through each suffix
        for s in suffix:
            # generate each version of the suffix and add to expanded_suffixes
            expanded_suffixes.extend([
                s,            # original
                s + " ",      # space-appended
            ])

        # Combine everything and remove duplicates
        all_affix_versions = expanded_prefixes + infix + expanded_suffixes
        all_affix_versions = list(set(all_affix_versions))

        print(f"Loaded {len(all_affixes)} affixes")
        print(f"Converted affixes to {len(all_affix_versions)} space-prepended and capitalized versions")

        return all_affix_versions

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

    def _generate_affix_ids(self, affixes:List[str]) -> List[int]:
        """
        Get token id's of affixes that are in the tokenizer's vocabulary.
        Args:
            affixes (list): list of affixes
            tokenizer
        Returns:
            list: list of token_ids corresponding to affixes that are in vocabulary
        """

        # get tiktoken affix_ids by checking if affix is in _mergeable_ranks
        if hasattr(self.tokenizer, "_mergeable_ranks"):
            affix_ids = [self.tokenizer._mergeable_ranks.get(aff.encode('utf-8')) for aff in affixes
                         if aff.encode('utf-8') in self.tokenizer._mergeable_ranks]

        # get HF Autotokenizer affix_ids by checking if encoding the affix results in only
        # one token_id; if it does not, then affix is not in vocabulary
        else:
            affix_ids = [self.tokenizer.encode(aff,add_special_tokens=False)[0] for aff in affixes
                         if len(self.tokenizer.encode(aff,add_special_tokens=False))==1]

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
        0. Inputs
            tokenizer whose vocab will be crawled to build expansions
            save_expansions_to (str): saves expansions into given json file name
            expansions_file (string): path to json file containing expansions
        1. Builds `merges` (dict):
            Keys: tuples of token_ids.
            Values: token_id of the result of merging the two tokens.
            eg. merges = {(token_id1, token_id2): token_id}
        2. Builds `expansions` (dict):
            Keys: token_id.
            Values: list of tuples of token_ids that the key token_id can be split into.
            eg. expansions = {token_id: [(token_id1_1, token_id1_2), (token_id2_1, token_id2_2), ...]}
        """

        #0. Check if user provides expansions_file, then load
        if isinstance(self.expansions_file, str):
            with open(self.expansions_file , "r") as f:
                loaded_expansions = json.load(f)

            # convert keys to integers
            loaded_expansions = {int(k): v for k, v in loaded_expansions.items()}

            return loaded_expansions


        # 1. Build merges dict
        # get tiktoken vocab
        if hasattr(self.tokenizer, "_mergeable_ranks"):
            ttokenizer_byt2int = self.tokenizer._mergeable_ranks
        # get HF Autotokenizer vocab
        else:
            ttokenizer_byt2int = self.tokenizer.get_vocab()

        ttokenizer_tokens_as_tuples = [tuple(token) for token in list(ttokenizer_byt2int.keys())] # Needed to be able to use `in` operator
        merges = {}
        for i, (token_as_bytes, token_id) in tqdm(
            enumerate(ttokenizer_byt2int.items()),
            total=len(ttokenizer_byt2int),
            desc="Building tokenizer's merges",
            ):
            # print(f"{i=}: {token_as_bytes.decode('utf-8')}")
            num_merges = 0
            ## split the token at each possible point and find a split/"merge" where both parts are already present earlier in the vocab.
            for j in range(1, len(token_as_bytes)):
                first_part = token_as_bytes[:j]
                second_part = token_as_bytes[j:]
                # print(f"{j=}: {first_part.decode('utf-8')} + {second_part.decode('utf-8')}")
                if tuple(first_part) in ttokenizer_tokens_as_tuples and tuple(second_part) in ttokenizer_tokens_as_tuples:
                    first_part_id = ttokenizer_byt2int[first_part]
                    second_part_id = ttokenizer_byt2int[second_part]
                    merges[(first_part_id, second_part_id)] = token_id
                    num_merges += 1
                    # print(f"{first_part}: {first_part_id} + {second_part}: {second_part_id} -> {token_as_bytes}: {token_id}")
            #assert num_merges >= 1, f"No merge found for {token_as_bytes=}: {list(token_as_bytes)=}"

        # 2. Build expansions dict
        expansions = {}
        for k, v in merges.items():
            if v in expansions:
                expansions[v].append(k)
            else:
                expansions[v] = [k]


        # 3. Save expansions if file name by user
        if isinstance(self.save_expansions_to, str):
            with open(self.save_expansions_to, "w") as f:
                json.dump(expansions, f, indent=2)

        return expansions

    def contract_randomly(
        self,
        token_ids: List[int],
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
        n = random.choices(self.num_toks_to_cont, weights=self.contract_prob)[0]

        # If token_ids is too short, return the entire string
        if len(token_ids) < n:
            return self.tokenizer.decode(token_ids), 0, len(token_ids)

        # Keep trying until we find a valid contraction
        max_attempts = 100
        for _ in range(max_attempts):
            # Pick random starting index
            start_idx = random.randint(0, len(token_ids) - n)

            # Get tokens to contract
            for_contraction = token_ids[start_idx:start_idx + n]

            # if tokenizer has bos and eos token ids
            if hasattr(self.tokenizer, "bos_token_id"):
                special_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
                # check if chosen tokens have bos or eos
                # keep choosing new tokens if current selection has bos or eos
                for _ in range(max_attempts):
                    if not any(tid in special_ids for tid in for_contraction):
                        break

                    start_idx = random.randint(0, len(token_ids) - n)
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

        # if there are no affixes, return token in a list
        # or if RNG dictates that the affix won't be split off
        if len(affix_matches) == 0 or random.random()>self.affix_awareness:
            return [token]

        # initialize list that will contain affix matches with space
        # these are more likely to be a prefix or suffix
        affixes_with_space = []

        # iterate through each affix match
        for idx,affix in affix_matches:
            # if an affix match has a space, add that match to affixes_with_spaces
            if ' ' in affix:
                affixes_with_space.append((idx,affix))

        # if there are affixes with spaces, constrain selection to these
        if len(affixes_with_space) > 0:
            affix_matches = affixes_with_space

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

        try:
            token_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in token_list]
        except:
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
        disable_tqdm: bool = True
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
            contracted, start_idx, end_idx = self.contract_randomly(token_ids)

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
        try:
            list_of_tok_strings = [
                self.tokenizer.decode_single_token_bytes(tid).decode("utf-8", "replace")
                for tid in token_ids
            ]

        except:
            list_of_tok_strings = [self.tokenizer.decode(tid) for tid in token_ids]

        return list_of_tok_strings
