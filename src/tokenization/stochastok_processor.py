"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import os
import json
from tqdm import tqdm
import random
import numpy as np

from .base_processor import TokenizerProcessor


class StochastokProcessor(TokenizerProcessor):
    """
    A processor that applies stochastok expansion to the tokenized data.
    Uses the shared expansion building logic from base processor.
    """
    def __init__(self, tokenizer, expand_prop=None):
        super().__init__(tokenizer)
        self.expand_prop = expand_prop
        # Use inherited set_expansions() from base processor
        self.set_expansions()

    def expand(self, token_ids, expand_prop=0.1, max_num_to_expand=None, disable_tqdm=True):
        """
        Expand the sequence of tokens by splitting tokens.
        Args:
            token_ids (list): list of token_ids to expand.
            expand_prop (float): proportion of tokens to (try) expanding.
        Returns:
            list: list of token_ids after expanding.
        """
        expand_prop = expand_prop if expand_prop is not None else self.expand_prop
        num_to_expand = int(len(token_ids) * expand_prop)
        num_expanded = 0
        for _ in tqdm(range(num_to_expand), disable=disable_tqdm):
            if max_num_to_expand is not None and num_expanded >= max_num_to_expand:
                break
            idx = np.random.randint(len(token_ids))
            token_id = token_ids[idx]
            if token_id in self.expansions:
                possible_expansions = self.expansions[token_id]
                chosen_expansion = random.choice(possible_expansions)
                token_ids = token_ids[:idx] + list(chosen_expansion) + token_ids[idx+1:]
                num_expanded += 1
        # print(f"Attempted to expand {num_to_expand}, expanded {num_expanded}")
        # print(f"Original length {len(token_ids)-num_expanded}, new length {len(token_ids)}")
        # print(f"Attempted to expand by {expand_prop*100}%, expanded by {num_expanded/len(token_ids)*100:.2f}%")
        return token_ids
