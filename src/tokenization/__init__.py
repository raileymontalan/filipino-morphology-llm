"""
Tokenization processors for morphologically-aware tokenization.

Main classes:
- MorphologyAwarePatokProcessor: Affix-aware tokenization with Aho-Corasick (RECOMMENDED)
- StochastokProcessor: Stochastic token expansion

Deprecated:
- PatokProcessor: Use MorphologyAwarePatokProcessor instead
"""

from .patok_morphology import MorphologyAwarePatokProcessor
from .stochastok_processor import StochastokProcessor
from .affix_decomposition import AffixDecomposer, AffixDecomposition, compare_tokenizers

# Deprecated - import triggers warning
from .patok_processor import PatokProcessor

__all__ = [
    # Recommended
    "MorphologyAwarePatokProcessor",
    "StochastokProcessor",
    # Analysis utilities
    "AffixDecomposer",
    "AffixDecomposition",
    "compare_tokenizers",
    # Deprecated
    "PatokProcessor",
]
