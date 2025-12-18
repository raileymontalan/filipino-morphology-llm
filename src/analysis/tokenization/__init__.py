"""
Tokenization analysis tools.

Compare tokenization approaches on Filipino morphology and measure
alignment between token and morpheme boundaries.
"""

from .compare_tokenizers import (
    analyze_tokenizer,
    compute_boundary_f1,
    find_token_boundaries,
    load_annotations,
    print_comparison,
    print_examples,
    tokenize_patok_style,
)
from .comprehensive_analysis import analyze_tokenization
from .comprehensive_analysis import (
    compare_tokenizers as comprehensive_compare_tokenizers,
)
from .comprehensive_analysis import create_morphological_annotation
from .comprehensive_analysis import (
    find_token_boundaries as comprehensive_find_token_boundaries,
)
from .comprehensive_analysis import load_annotations as comprehensive_load_annotations
from .comprehensive_analysis import tokenize_with_gpt2, tokenize_with_patok
from .simple_analysis import compute_fragmentation, compute_morph_score
from .simple_analysis import find_token_boundaries as simple_find_token_boundaries
from .simple_analysis import load_annotations as simple_load_annotations
from .simple_analysis import tokenize_word

__all__ = [
    # Simple analysis
    "simple_load_annotations",
    "tokenize_word",
    "simple_find_token_boundaries",
    "compute_morph_score",
    "compute_fragmentation",
    # Comprehensive analysis
    "comprehensive_load_annotations",
    "tokenize_with_gpt2",
    "tokenize_with_patok",
    "comprehensive_find_token_boundaries",
    "create_morphological_annotation",
    "analyze_tokenization",
    "comprehensive_compare_tokenizers",
    # Compare tokenizers
    "load_annotations",
    "tokenize_patok_style",
    "find_token_boundaries",
    "compute_boundary_f1",
    "analyze_tokenizer",
    "print_comparison",
    "print_examples",
]
