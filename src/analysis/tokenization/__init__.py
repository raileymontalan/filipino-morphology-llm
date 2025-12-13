"""
Tokenization analysis tools.

Compare tokenization approaches on Filipino morphology and measure
alignment between token and morpheme boundaries.
"""

from .simple_analysis import (
    load_annotations as simple_load_annotations,
    tokenize_word,
    find_token_boundaries as simple_find_token_boundaries,
    compute_morph_score,
    compute_fragmentation,
)
from .comprehensive_analysis import (
    load_annotations as comprehensive_load_annotations,
    tokenize_with_gpt2,
    tokenize_with_patok,
    find_token_boundaries as comprehensive_find_token_boundaries,
    create_morphological_annotation,
    analyze_tokenization,
    compare_tokenizers as comprehensive_compare_tokenizers,
)
from .compare_tokenizers import (
    load_annotations,
    tokenize_patok_style,
    find_token_boundaries,
    compute_boundary_f1,
    analyze_tokenizer,
    print_comparison,
    print_examples,
)

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
