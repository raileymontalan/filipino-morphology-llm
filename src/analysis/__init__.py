"""
Analysis tools for morphological and information-theoretic evaluation.

Structure:
- morphological_metrics.py: Core morphological metrics (MorphScore, Boundary F1, etc.)
- information_theory.py: Information-theoretic analysis
- tokenization/: Tokenization comparison tools
- affixes/: Affix coverage analysis
- datasets/: Dataset comparison tools
"""

from .information_theory import (
    InformationTheoreticAnalysis,
    MorphemeTokenAlignment,
    generate_information_theoretic_report,
)
from .morphological_metrics import (
    MorphologicalAnnotation,
    MorphologicalMetrics,
    compare_tokenizers_morphologically,
    generate_morphological_report,
)

__all__ = [
    # Morphological metrics
    "MorphologicalMetrics",
    "MorphologicalAnnotation",
    "compare_tokenizers_morphologically",
    "generate_morphological_report",
    # Information-theoretic analysis
    "InformationTheoreticAnalysis",
    "MorphemeTokenAlignment",
    "generate_information_theoretic_report",
]
