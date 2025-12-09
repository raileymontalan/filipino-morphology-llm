"""
Load a benchmark loader, given the benchmark name.
"""
from functools import partial

from evaluation.loaders.arc import load_arc
from evaluation.loaders.blimp import load_blimp
from evaluation.loaders.hellaswag import load_hellaswag
from evaluation.loaders.mmlu import load_mmlu
from evaluation.loaders.winogrande import load_winogrande
from evaluation.loaders.langgame import load_langgame
from evaluation.loaders.cute import load_cute
from evaluation.loaders.pacute import load_pacute
from evaluation.loaders.multi_digit_addition import load_multi_digit_addition
from evaluation.loaders.hierarchical import load_hierarchical

EVALS_DICT = {
    # Standard benchmarks
    "arc": partial(load_arc, split="test"),
    "winograd": partial(load_winogrande, split="test"),
    "mmlu": partial(load_mmlu, split="test"),
    "hellaswag": partial(load_hellaswag, split="test"),
    "blimp": partial(load_blimp, split="test"),
    
    # LangGame - MCQ (default) and GEN variants
    "langgame": partial(load_langgame, format="mcq"),
    "langgame-mcq": partial(load_langgame, format="mcq"),
    "langgame-gen": partial(load_langgame, format="gen"),
    
    # CUTE - GEN only (1400 samples: 100 per task × 14 tasks)
    "cute": partial(load_cute, split="test", max_per_task=100),
    "cute-gen": partial(load_cute, split="test", max_per_task=100),
    
    # Hierarchical - MCQ (default) and GEN variants
    "hierarchical": partial(load_hierarchical, format="mcq"),
    "hierarchical-mcq": partial(load_hierarchical, format="mcq"),
    "hierarchical-gen": partial(load_hierarchical, format="gen"),
    
    # Multi-digit Addition - GEN (default) and MCQ variants
    "multi-digit-addition": partial(load_multi_digit_addition, format="gen", max_samples=1000),
    "multi-digit-addition-gen": partial(load_multi_digit_addition, format="gen", max_samples=1000),
    "multi-digit-addition-mcq": partial(load_multi_digit_addition, format="mcq", max_samples=1000),
    
    # PACUTE - MCQ (default) and GEN variants for all categories
    "pacute": partial(load_pacute, split="test"),
    "pacute-mcq": partial(load_pacute, split="test"),
    "pacute-gen": partial(load_pacute, split="test", format="gen"),
    "pacute-affixation": partial(load_pacute, split="test", categories=["affixation"]),
    "pacute-affixation-mcq": partial(load_pacute, split="test", categories=["affixation"]),
    "pacute-affixation-gen": partial(load_pacute, split="test", categories=["affixation"], format="gen"),
    "pacute-composition": partial(load_pacute, split="test", categories=["composition"]),
    "pacute-composition-mcq": partial(load_pacute, split="test", categories=["composition"]),
    "pacute-composition-gen": partial(load_pacute, split="test", categories=["composition"], format="gen"),
    "pacute-manipulation": partial(load_pacute, split="test", categories=["manipulation"]),
    "pacute-manipulation-mcq": partial(load_pacute, split="test", categories=["manipulation"]),
    "pacute-manipulation-gen": partial(load_pacute, split="test", categories=["manipulation"], format="gen"),
    "pacute-syllabification": partial(load_pacute, split="test", categories=["syllabification"]),
    "pacute-syllabification-mcq": partial(load_pacute, split="test", categories=["syllabification"]),
    "pacute-syllabification-gen": partial(load_pacute, split="test", categories=["syllabification"], format="gen"),
}


def load_benchmark(benchmark_name):
    """
    Given the benchmark name, build the benchmark
    """
    return EVALS_DICT[benchmark_name]()
