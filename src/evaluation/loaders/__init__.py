"""Benchmark loaders for loading evaluation datasets."""

from .registry import EVALS_DICT, load_benchmark

__all__ = [
    "load_pacute",
    "load_benchmark",
    "EVALS_DICT",
]
