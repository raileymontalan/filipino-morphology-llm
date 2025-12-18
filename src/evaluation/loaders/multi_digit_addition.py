"""
Multi-digit addition benchmark

Tests arithmetic capabilities through addition problems.
Format: "X+Y=" → "Z"
"""

import json
import os
import random


def load_multi_digit_addition(format="gen", max_samples=1000, **kwargs):
    """
    Load multi-digit addition benchmark from JSONL file.

    Args:
        format: "gen" or "mcq" (default: "gen")
        max_samples: Maximum number of examples to load (default: 1000)

    Yields:
        - prefix: The question (e.g., "840+425=")
        - ground_truth: The correct answer (e.g., "1265")
        - false_options: List of incorrect options (empty for gen format)
    """
    current_file_path = os.path.abspath(__file__)
    dir_of_file = os.path.dirname(current_file_path)
    jsonl_path = os.path.join(dir_of_file, f"../../../data/benchmarks/multi_digit_addition_{format}.jsonl")

    # Load all samples from JSONL
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples is not None and len(samples) >= max_samples:
                break

    print(f"Multi-digit Addition ({format.upper()}): Loaded {len(samples)} examples.")

    # Shuffle samples
    indices = list(range(len(samples)))
    random.shuffle(indices)

    for i in indices:
        sample = samples[i]
        prefix = sample["question"]
        sample_id = sample.get("id", f"multi_digit_addition_{format}_{i:05d}")

        if format == "mcq":
            options = sample["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:  # gen
            ground_truth = sample["answer"]
            false_options = []

        yield prefix, ground_truth, false_options, sample_id
