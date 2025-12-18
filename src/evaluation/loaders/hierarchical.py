"""
Loader for Hierarchical benchmark - diagnostic tasks across 6 compositional levels.
"""

import json
import random
from pathlib import Path


def load_hierarchical(format="mcq", **kwargs):
    """
    Load hierarchical benchmark.

    Args:
        format: 'mcq' or 'gen'
        **kwargs: Additional arguments (ignored)

    Yields:
        - prefix: The question
        - ground_truth: The correct answer
        - false_options: List of incorrect options (empty for gen format)
        - sample_id: Unique identifier for the sample
    """
    benchmarks_dir = Path("data/benchmarks")
    filepath = benchmarks_dir / f"hierarchical_{format}.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Hierarchical benchmark file not found: {filepath}\n"
            "Generate it with: python src/evaluation/datasets/scripts/generate_hierarchical_benchmark.py"
        )

    tasks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line.strip())
            tasks.append(task)

    print(f"Loaded {len(tasks)} hierarchical tasks ({format} format)")

    # Shuffle indices
    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        # Hierarchical benchmark uses 'prompt_tl' for Tagalog prompts
        prefix = task.get("prompt_tl", task.get("question", ""))
        sample_id = task.get("id", f"hierarchical_{format}_{i:05d}")

        if format == "mcq":
            options = task["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:  # gen
            ground_truth = task["answer"]
            false_options = []

        yield prefix, ground_truth, false_options, sample_id
