"""
CUTE: Character Understanding Test Evaluation.

Tests character-level understanding through orthographic manipulation tasks.
Based on Edman et al. (2024) "CUTE: Measuring LLMs' Understanding of Their Tokens"

14 task types:
- Character-level: spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char
- Word-level: contains_word, ins_word, del_word, swap_word, sub_word
- Semantic/Orthographic: orth, sem

Total: 14,000 examples (1,000 per task)

Dataset: https://huggingface.co/datasets/leukas/cute
"""

import random


def load_cute(split="test", task_types=None, max_per_task=100, **kwargs):
    """
    Load CUTE benchmark from local JSONL file.

    Args:
        split: Not used (all data treated as test)
        task_types: List of task types to include. Options:
                   ['spell', 'spell_inverse', 'contains_char', 'contains_word',
                    'orth', 'sem', 'ins_char', 'ins_word', 'del_char', 'del_word',
                    'sub_char', 'sub_word', 'swap_char', 'swap_word']
                   If None, loads all task types.
        max_per_task: Maximum number of examples per task type (default: 100)

    Yields:
        For MCQ format compatibility:
        - prefix: The prompt (question)
        - ground_truth: The correct answer
        - false_options: Empty list (generative task, not MCQ)

    Note: CUTE is a generative benchmark, not MCQ. The false_options will be empty.
    """
    import json
    from pathlib import Path

    # Load from local JSONL file
    jsonl_path = Path("data/benchmarks/cute_gen.jsonl")

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"CUTE benchmark file not found: {jsonl_path}\n"
            "Generate it with: python src/evaluation/datasets/scripts/generate_cute_benchmark.py"
        )

    # Available task types (derived from task naming in prompts)
    all_task_types = [
        "spell",
        "spell_inverse",
        "contains_char",
        "contains_word",
        "orth",
        "sem",
        "ins_char",
        "ins_word",
        "del_char",
        "del_word",
        "sub_char",
        "sub_word",
        "swap_char",
        "swap_char",
    ]

    if task_types is None:
        task_types = all_task_types

    # Load all samples from JSONL
    all_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            all_samples.append(sample)

    # Filter by task type if specified
    tasks = []
    task_counts = {}
    for sample in all_samples:
        task_type = sample.get("task_type", "unknown")
        if task_types is None or task_type in task_types:
            if task_type not in task_counts:
                task_counts[task_type] = 0
            task_counts[task_type] += 1
            tasks.append(sample)

    total = len(tasks)
    num_tasks = len(task_counts)
    print(f"CUTE (GEN): Loaded {total} character understanding tasks from local file ({num_tasks} task types).")
    print("Note: CUTE is a generative benchmark (prompt → answer), not MCQ format.")

    # Shuffle tasks
    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        prefix = task["question"]
        ground_truth = task["answer"]
        false_options = []  # Generative task, no MCQ options
        sample_id = task.get("id", f"cute_gen_{i:05d}")

        yield prefix, ground_truth, false_options, sample_id
