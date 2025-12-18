"""
PACUTE: Philippine Annotated Corpus for Understanding Tagalog Entities

Full morphological understanding benchmark covering:
- Affixation (280 MCQ items)
- Composition (280 MCQ items)
- Manipulation (320 MCQ items)
- Syllabification (160 MCQ items)
Total: 1,040 MCQ tasks
"""

import json
import os
import random


def load_pacute(split="test", categories=None, **kwargs):
    """
    Load PACUTE benchmark.

    Args:
        split: Not used (all data treated as test)
        categories: List of categories to include. Options:
                   ['affixation', 'composition', 'manipulation', 'syllabification']
                   If None, loads all categories.
    """
    # Find project root by looking for data/benchmarks directory
    current_file_path = os.path.abspath(__file__)
    search_dir = current_file_path

    for _ in range(10):  # Search up to 10 levels
        search_dir = os.path.dirname(search_dir)
        if os.path.exists(os.path.join(search_dir, "data/benchmarks")):
            project_root = search_dir
            break
    else:
        raise FileNotFoundError("Could not find data/benchmarks directory")

    # Default to all categories
    if categories is None:
        categories = ["affixation", "composition", "manipulation", "syllabification"]

    tasks = []
    category_counts = {}

    for category in categories:
        mcq_file = os.path.join(project_root, f"data/benchmarks/{category}_mcq.jsonl")

        if not os.path.exists(mcq_file):
            print(f"Warning: PACUTE file not found: {mcq_file}")
            continue

        count = 0
        with open(mcq_file) as f:
            for line in f:
                task = json.loads(line)
                task["_category"] = category  # Track source category
                tasks.append(task)
                count += 1

        category_counts[category] = count

    total = len(tasks)
    print(f"PACUTE: Loaded {total} tasks across {len(categories)} categories:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} tasks")

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        # Get the English prompt
        prompt_data = task["prompts"][0]
        prefix = prompt_data["text_en"]
        sample_id = task.get("id", f"pacute_mcq_{i:05d}")

        # Extract options
        mcq_options = prompt_data["mcq_options"]
        ground_truth = mcq_options["correct"]
        false_options = [
            mcq_options["incorrect1"],
            mcq_options["incorrect2"],
            mcq_options["incorrect3"],
        ]

        yield prefix, ground_truth, false_options, sample_id
