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
import random
import os


def load_pacute(split="test", categories=None, format="mcq", **kwargs):
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
        categories = ['affixation', 'composition', 'manipulation', 'syllabification']

    tasks = []
    category_counts = {}

    data_file_suffix = "gen" if format == "gen" else "mcq"

    for category in categories:
        data_file = os.path.join(project_root, f"data/benchmarks/{category}_{data_file_suffix}.jsonl")

        if not os.path.exists(data_file):
            print(f"Warning: PACUTE file not found: {data_file}")
            continue

        count = 0
        with open(data_file) as f:
            for line in f:
                task = json.loads(line)
                task['_category'] = category  # Track source category
                tasks.append(task)
                count += 1

        category_counts[category] = count

    total = len(tasks)
    print(f"PACUTE: Loaded {total} tasks ({format}) across {len(categories)} categories:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} tasks")

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        prompt_data = task["prompts"][0]
        prefix = prompt_data["text_en"]
        sample_id = task.get("id", f"pacute_{format}_{i:05d}")

        if format == "gen":
            # Gen files use a top-level "label" field; no MCQ options.
            ground_truth = task["label"]
            false_options = []
        else:
            mcq_options = prompt_data["mcq_options"]
            ground_truth = mcq_options["correct"]
            false_options = [
                v for k, v in sorted(mcq_options.items())
                if k.startswith("incorrect")
            ]

        yield prefix, ground_truth, false_options, sample_id, task.get("subcategory")
