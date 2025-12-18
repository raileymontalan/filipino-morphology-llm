"""
Generate CUTE Benchmark.

Downloads CUTE dataset from HuggingFace and saves locally with subsampling.

CUTE: Character Understanding Test Evaluation
Tests character-level understanding through orthographic manipulation tasks.
Based on Edman et al. (2024) "CUTE: Measuring LLMs' Understanding of Their Tokens"

14 task types (100 samples each = 1,400 total):
- Character-level: spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char
- Word-level: contains_word, ins_word, del_word, swap_word, sub_word
- Semantic/Orthographic: orth, sem

Dataset: https://huggingface.co/datasets/leukas/cute
"""

import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)


def main():
    print("=" * 70)
    print("Generating CUTE Benchmark")
    print("=" * 70)
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ Error: 'datasets' library not available")
        print("Install with: pip install datasets")
        return 1

    benchmarks_dir = Path("data/benchmarks")
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    output_file = benchmarks_dir / "cute_gen.jsonl"

    print("Loading CUTE dataset from HuggingFace...")
    dataset = load_dataset("leukas/cute")

    # All task types
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
        "swap_word",
    ]

    samples = []
    task_counts = {}

    for task_type in all_task_types:
        if task_type in dataset:
            task_items = list(dataset[task_type])
            # Subsample to 100 per task
            random.shuffle(task_items)
            task_items = task_items[:100]

            for item in task_items:
                gen_sample = {
                    "question": item["prompt"],
                    "answer": item["answer"],
                    "task_type": task_type,
                }
                samples.append(gen_sample)

            task_counts[task_type] = len(task_items)
            print(f"  ✓ {task_type}: {len(task_items)} samples")

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print()
    print(f"✓ Created {output_file} with {len(samples)} samples")
    print(f"  Total task types: {len(task_counts)}")
    print("  Samples per task: 100 (subsampled)")
    print()

    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Output: {output_file}")
    print(f"Total samples: {len(samples)}")
    print("Format: GEN (question → answer)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
