#!/usr/bin/env python3
"""
Demo script for hierarchical task framework.

Shows how to:
1. Generate hierarchical tasks
2. Evaluate a model on hierarchical tasks
3. Analyze results to identify capability bottlenecks
4. Compare multiple models
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

from evaluation.datasets.generators.hierarchical import HierarchicalTaskGenerator
from evaluation.evaluators.hierarchical import (
    HierarchicalAnalyzer,
    compare_multiple_models,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def demo_generate_tasks():
    """Demo: Generate hierarchical tasks."""
    print("=" * 60)
    print("DEMO 1: Generating Hierarchical Tasks")
    print("=" * 60)
    print()

    # Load word data
    data_dir = Path("data/corpora/pacute_data")
    syllables_file = data_dir / "syllables.jsonl"

    if not syllables_file.exists():
        print(f"Error: {syllables_file} not found")
        print("Please ensure PACUTE data is in data/corpora/pacute_data/")
        return

    words_df = pd.read_json(syllables_file, lines=True)
    print(f"Loaded {len(words_df)} words from {syllables_file}")
    print()

    # Create sample affixes dataframe for demo
    # In practice, this would come from your affix annotations
    affixes_data = [
        {"word": "kumain", "root": "kain", "infix": "um"},
        {"word": "nagluto", "root": "luto", "prefix": "nag"},
        {"word": "kinain", "root": "kain", "infix": "in"},
        {"word": "lutuan", "root": "luto", "suffix": "an"},
    ]
    affixes_df = pd.DataFrame(affixes_data)

    # Initialize generator
    generator = HierarchicalTaskGenerator(words_df, affixes_df)

    print("Generating tasks for each level...")
    print()

    # Generate Level 0 tasks
    level0_tasks = generator.generate_level0_character_identification(n=5, format="gen")
    print(f"Level 0 - Character Recognition ({len(level0_tasks)} tasks):")
    print(f"  Example: {level0_tasks[0].prompt_en}")
    print(f"  Answer: {level0_tasks[0].answer}")
    print()

    # Generate Level 1 tasks
    level1_tasks = generator.generate_level1_character_deletion(n=5, format="gen")
    print(f"Level 1 - Character Manipulation ({len(level1_tasks)} tasks):")
    print(f"  Example: {level1_tasks[0].prompt_en}")
    print(f"  Answer: {level1_tasks[0].answer}")
    print()

    # Generate Level 2 tasks
    level2_tasks = generator.generate_level2_syllable_counting(n=5, format="gen")
    print(f"Level 2 - Morpheme Decomposition ({len(level2_tasks)} tasks):")
    print(f"  Example: {level2_tasks[0].prompt_en}")
    print(f"  Answer: {level2_tasks[0].answer}")
    print()

    # Generate all levels
    print("Generating complete hierarchical suite...")
    all_tasks = generator.generate_all_levels(n_per_subcategory=10, format="gen")

    total_tasks = sum(len(tasks) for tasks in all_tasks.values())
    print(f"Generated {total_tasks} total tasks across {len(all_tasks)} levels")
    print()

    for level, tasks in all_tasks.items():
        if tasks:
            print(f"  Level {level}: {len(tasks)} tasks")

    # Save tasks
    output_dir = Path("data/benchmarks/hierarchical_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.save_tasks(all_tasks, str(output_dir), format="gen")
    print()
    print(f"Saved tasks to {output_dir}/")
    print()


def demo_analyze_results():
    """Demo: Analyze hierarchical results."""
    print("=" * 60)
    print("DEMO 2: Analyzing Hierarchical Results")
    print("=" * 60)
    print()

    # Create synthetic results for demonstration
    print("Creating synthetic results for demonstration...")
    print("(In practice, these would come from your model's predictions)")
    print()

    results_dir = Path("data/benchmarks/hierarchical_demo")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Simulate three models with different failure patterns
    models = {
        "baseline": {
            # Baseline: Performance degrades steadily
            0: 0.95,
            1: 0.75,
            2: 0.55,
            3: 0.40,
            4: 0.35,
            5: 0.25,
        },
        "stochastok": {
            # StochasTok: Slight improvement, especially at lower levels
            0: 0.95,
            1: 0.80,
            2: 0.60,
            3: 0.50,
            4: 0.45,
            5: 0.35,
        },
        "patok": {
            # Patok: Major improvement at morphological levels (2-4)
            0: 0.95,
            1: 0.80,
            2: 0.78,
            3: 0.70,
            4: 0.68,
            5: 0.50,
        },
    }

    # Generate synthetic results for each model
    for model_name, level_accs in models.items():
        results_file = results_dir / f"results_{model_name}.jsonl"

        with open(results_file, "w") as f:
            for level, target_acc in level_accs.items():
                # Generate 50 examples per level
                for i in range(50):
                    correct = (i / 50) < target_acc
                    result = {
                        "level": level,
                        "category": "test",
                        "subcategory": f"test_{level}",
                        "correct": correct,
                        "predicted_answer": "pred",
                        "gold_answer": "gold",
                        "word": f"word_{i}",
                    }
                    f.write(json.dumps(result) + "\n")

        print(f"Created synthetic results for {model_name}: {results_file}")

    print()

    # Analyze each model
    print("Analyzing model performance...")
    print()

    analyzers = {}
    for model_name in models.keys():
        results_file = results_dir / f"results_{model_name}.jsonl"
        analyzer = HierarchicalAnalyzer(str(results_file))
        analyzers[model_name] = analyzer

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'=' * 60}")

        # Generate diagnostic report
        report = analyzer.generate_diagnostic_report()
        print(report)

    # Compare models
    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON")
    print("=" * 60)
    print()

    comparison = compare_multiple_models(analyzers)
    print("Performance Comparison:")
    print(comparison.to_string(index=False))
    print()

    # Identify where Patok helps most
    patok_vs_baseline = analyzers["patok"].compare_models(analyzers["baseline"])
    print("\nPatok vs Baseline - Improvements:")
    print("-" * 60)
    for _, row in patok_vs_baseline.iterrows():
        if row["difference"] > 0.05:
            print(f"Level {int(row['level'])}: +{row['difference']:.1%}")
            print(f"(Patok {row['model1_acc']:.1%} vs Baseline {row['model2_acc']:.1%})")

    print()

    # Key insights
    print("=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print()
    print("1. Baseline Model:")
    print("   - Fails at Level 2 (morpheme decomposition)")
    print("   - Tokenization doesn't align with morphological boundaries")
    print()
    print("2. StochasTok:")
    print("   - Modest improvements across all levels")
    print("   - Helps with subword awareness but not morphology-specific")
    print()
    print("3. Patok (Affix-Aware):")
    print("   - Major gains at Level 2 (+23% over baseline)")
    print("   - Improvements cascade to Level 3 and 4")
    print("   - Shows affix-aware tokenization provides fundamental")
    print("     morphological understanding, not just task-specific gains")
    print()


def demo_diagnostic_cascade():
    """Demo: Show how failures cascade through levels."""
    print("=" * 60)
    print("DEMO 3: Understanding Failure Cascades")
    print("=" * 60)
    print()

    print("Failure Cascade Pattern:")
    print("-" * 60)
    print()
    print("If a model FAILS at Level N, we expect FAILURE at Level N+1:")
    print()
    print("Level 0 (95%) ✓ → Model can see characters")
    print("     ↓")
    print("Level 1 (75%) ✓ → Model can manipulate characters")
    print("     ↓")
    print("Level 2 (40%) ✗ → BOTTLENECK: Cannot identify morphemes")
    print("     ↓         → Diagnosis: Tokenization ignores morphology")
    print("Level 3 (25%) ✗ → Expected: Cannot manipulate morphemes")
    print("     ↓")
    print("Level 4 (20%) ✗ → Expected: Cannot compose morphemes")
    print()
    print("Solution: Use Patok to improve Level 2 (morpheme decomposition)")
    print("Expected effect: Improvements cascade to Level 3 & 4")
    print()
    print("After Patok:")
    print("-" * 60)
    print()
    print("Level 0 (95%) ✓ → Same (Patok doesn't affect character recognition)")
    print("     ↓")
    print("Level 1 (80%) ✓ → Slight improvement")
    print("     ↓")
    print("Level 2 (78%) ✓ → MAJOR IMPROVEMENT (+38%)")
    print("     ↓")
    print("Level 3 (70%) ✓ → Cascaded improvement (+45%)")
    print("     ↓")
    print("Level 4 (68%) ✓ → Cascaded improvement (+48%)")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  HIERARCHICAL TASK FRAMEWORK DEMONSTRATION".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Check if data exists
    data_file = Path("data/corpora/pacute_data/syllables.jsonl")
    if not data_file.exists():
        print("⚠ Warning: Sample data not found")
        print(f"  Expected: {data_file}")
        print("  Running limited demo with synthetic data only...")
        print()
        demo_diagnostic_cascade()
        demo_analyze_results()
        return

    try:
        # Run all demos
        demo_generate_tasks()
        input("\nPress Enter to continue to analysis demo...")
        demo_analyze_results()
        input("\nPress Enter to continue to cascade explanation...")
        demo_diagnostic_cascade()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Generate your own hierarchical tasks with real affix data")
        print("  2. Evaluate your models on these tasks")
        print("  3. Use HierarchicalAnalyzer to identify bottlenecks")
        print("  4. Use insights to improve tokenization (e.g., Patok)")
        print()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
