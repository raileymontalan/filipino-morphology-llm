#!/usr/bin/env python3
"""
Generate Benchmarks.

Generates all benchmark datasets for evaluation:
1. PACUTE Benchmarks (affixation, composition, manipulation, syllabification)
2. Hierarchical Benchmarks (multi-level linguistic understanding)
3. CUTE Dataset (character understanding tasks)
4. LangGame Dataset (language reasoning tasks)
5. Multi-digit Addition Dataset (mathematical reasoning)

After generation, automatically adds unique IDs to all benchmark samples.

Usage:
    python scripts/generate_benchmarks.py
    python scripts/generate_benchmarks.py --benchmarks pacute hierarchical cute
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def generate_pacute():
    """Generate PACUTE benchmarks."""
    print("=" * 80)
    print("Generating PACUTE Benchmarks")
    print("=" * 80)
    print()

    from evaluation.datasets.scripts.generate_pacute_benchmarks import (
        main as generate_pacute_main,
    )

    generate_pacute_main()
    print()


def generate_hierarchical():
    """Generate hierarchical benchmarks."""
    print("=" * 80)
    print("Generating Hierarchical Benchmarks")
    print("=" * 80)
    print()

    from evaluation.datasets.scripts.generate_hierarchical_benchmark import (
        main as generate_hierarchical_main,
    )

    generate_hierarchical_main()
    print()


def generate_langgame():
    """Generate LangGame dataset."""
    print("=" * 80)
    print("Generating LangGame Dataset")
    print("=" * 80)
    print()

    try:
        from evaluation.datasets.scripts.generate_langgame_benchmark import (
            main as generate_langgame_main,
        )

        generate_langgame_main()
        print()
    except ImportError as e:
        print("Warning: Could not generate LangGame dataset")
        print(f"  {e}")
        print("  This dataset may require additional dependencies")
        print()


def generate_math():
    """Generate multi-digit addition dataset."""
    print("=" * 80)
    print("Generating Multi-digit Addition Dataset")
    print("=" * 80)
    print()

    from evaluation.datasets.scripts.generate_math_benchmark import (
        main as generate_math_main,
    )

    generate_math_main()
    print()


def generate_cute():
    """Generate CUTE dataset."""
    print("=" * 80)
    print("Generating CUTE Dataset")
    print("=" * 80)
    print()

    try:
        from evaluation.datasets.scripts.generate_cute_benchmark import (
            main as generate_cute_main,
        )

        generate_cute_main()
        print()
    except ImportError as e:
        print("Warning: Could not generate CUTE dataset")
        print(f"  {e}")
        print("  This dataset requires the 'datasets' library")
        print()


def add_ids_to_benchmarks():
    """Add unique IDs to all benchmark files."""
    print("=" * 80)
    print("Adding IDs to Benchmark Files")
    print("=" * 80)
    print()

    benchmarks_dir = Path("data/benchmarks")

    if not benchmarks_dir.exists():
        print(f"Warning: {benchmarks_dir} not found")
        return

    jsonl_files = sorted(benchmarks_dir.glob("*.jsonl"))

    for filepath in jsonl_files:
        try:
            # Parse filename to get benchmark name and format
            filename = filepath.stem  # e.g., "langgame_mcq"
            parts = filename.rsplit("_", 1)
            if len(parts) == 2:
                benchmark_name, format_type = parts
            else:
                benchmark_name = filename
                format_type = "unknown"

            # Read all samples
            samples = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line.strip()))

            # Check if IDs already exist
            has_ids = all("id" in sample for sample in samples)
            if has_ids:
                print(f"  ✓ {filepath.name} - Already has IDs")
                continue

            # Add IDs
            updated_samples = []
            for idx, sample in enumerate(samples):
                sample["id"] = f"{benchmark_name}_{format_type}_{idx:05d}"
                updated_samples.append(sample)

            # Write back
            with open(filepath, "w", encoding="utf-8") as f:
                for sample in updated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            print(f"  ✓ {filepath.name} - Added {len(updated_samples)} IDs")

        except Exception as e:
            print(f"  ✗ {filepath.name} - Error: {e}")

    print()


def main():
    """Generate all evaluation benchmarks and add unique IDs."""
    parser = argparse.ArgumentParser(
        description="Generate all evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["pacute", "hierarchical", "langgame", "math", "cute", "all"],
        default=["all"],
        help="Which benchmarks to generate (default: all)",
    )

    args = parser.parse_args()

    # Determine which benchmarks to generate
    if "all" in args.benchmarks:
        benchmarks_to_generate = ["pacute", "hierarchical", "langgame", "math", "cute"]
    else:
        benchmarks_to_generate = args.benchmarks

    print("=" * 80)
    print("Filipino Morphology LLM - Benchmark Generation")
    print("=" * 80)
    print()

    # Generate benchmarks
    success_count = 0
    fail_count = 0

    if "pacute" in benchmarks_to_generate:
        try:
            generate_pacute()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate PACUTE: {e}")
            print()
            fail_count += 1

    if "hierarchical" in benchmarks_to_generate:
        try:
            generate_hierarchical()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate Hierarchical: {e}")
            print()
            fail_count += 1

    if "langgame" in benchmarks_to_generate:
        try:
            generate_langgame()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate LangGame: {e}")
            print()
            fail_count += 1

    if "math" in benchmarks_to_generate:
        try:
            generate_math()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate Math: {e}")
            print()
            fail_count += 1

    if "cute" in benchmarks_to_generate:
        try:
            generate_cute()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate CUTE: {e}")
            print()
            fail_count += 1

    # Add IDs to all benchmark files
    if success_count > 0:
        try:
            add_ids_to_benchmarks()
        except Exception as e:
            print(f"Warning: Failed to add IDs to benchmarks: {e}")
            print()

    # Generate benchmark variants (MCQ <-> GEN conversions)
    if success_count > 0:
        try:
            print("=" * 80)
            print("Generating Benchmark Variants")
            print("=" * 80)
            print()
            from evaluation.datasets.scripts.generate_benchmark_variants import (
                main as generate_variants_main,
            )

            generate_variants_main()
            print()
        except Exception as e:
            print(f"Warning: Failed to generate benchmark variants: {e}")
            print()

    # Summary
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"{success_count} benchmarks generated successfully")
    if fail_count > 0:
        print(f"{fail_count} benchmarks failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
