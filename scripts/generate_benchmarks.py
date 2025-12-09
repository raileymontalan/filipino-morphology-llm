#!/usr/bin/env python3
"""
Generate Benchmarks

Generates all benchmark datasets for evaluation:
1. PACUTE Benchmarks (affixation, composition, manipulation, syllabification)
2. Hierarchical Benchmarks (multi-level linguistic understanding)
3. CUTE Dataset (character understanding tasks)
4. LangGame Dataset (language reasoning tasks)
5. Multi-digit Addition Dataset (mathematical reasoning)

Usage:
    python scripts/generate_benchmarks.py
    python scripts/generate_benchmarks.py --benchmarks pacute hierarchical cute
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def generate_pacute():
    """Generate PACUTE benchmarks."""
    print("=" * 80)
    print("Generating PACUTE Benchmarks")
    print("=" * 80)
    print()
    
    from evaluation.datasets.scripts.generate_pacute_benchmarks import main as generate_pacute_main
    generate_pacute_main()
    print()


def generate_hierarchical():
    """Generate hierarchical benchmarks."""
    print("=" * 80)
    print("Generating Hierarchical Benchmarks")
    print("=" * 80)
    print()
    
    from evaluation.datasets.scripts.generate_hierarchical_benchmark import main as generate_hierarchical_main
    generate_hierarchical_main()
    print()


def generate_langgame():
    """Generate LangGame dataset."""
    print("=" * 80)
    print("Generating LangGame Dataset")
    print("=" * 80)
    print()
    
    try:
        from evaluation.datasets.scripts.generate_langgame_benchmark import main as generate_langgame_main
        generate_langgame_main()
        print()
    except ImportError as e:
        print(f"Warning: Could not generate LangGame dataset")
        print(f"  {e}")
        print(f"  This dataset may require additional dependencies")
        print()


def generate_math():
    """Generate multi-digit addition dataset."""
    print("=" * 80)
    print("Generating Multi-digit Addition Dataset")
    print("=" * 80)
    print()
    
    from evaluation.datasets.scripts.generate_math_benchmark import main as generate_math_main
    generate_math_main()
    print()


def generate_cute():
    """Generate CUTE dataset."""
    print("=" * 80)
    print("Generating CUTE Dataset")
    print("=" * 80)
    print()
    
    try:
        from evaluation.datasets.scripts.generate_cute_benchmark import main as generate_cute_main
        generate_cute_main()
        print()
    except ImportError as e:
        print(f"Warning: Could not generate CUTE dataset")
        print(f"  {e}")
        print(f"  This dataset requires the 'datasets' library")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate all evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        choices=['pacute', 'hierarchical', 'langgame', 'math', 'cute', 'all'],
        default=['all'],
        help='Which benchmarks to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Determine which benchmarks to generate
    if 'all' in args.benchmarks:
        benchmarks_to_generate = ['pacute', 'hierarchical', 'langgame', 'math', 'cute']
    else:
        benchmarks_to_generate = args.benchmarks
    
    print("=" * 80)
    print("Filipino Morphology LLM - Benchmark Generation")
    print("=" * 80)
    print()
    
    # Generate benchmarks
    success_count = 0
    fail_count = 0
    
    if 'pacute' in benchmarks_to_generate:
        try:
            generate_pacute()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate PACUTE: {e}")
            print()
            fail_count += 1
    
    if 'hierarchical' in benchmarks_to_generate:
        try:
            generate_hierarchical()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate Hierarchical: {e}")
            print()
            fail_count += 1
    
    if 'langgame' in benchmarks_to_generate:
        try:
            generate_langgame()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate LangGame: {e}")
            print()
            fail_count += 1
    
    if 'math' in benchmarks_to_generate:
        try:
            generate_math()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate Math: {e}")
            print()
            fail_count += 1
    
    if 'cute' in benchmarks_to_generate:
        try:
            generate_cute()
            success_count += 1
        except Exception as e:
            print(f"Failed to generate CUTE: {e}")
            print()
            fail_count += 1
    
    # Summary
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"{success_count} benchmarks generated successfully")
    if fail_count > 0:
        print(f"{fail_count} benchmarks failed")
    print("=" * 80)


if __name__ == '__main__':
    main()
