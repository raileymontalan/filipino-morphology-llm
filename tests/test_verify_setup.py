#!/usr/bin/env python3
"""Verify monorepo setup is complete and functional."""

import os
import sys
from pathlib import Path


def check_directory_structure():
    """Check that all required directories exist."""
    print("Checking directory structure...")

    required_dirs = [
        "src/tokenization",
        "src/evaluation",
        "src/analysis",
        "data/benchmarks",
        "configs",
        "tests",
        "scripts",
        "docs",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}")

    if missing_dirs:
        print("\n  ✗ Missing directories:")
        for dir_path in missing_dirs:
            print(f"    - {dir_path}")
        return False

    print("  All directories present!\n")
    return True


def check_key_files():
    """Check that key files exist."""
    print("Checking key files...")

    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "src/__init__.py",
        "src/tokenization/__init__.py",
        "src/tokenization/base_processor.py",
        "src/tokenization/patok_morphology.py",
        "src/tokenization/stochastok_processor.py",
        "src/evaluation/__init__.py",
        "src/evaluation/downstream.py",
        "src/analysis/__init__.py",
        "src/analysis/inference_analysis.py",
        "src/analysis/visualizations.py",
        "scripts/generate_benchmarks.py",
        "scripts/run_analysis.py",
        "scripts/run_evaluation.py",
        "docs/EVALUATION.md",
        "docs/TRAINING.md",
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")

    if missing_files:
        print("\n  ✗ Missing files:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        return False

    print("  All key files present!\n")
    return True


def check_imports():
    """Check that key modules can be imported."""
    print("Checking imports...")

    sys.path.insert(0, str(Path.cwd() / "src"))

    imports_to_test = [
        ("tokenization.base_processor", ["TokenizerProcessor"]),
        ("tokenization.patok_morphology", ["MorphologyAwarePatokProcessor"]),
        ("tokenization.stochastok_processor", ["StochastokProcessor"]),
        ("evaluation.downstream", ["main"]),
        ("analysis.inference_analysis", ["main"]),
        ("analysis.visualizations", ["create_all_visualizations"]),
    ]

    failed_imports = []
    for module_name, items in imports_to_test:
        try:
            module = __import__(module_name, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    failed_imports.append(f"{module_name}.{item}")
                    print(f"  ✗ {module_name}.{item}")
                else:
                    print(f"  ✓ {module_name}.{item}")
        except ImportError as e:
            failed_imports.append(f"{module_name} ({e})")
            print(f"  ✗ {module_name}: {e}")

    if failed_imports:
        print("\n  Some imports failed.")
        return False

    print("  All imports successful!\n")
    return True


def check_data_files():
    """Check that data files are present."""
    print("Checking data files...")

    affix_file = Path("data/affixes/filipino_affixes.txt")
    if affix_file.exists():
        with open(affix_file) as f:
            num_affixes = len([line for line in f if line.strip()])
        print(f"  ✓ Filipino affixes: {num_affixes} affixes")
    else:
        print("  ⚠ Filipino affixes file not found (optional for tokenization)")
        print("    Location: data/affixes/filipino_affixes.txt")

    benchmark_dir = Path("data/benchmarks")
    if benchmark_dir.exists():
        benchmark_files = list(benchmark_dir.glob("*.jsonl"))
        print(f"  ✓ Benchmark files: {len(benchmark_files)} files")
        if benchmark_files:
            for f in sorted(benchmark_files)[:5]:  # Show first 5
                print(f"    - {f.name}")
            if len(benchmark_files) > 5:
                print(f"    ... and {len(benchmark_files) - 5} more")
    else:
        print("  ⚠ Benchmark directory not found (generate with scripts/generate_benchmarks.py)")

    print()
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Filipino Morphology-LLM Monorepo Verification")
    print("=" * 60)
    print()

    # Change to script's parent directory (should be repo root)
    os.chdir(Path(__file__).parent.parent)
    print(f"Working directory: {os.getcwd()}\n")

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Key Files", check_key_files),
        ("Data Files", check_data_files),
        ("Imports", check_imports),
    ]

    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))

    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    all_passed = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("✓ Monorepo setup is complete!")
        print("\nNext steps:")
        print("  1. Install package: pip install -e .")
        print("  2. Run tests: pytest tests/")
        print("  3. Generate benchmarks: python scripts/generate_benchmarks.py")
        print("  4. Run evaluation: python scripts/run_evaluation.py --help")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
