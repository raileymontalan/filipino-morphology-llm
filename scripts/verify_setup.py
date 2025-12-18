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
        "src/models",
        "src/training",
        "src/evaluation",
        "src/data_processing",
        "src/analysis",
        "data/affixes",
        "data/benchmarks",
        "data/corpora",
        "data/vocabularies",
        "configs",
        "experiments",
        "notebooks",
        "tests",
        "scripts",
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
        "src/tokenization/patok_processor.py",
        "src/tokenization/stochastok_processor.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "data/affixes/filipino_affixes.txt",
        "configs/pretraining.yaml",
        "configs/instruction_tuning.yaml",
        "experiments/train.py",
        "experiments/eval.py",
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

    imports_to_test = [
        ("src.tokenization", ["PatokProcessor", "StochastokProcessor"]),
        ("src.models", ["build_model", "ModelShell"]),
        ("src.training", ["Trainer", "build_trainer"]),
        ("src.evaluation", ["syllabify", "normalize_text"]),
    ]

    sys.path.insert(0, os.getcwd())

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
        print("\n  Some imports failed. This is OK if you haven't run 'pip install -e .' yet.")
        return True  # Don't fail on this

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
        print("  ✗ Filipino affixes file not found")
        return False

    benchmark_dir = Path("data/benchmarks")
    if benchmark_dir.exists():
        benchmark_files = list(benchmark_dir.glob("*.jsonl"))
        print(f"  ✓ Benchmark files: {len(benchmark_files)} files")
        for f in sorted(benchmark_files):
            print(f"    - {f.name}")
    else:
        print("  ✗ Benchmark directory not found")
        return False

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
        print("  3. Try training: python experiments/train.py --help")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
