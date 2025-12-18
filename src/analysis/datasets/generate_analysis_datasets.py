#!/usr/bin/env python3
"""
Generate Analysis Datasets.

Generates datasets needed for Filipino morphology tokenization analysis:
- Affix annotations with morpheme boundaries
- Additional analysis datasets as needed

Usage:
    python scripts/generate_analysis_datasets.py
    python scripts/generate_analysis_datasets.py --annotations-only
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def generate_affix_annotations():
    """Generate affix annotations for tokenization analysis."""
    print("=" * 80)
    print("Generating Affix Annotations")
    print("=" * 80)
    print()

    from analysis.datasets.generate_annotations import main as generate_annotations_main

    generate_annotations_main()
    print()


def main():
    """Generate datasets needed for Filipino morphology tokenization analysis."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Generate only affix annotations",
    )

    parser.parse_args()

    print("=" * 80)
    print("Filipino Morphology Analysis - Dataset Generation")
    print("=" * 80)
    print()

    # Generate affix annotations
    generate_affix_annotations()

    # Add more dataset generation steps here as needed
    # if not args.annotations_only:
    #     generate_other_dataset()

    print("=" * 80)
    print("✓ All analysis datasets generated successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
