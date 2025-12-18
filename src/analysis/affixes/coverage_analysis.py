#!/usr/bin/env python3
"""
Analyze affix coverage across different tokenizers and generate decomposition tables.

Usage:
    python scripts/analyze_affix_coverage.py --tokenizer gpt2
    python scripts/analyze_affix_coverage.py --compare gpt2 cl100k_base
"""

import argparse
import os
import sys
from pathlib import Path

from src.tokenization.affix_decomposition import AffixDecomposer, compare_tokenizers

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    """Analyze affix coverage across different tokenizers."""
    parser = argparse.ArgumentParser(description="Analyze affix vocabulary coverage")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to analyze (gpt2, cl100k_base, etc.)",
    )
    parser.add_argument(
        "--affixes-file",
        type=str,
        default="data/affixes/filipino_affixes.txt",
        help="Path to affixes file",
    )
    parser.add_argument("--compare", nargs="+", help="Compare multiple tokenizers")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vocabularies",
        help="Output directory for analysis files",
    )
    parser.add_argument(
        "--export-table",
        action="store_true",
        help="Export decomposition table for Patok",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if affixes file exists
    affixes_file = Path(args.affixes_file)
    if not affixes_file.exists():
        print(f"Error: Affixes file not found: {affixes_file}")
        print("\nPlease ensure the file exists with one affix per line.")
        print("Example content:")
        print("  mag-")
        print("  nag-")
        print("  -um-")
        print("  -in-")
        return 1

    # Compare multiple tokenizers
    if args.compare:
        print("=" * 70)
        print("COMPARING TOKENIZERS")
        print("=" * 70)
        print()

        try:
            comparison_df = compare_tokenizers(args.compare, str(affixes_file))
            print(comparison_df.to_string(index=False))
            print()

            # Save comparison
            comparison_file = output_dir / "tokenizer_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"Saved comparison to {comparison_file}")
            print()

            # Find best tokenizer
            best = comparison_df.loc[comparison_df["coverage_rate"].idxmax()]
            print(f"Best coverage: {best['tokenizer']} ({best['coverage_rate']:.1%})")
            print()

        except Exception as e:
            print(f"Error comparing tokenizers: {e}")
            return 1

    # Analyze single tokenizer
    else:
        print(f"Analyzing tokenizer: {args.tokenizer}")
        print()

        try:
            decomposer = AffixDecomposer(args.tokenizer, str(affixes_file))

            # Generate report
            report = decomposer.generate_report()
            print(report)

            # Export detailed analysis
            analysis_file = output_dir / f"affix_analysis_{args.tokenizer}.json"
            decomposer.export_analysis(str(analysis_file))
            print(f"\nDetailed analysis saved to {analysis_file}")

            # Export decomposition table if requested
            if args.export_table:
                decomp_table = decomposer.build_decomposition_table()

                # Save as JSON
                import json

                table_file = output_dir / f"decomposition_table_{args.tokenizer}.json"
                with open(table_file, "w") as f:
                    # Convert token IDs to strings for JSON serialization
                    json_table = {affix: token_ids for affix, token_ids in decomp_table.items()}
                    json.dump(json_table, f, indent=2)

                print(f"Decomposition table saved to {table_file}")
                print(f"  (Contains {len(decomp_table)} affix mappings)")
                print()

                # Show example entries
                print("Example decompositions:")
                for affix, token_ids in list(decomp_table.items())[:5]:
                    decomp = decomposer.get_best_decomposition(affix)
                    if decomp:
                        print(f"  {decomp}")

        except Exception as e:
            print(f"Error analyzing tokenizer: {e}")
            import traceback

            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
