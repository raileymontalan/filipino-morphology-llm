#!/usr/bin/env python3
"""
Analysis Runner.

Unified script to run various analysis tools on Filipino morphology tokenization.

Usage:
    # Tokenization analysis
    python scripts/run_analysis.py tokenization simple --limit 100
    python scripts/run_analysis.py tokenization comprehensive --limit 100
    python scripts/run_analysis.py tokenization compare --limit 100

    # Affix analysis
    python scripts/run_analysis.py affixes coverage --tokenizer gpt2
    python scripts/run_analysis.py affixes coverage --compare gpt2 cl100k_base

    # Dataset analysis
    python scripts/run_analysis.py datasets compare --baseline raileymontalan/SEA-PILE-v2-tl-tokenized

    # Inference analysis
    python scripts/run_analysis.py inference results.jsonl --format mcq

    # Visualizations
    python scripts/run_analysis.py visualize results_dir/ --output plots/
"""

import argparse
import sys
from pathlib import Path

from setup_paths import setup_project_paths

# Add src to path
setup_project_paths()


def run_tokenization_simple(args):
    """Run simple tokenization analysis."""
    sys.argv = ["simple_analysis.py", "--limit", str(args.limit)]
    if args.annotations:
        sys.argv.extend(["--annotations", args.annotations])

    # Import and run the main function
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "simple_analysis",
        Path(__file__).parent.parent / "src" / "analysis" / "tokenization" / "simple_analysis.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_tokenization_comprehensive(args):
    """Run comprehensive tokenization analysis."""
    sys.argv = ["comprehensive_analysis.py", "--limit", str(args.limit)]
    if args.annotations:
        sys.argv.extend(["--annotations", args.annotations])
    if args.output:
        sys.argv.extend(["--output", args.output])

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "comprehensive_analysis",
        Path(__file__).parent.parent / "src" / "analysis" / "tokenization" / "comprehensive_analysis.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_tokenization_compare(args):
    """Run tokenizer comparison."""
    sys.argv = ["compare_tokenizers.py", "--limit", str(args.limit)]
    if args.annotations:
        sys.argv.extend(["--annotations", args.annotations])
    if args.output:
        sys.argv.extend(["--output", args.output])

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_tokenizers",
        Path(__file__).parent.parent / "src" / "analysis" / "tokenization" / "compare_tokenizers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_affix_coverage(args):
    """Run affix coverage analysis."""
    sys.argv = ["coverage_analysis.py"]
    if args.tokenizer:
        sys.argv.extend(["--tokenizer", args.tokenizer])
    if args.compare:
        sys.argv.extend(["--compare"] + args.compare)
    if args.affixes_file:
        sys.argv.extend(["--affixes-file", args.affixes_file])
    if args.output:
        sys.argv.extend(["--output", args.output])

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "coverage_analysis",
        Path(__file__).parent.parent / "src" / "analysis" / "affixes" / "coverage_analysis.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_dataset_compare(args):
    """Run dataset comparison."""
    sys.argv = ["compare_datasets.py"]
    if args.baseline:
        sys.argv.extend(["--baseline", args.baseline])
    if args.stochastok:
        sys.argv.extend(["--stochastok", args.stochastok])
    if args.patok:
        sys.argv.extend(["--patok", args.patok])
    if args.num_samples:
        sys.argv.extend(["--num-samples", str(args.num_samples)])
    if args.output:
        sys.argv.extend(["--output", args.output])

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_datasets",
        Path(__file__).parent.parent / "src" / "analysis" / "datasets" / "compare_datasets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main():
    """Run various analysis tools for Filipino morphology tokenization."""
    parser = argparse.ArgumentParser(
        description="Run various analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="category", help="Analysis category")

    # Tokenization analysis
    tokenization_parser = subparsers.add_parser("tokenization", help="Tokenization analysis")
    tokenization_subparsers = tokenization_parser.add_subparsers(dest="analysis", help="Analysis type")

    # Simple tokenization analysis
    simple_parser = tokenization_subparsers.add_parser("simple", help="Simple tokenization analysis")
    simple_parser.add_argument("--limit", type=int, default=100, help="Number of samples to analyze")
    simple_parser.add_argument(
        "--annotations",
        type=str,
        default="data/corpora/affix_annotations.jsonl",
        help="Path to annotations file",
    )

    # Comprehensive tokenization analysis
    comprehensive_parser = tokenization_subparsers.add_parser("comprehensive", help="Comprehensive analysis")
    comprehensive_parser.add_argument("--limit", type=int, default=100, help="Number of samples to analyze")
    comprehensive_parser.add_argument(
        "--annotations",
        type=str,
        default="data/corpora/affix_annotations.jsonl",
        help="Path to annotations file",
    )
    comprehensive_parser.add_argument("--output", type=str, help="Output file for results")

    # Compare tokenizers
    compare_parser = tokenization_subparsers.add_parser("compare", help="Compare tokenizers")
    compare_parser.add_argument("--limit", type=int, default=100, help="Number of samples to analyze")
    compare_parser.add_argument(
        "--annotations",
        type=str,
        default="data/corpora/affix_annotations.jsonl",
        help="Path to annotations file",
    )
    compare_parser.add_argument("--output", type=str, help="Output file for results")

    # Affix analysis
    affix_parser = subparsers.add_parser("affixes", help="Affix analysis")
    affix_subparsers = affix_parser.add_subparsers(dest="analysis", help="Analysis type")

    # Affix coverage
    coverage_parser = affix_subparsers.add_parser("coverage", help="Affix coverage analysis")
    coverage_parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to analyze")
    coverage_parser.add_argument("--compare", nargs="+", help="Tokenizers to compare")
    coverage_parser.add_argument(
        "--affixes-file",
        type=str,
        default="data/affixes/filipino_affixes.txt",
        help="Path to affixes file",
    )
    coverage_parser.add_argument("--output", type=str, help="Output file for results")

    # Dataset analysis
    dataset_parser = subparsers.add_parser("datasets", help="Dataset analysis")
    dataset_subparsers = dataset_parser.add_subparsers(dest="analysis", help="Analysis type")

    # Compare datasets
    compare_ds_parser = dataset_subparsers.add_parser("compare", help="Compare tokenized datasets")
    compare_ds_parser.add_argument(
        "--baseline",
        type=str,
        default="raileymontalan/SEA-PILE-v2-tl-tokenized",
        help="Baseline dataset",
    )
    compare_ds_parser.add_argument(
        "--stochastok",
        type=str,
        default="raileymontalan/SEA-PILE-v2-tl-tokenized-stochastok0.1",
        help="StochasTok dataset",
    )
    compare_ds_parser.add_argument(
        "--patok",
        type=str,
        default="raileymontalan/SEA-PILE-v2-tl-tokenized-patok0.3-0.3-0.7",
        help="Patok dataset",
    )
    compare_ds_parser.add_argument("--num-samples", type=int, help="Number of samples to compare")
    compare_ds_parser.add_argument("--output", type=str, help="Output file for results")

    # Inference analysis
    inference_parser = subparsers.add_parser("inference", help="Analyze inference results")
    inference_parser.add_argument("file", type=str, help="Inference results file (JSONL)")
    inference_parser.add_argument(
        "--format",
        type=str,
        choices=["mcq", "generative"],
        default="mcq",
        help="Result format",
    )
    inference_parser.add_argument("--output", type=str, help="Output file for analysis")

    # Visualizations
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("results_dir", type=str, help="Directory with evaluation results")
    viz_parser.add_argument("--output", type=str, default="plots/", help="Output directory for plots")
    viz_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pd", "svg"],
        default="png",
        help="Output format",
    )

    args = parser.parse_args()

    if not args.category:
        parser.print_help()
        return

    # Route to appropriate handler
    if args.category == "tokenization":
        if args.analysis == "simple":
            run_tokenization_simple(args)
        elif args.analysis == "comprehensive":
            run_tokenization_comprehensive(args)
        elif args.analysis == "compare":
            run_tokenization_compare(args)
        else:
            tokenization_parser.print_help()

    elif args.category == "affixes":
        if args.analysis == "coverage":
            run_affix_coverage(args)
        else:
            affix_parser.print_help()

    elif args.category == "datasets":
        if args.analysis == "compare":
            run_dataset_compare(args)
        else:
            dataset_parser.print_help()

    elif args.category == "inference":
        run_inference_analysis(args)

    elif args.category == "visualize":
        run_visualizations(args)

    else:
        parser.print_help()


def run_inference_analysis(args):
    """Run inference results analysis."""
    from analysis.inference_analysis import (
        analyze_generative_results,
        analyze_mcq_results,
        load_inference_results,
    )

    print(f"Loading inference results from: {args.file}")
    results = load_inference_results(args.file)

    if args.format == "mcq":
        analyze_mcq_results(results, output_file=args.output)
    else:
        analyze_generative_results(results, output_file=args.output)


def run_visualizations(args):
    """Create visualizations from results."""
    from pathlib import Path

    from analysis.visualizations import create_all_visualizations

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating visualizations from: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}")

    create_all_visualizations(results_dir=str(results_dir), output_dir=str(output_dir), fmt=args.format)
    print(f"✓ Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
