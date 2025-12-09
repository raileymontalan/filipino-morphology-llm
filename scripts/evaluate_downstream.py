"""
Evaluate LLM on downstream tasks: CUTE, PACUTE, LangGame

Usage:
    python scripts/evaluate_downstream.py --model gpt2 --benchmark cute
    python scripts/evaluate_downstream.py --model gpt2 --benchmark pacute
    python scripts/evaluate_downstream.py --model gpt2 --benchmark langgame
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def evaluate_benchmark(benchmark_name, model_name="gpt2", max_samples=None):
    """
    Evaluate a model on a benchmark.

    Args:
        benchmark_name: Name of benchmark (cute, pacute, langgame, etc.)
        model_name: Model to evaluate (for now just prints tasks)
        max_samples: Maximum number of samples to evaluate (None = all)
    """
    print(f"Loading benchmark: {benchmark_name}")
    print(f"Model: {model_name}")
    print("=" * 80)

    # Load appropriate benchmark
    from evaluation.loaders import load_benchmark
    
    try:
        loader = load_benchmark(benchmark_name)
    except KeyError:
        # Try with more specific format
        if benchmark_name.startswith("pacute-"):
            category = benchmark_name.split("-")[1]
            from evaluation.loaders.pacute import load_pacute
            loader = load_pacute(categories=[category])
        elif benchmark_name.startswith("langgame"):
            split = benchmark_name.split("-")[1] if "-" in benchmark_name else "val"
            from evaluation.loaders.langgame import load_langgame
            loader = load_langgame(split=split)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Collect tasks
    tasks = []
    for i, (prefix, ground_truth, false_options) in enumerate(loader):
        if max_samples and i >= max_samples:
            break
        tasks.append({
            'prefix': prefix,
            'ground_truth': ground_truth,
            'false_options': false_options
        })

    print(f"\nLoaded {len(tasks)} tasks")
    print("\nSample tasks:")
    print("-" * 80)

    for i, task in enumerate(tasks[:5]):
        print(f"\n{i+1}. Question: {task['prefix']}")
        print(f"   Correct: {task['ground_truth']}")
        print(f"   Incorrect: {task['false_options']}")

    print("\n" + "=" * 80)
    print("EVALUATION SETUP COMPLETE")
    print("=" * 80)
    print(f"\nTo implement full evaluation:")
    print("1. Load your LLM model")
    print("2. For each task, compute log probabilities for all options")
    print("3. Select option with highest probability")
    print("4. Compare to ground_truth")
    print("5. Calculate accuracy")

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Filipino morphology benchmarks")
    parser.add_argument("--benchmark", type=str, required=True,
                       choices=["cute", "pacute", "pacute-affixation", "pacute-composition",
                               "pacute-manipulation", "pacute-syllabification",
                               "langgame", "langgame-mcq", "langgame-gen"],
                       help="Benchmark to evaluate on")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name (for logging)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")

    args = parser.parse_args()

    tasks = evaluate_benchmark(args.benchmark, args.model, args.max_samples)

    if args.output:
        output_data = {
            'benchmark': args.benchmark,
            'model': args.model,
            'num_tasks': len(tasks),
            'tasks': tasks
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
