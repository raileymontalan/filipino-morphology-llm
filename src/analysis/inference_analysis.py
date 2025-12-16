"""
Analyze inference results to identify error patterns and model performance.

This script analyzes the detailed inference results saved during evaluation
to help identify:
- Which samples models got correct/incorrect
- Common error patterns
- Performance across different question types
- Comparison between models on same samples
"""
import json
import os
from collections import defaultdict
from pathlib import Path
import argparse


def load_inference_results(inference_file):
    """Load inference results from JSONL file."""
    results = []
    with open(inference_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def analyze_mcq_results(results, output_file=None):
    """Analyze MCQ format results."""
    print(f"\nAnalyzing {len(results)} MCQ samples...")
    
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    print(f"  Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    print(f"  Incorrect: {len(incorrect)} ({len(incorrect)/len(results)*100:.1f}%)")
    
    # Analyze confidence
    correct_confidences = [max(r['logprobs']) for r in correct]
    incorrect_confidences = [max(r['logprobs']) for r in incorrect]
    
    if correct_confidences:
        print(f"  Avg confidence (correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
    if incorrect_confidences:
        print(f"  Avg confidence (incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.4f}")
    
    # Show top errors
    if incorrect:
        print(f"\nTop 10 Errors:")
        for i, result in enumerate(incorrect[:10], 1):
            print(f"\n{i}. ID: {result['id']}")
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Ground Truth: {result['ground_truth']}")
            print(f"   Predicted: {result['predicted_answer']}")
            print(f"   Predicted Index: {result['predicted_idx']}")
    
    # Save detailed analysis if output file specified
    if output_file:
        analysis = {
            'total_samples': len(results),
            'correct_count': len(correct),
            'incorrect_count': len(incorrect),
            'accuracy': len(correct) / len(results),
            'correct_samples': [r['id'] for r in correct],
            'incorrect_samples': [
                {
                    'id': r['id'],
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'predicted': r['predicted_answer'],
                    'predicted_idx': r['predicted_idx'],
                    'logprobs': r['logprobs']
                } for r in incorrect
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")


def analyze_generative_results(results, output_file=None):
    """Analyze generative format results."""
    print(f"\nAnalyzing {len(results)} generative samples...")
    
    exact_matches = [r for r in results if r['exact_match']]
    contains_matches = [r for r in results if r['contains_match'] and not r['exact_match']]
    prefix_matches = [r for r in results if r['prefix_match'] and not r['contains_match']]
    no_match = [r for r in results if not r['exact_match'] and not r['contains_match'] and not r['prefix_match']]
    
    print(f"  Exact Match: {len(exact_matches)} ({len(exact_matches)/len(results)*100:.1f}%)")
    print(f"  Contains Match: {len(contains_matches)} ({len(contains_matches)/len(results)*100:.1f}%)")
    print(f"  Prefix Match: {len(prefix_matches)} ({len(prefix_matches)/len(results)*100:.1f}%)")
    print(f"  No Match: {len(no_match)} ({len(no_match)/len(results)*100:.1f}%)")
    
    # Show top errors
    if no_match:
        print(f"\nTop 10 Complete Failures:")
        for i, result in enumerate(no_match[:10], 1):
            print(f"\n{i}. ID: {result['id']}")
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Ground Truth: {result['ground_truth']}")
            print(f"   Generated: {result['generated']}")
    
    # Save detailed analysis if output file specified
    if output_file:
        analysis = {
            'total_samples': len(results),
            'exact_match_count': len(exact_matches),
            'contains_match_count': len(contains_matches),
            'prefix_match_count': len(prefix_matches),
            'no_match_count': len(no_match),
            'exact_match_accuracy': len(exact_matches) / len(results),
            'exact_match_samples': [r['id'] for r in exact_matches],
            'no_match_samples': [
                {
                    'id': r['id'],
                    'question': r['question'],
                    'ground_truth': r['ground_truth'],
                    'generated': r['generated']
                } for r in no_match
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")


def compare_models(inference_files, output_file=None):
    """Compare multiple models on the same samples."""
    print(f"\nComparing {len(inference_files)} models...")
    
    # Load all results
    all_results = {}
    for model_name, filepath in inference_files.items():
        all_results[model_name] = {r['id']: r for r in load_inference_results(filepath)}
    
    # Find common samples
    common_ids = set.intersection(*[set(results.keys()) for results in all_results.values()])
    print(f"  Common samples: {len(common_ids)}")
    
    # Compare performance
    for sample_id in sorted(common_ids):
        sample_results = {model: results[sample_id] for model, results in all_results.items()}
        
        # Check if results differ
        if len(set(r.get('is_correct', r.get('exact_match')) for r in sample_results.values())) > 1:
            print(f"\nSample {sample_id}:")
            print(f"  Question: {sample_results[list(sample_results.keys())[0]]['question'][:80]}...")
            for model, result in sample_results.items():
                if 'is_correct' in result:
                    print(f"  {model}: {'✓' if result['is_correct'] else '✗'} (predicted: {result['predicted_answer']})")
                else:
                    print(f"  {model}: {'✓' if result['exact_match'] else '✗'} (generated: {result['generated'][:50]}...)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inference results from evaluation"
    )
    parser.add_argument(
        "--inference-file",
        type=str,
        help="Path to inference results JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (to auto-locate inference files)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark name (to auto-locate inference files)"
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        help="List of models to compare (requires --benchmark)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for detailed analysis (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.compare_models:
        if not args.benchmark:
            print("Error: --benchmark required when comparing models")
            return
        
        # Find inference files for each model
        inference_files = {}
        for model in args.compare_models:
            inference_dir = Path(f"results/{model}/inference")
            if not inference_dir.exists():
                print(f"Warning: No inference directory for {model}")
                continue
            
            # Find most recent file for this benchmark
            files = list(inference_dir.glob(f"{args.benchmark}_*.jsonl"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                inference_files[model] = str(latest)
        
        if len(inference_files) < 2:
            print("Error: Need at least 2 models with inference files")
            return
        
        compare_models(inference_files, args.output)
    
    elif args.inference_file:
        results = load_inference_results(args.inference_file)
        if not results:
            print("No results found")
            return
        
        # Determine format
        if 'is_correct' in results[0]:
            analyze_mcq_results(results, args.output)
        elif 'exact_match' in results[0]:
            analyze_generative_results(results, args.output)
        else:
            print("Unknown result format")
    
    elif args.model and args.benchmark:
        # Auto-locate inference file
        inference_dir = Path(f"results/{args.model}/inference")
        if not inference_dir.exists():
            print(f"Error: No inference directory for {args.model}")
            return
        
        files = list(inference_dir.glob(f"{args.benchmark}_*.jsonl"))
        if not files:
            print(f"Error: No inference files for {args.benchmark}")
            return
        
        latest = max(files, key=lambda p: p.stat().st_mtime)
        print(f"Analyzing: {latest}")
        
        results = load_inference_results(str(latest))
        if 'is_correct' in results[0]:
            analyze_mcq_results(results, args.output)
        elif 'exact_match' in results[0]:
            analyze_generative_results(results, args.output)
    
    else:
        print("Error: Specify --inference-file or --model and --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()
