"""
Run comprehensive benchmark evaluation on multiple models.

Benchmarks: CUTE, LangGame, PACUTE
Models: GPT2, Gemma, Llama, Qwen, GPT-OSS (PT and IT versions)
Setting: MCQ using log probabilities, reporting F1, precision, recall, accuracy
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.loaders import load_benchmark


# Model configurations: (HuggingFace model name, type)
MODEL_CONFIGS = {
    # GPT-2
    "gpt2": ("gpt2", "pt"),
    "gpt2-medium": ("gpt2-medium", "pt"),
    "gpt2-large": ("gpt2-large", "pt"),

    # Gemma
    "gemma-2b": ("google/gemma-2b", "pt"),
    "gemma-2b-it": ("google/gemma-2b-it", "it"),
    "gemma-7b": ("google/gemma-7b", "pt"),
    "gemma-7b-it": ("google/gemma-7b-it", "it"),

    # Llama
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B", "pt"),
    "llama-3.2-1b-it": ("meta-llama/Llama-3.2-1B-Instruct", "it"),
    "llama-3.2-3b": ("meta-llama/Llama-3.2-3B", "pt"),
    "llama-3.2-3b-it": ("meta-llama/Llama-3.2-3B-Instruct", "it"),

    # Qwen
    "qwen-2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "pt"),
    "qwen-2.5-0.5b-it": ("Qwen/Qwen2.5-0.5B-Instruct", "it"),
    "qwen-2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "pt"),
    "qwen-2.5-1.5b-it": ("Qwen/Qwen2.5-1.5B-Instruct", "it"),
    "qwen-2.5-3b": ("Qwen/Qwen2.5-3B", "pt"),
    "qwen-2.5-3b-it": ("Qwen/Qwen2.5-3B-Instruct", "it"),

    # GPT-OSS (OpenGPT models - alternative open source GPT implementations)
    "cerebras-gpt-111m": ("cerebras/Cerebras-GPT-111M", "pt"),
    "cerebras-gpt-256m": ("cerebras/Cerebras-GPT-256M", "pt"),
    "cerebras-gpt-590m": ("cerebras/Cerebras-GPT-590M", "pt"),
    "cerebras-gpt-1.3b": ("cerebras/Cerebras-GPT-1.3B", "pt"),
}


class HuggingFaceEvaluator:
    """Evaluator for HuggingFace models on MCQ benchmarks."""

    def __init__(self, model_name, hf_model_name, device="cuda"):
        """
        Initialize evaluator with a HuggingFace model.

        Args:
            model_name: Short name for logging
            hf_model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.hf_model_name = hf_model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name} ({hf_model_name})")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

        print(f"Model loaded successfully")

    def compute_logprob(self, prefix, continuation):
        """
        Compute log probability of continuation given prefix.

        Args:
            prefix: The prompt/question
            continuation: The answer option to score

        Returns:
            Log probability of the continuation
        """
        # Convert to string and combine prefix and continuation
        prefix = str(prefix)
        continuation = str(continuation)
        full_text = prefix + " " + continuation

        # Tokenize
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Get continuation tokens
        continuation_tokens = full_tokens[len(prefix_tokens):]

        if len(continuation_tokens) == 0:
            return -100.0  # Very low probability for empty continuation

        # Convert to tensors
        input_ids = torch.tensor([full_tokens]).to(self.device)

        # Get logits
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Get log probabilities for continuation tokens
        log_probs = F.log_softmax(logits[0], dim=-1)

        # Sum log probabilities for the continuation tokens
        total_log_prob = 0.0
        for i, token_id in enumerate(continuation_tokens):
            # Position in the sequence (accounting for prefix)
            pos = len(prefix_tokens) + i - 1
            if pos >= 0 and pos < log_probs.shape[0]:
                total_log_prob += log_probs[pos, token_id].item()

        return total_log_prob

    def evaluate_mcq(self, prefix, ground_truth, false_options):
        """
        Evaluate a single MCQ question.

        Args:
            prefix: The question/prompt
            ground_truth: Correct answer
            false_options: List of incorrect answers

        Returns:
            Tensor of log probabilities [ground_truth_logprob, false_option1_logprob, ...]
        """
        all_options = [ground_truth] + false_options
        logprobs = []

        for option in all_options:
            logprob = self.compute_logprob(prefix, option)
            logprobs.append(logprob)

        return torch.tensor(logprobs)

    def evaluate_generative(self, prefix, ground_truth, max_new_tokens=50):
        """
        Evaluate a generative task by generating text and comparing to ground truth.

        Args:
            prefix: The question/prompt
            ground_truth: Expected answer
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generated text and whether it matches ground truth
        """
        prefix = str(prefix)
        ground_truth = str(ground_truth).strip().lower()

        # Tokenize input
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated text
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the newly generated part (remove prompt)
        generated_answer = generated[len(prefix):].strip().lower()

        # Check for exact match
        exact_match = generated_answer == ground_truth

        # Check for contains match (generated contains ground truth)
        contains_match = ground_truth in generated_answer

        # Check for prefix match (useful for short answers)
        prefix_match = generated_answer.startswith(ground_truth)

        return {
            'generated': generated_answer,
            'ground_truth': ground_truth,
            'exact_match': exact_match,
            'contains_match': contains_match,
            'prefix_match': prefix_match,
        }

    def evaluate_benchmark(self, benchmark_name, max_samples=None):
        """
        Evaluate on a benchmark.

        Args:
            benchmark_name: Name of the benchmark (cute, pacute, langgame, etc.)
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            Dictionary with results
        """
        print(f"\nEvaluating on {benchmark_name}...")

        # Load benchmark
        try:
            benchmark_loader = load_benchmark(benchmark_name)
        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            return None

        # Detect benchmark format by checking first item
        first_item = None
        benchmark_items = []
        is_generative = False

        for i, item in enumerate(benchmark_loader):
            benchmark_items.append(item)
            if i == 0:
                first_item = item
                prefix, ground_truth, false_options = item
                # Generative format has empty false_options
                is_generative = len(false_options) == 0
                format_type = "generative" if is_generative else "MCQ"
                print(f"Detected {format_type} format for {benchmark_name}")
            if max_samples and i >= max_samples - 1:
                break

        if len(benchmark_items) == 0:
            print(f"No samples loaded for {benchmark_name}")
            return None

        # Evaluate based on format
        if is_generative:
            return self._evaluate_generative_benchmark(benchmark_items, benchmark_name)
        else:
            return self._evaluate_mcq_benchmark(benchmark_items, benchmark_name)

    def _evaluate_mcq_benchmark(self, benchmark_items, benchmark_name):
        """Evaluate MCQ format benchmark."""
        confidences = []
        correct_count = 0
        total_count = 0

        for prefix, ground_truth, false_options in tqdm(benchmark_items, desc=benchmark_name):
            # Evaluate
            logprobs = self.evaluate_mcq(prefix, ground_truth, false_options)
            confidences.append(logprobs)

            # Check if correct
            predicted_idx = torch.argmax(logprobs).item()
            if predicted_idx == 0:
                correct_count += 1
            total_count += 1

        if total_count == 0:
            print(f"No samples evaluated for {benchmark_name}")
            return None

        # Pad confidences to same length
        max_length = max([len(c) for c in confidences])
        padded_confidences = []
        for c in confidences:
            padded = F.pad(c, (0, max_length - len(c)), value=-1e10)
            padded_confidences.append(padded)

        confidences_tensor = torch.stack(padded_confidences)

        # Calculate metrics
        results = self.calculate_metrics(confidences_tensor)
        results['num_samples'] = total_count
        results['format'] = 'mcq'

        return results

    def _evaluate_generative_benchmark(self, benchmark_items, benchmark_name):
        """Evaluate generative format benchmark."""
        exact_matches = 0
        contains_matches = 0
        prefix_matches = 0
        total_count = 0

        for prefix, ground_truth, _ in tqdm(benchmark_items, desc=benchmark_name):
            # Evaluate
            result = self.evaluate_generative(prefix, ground_truth)

            # Count matches
            if result['exact_match']:
                exact_matches += 1
            if result['contains_match']:
                contains_matches += 1
            if result['prefix_match']:
                prefix_matches += 1
            total_count += 1

        if total_count == 0:
            print(f"No samples evaluated for {benchmark_name}")
            return None

        # Calculate metrics
        exact_accuracy = exact_matches / total_count
        contains_accuracy = contains_matches / total_count
        prefix_accuracy = prefix_matches / total_count

        results = {
            'num_samples': total_count,
            'exact_match_accuracy': exact_accuracy,
            'contains_match_accuracy': contains_accuracy,
            'prefix_match_accuracy': prefix_accuracy,
            'format': 'generative'
        }

        return results

    def calculate_metrics(self, confidences):
        """
        Calculate evaluation metrics.

        Args:
            confidences: (B, N) tensor of log probabilities

        Returns:
            Dictionary of metrics
        """
        # Predictions
        _, predicted = torch.max(confidences, 1)

        # Accuracy
        accuracy = (predicted == 0).float().mean().item()

        # F1, Precision, Recall
        tp = (predicted == 0).float().sum().item()
        fn = (predicted != 0).float().sum().item()
        fp = 0  # In MCQ, we always predict exactly one answer

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Path confidence (softmax probability on correct answer)
        softmaxed = F.softmax(confidences, dim=-1)
        path_confidence = softmaxed[:, 0].mean().item()

        # Normalized accuracy
        num_options = confidences.shape[1]
        normalized_accuracy = (accuracy * num_options - 1) / (num_options - 1)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'path_confidence': path_confidence,
            'normalized_accuracy': normalized_accuracy,
            'num_options': num_options,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["pacute", "cute", "langgame"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="both",
        choices=["mcq", "gen", "both"],
        help="Evaluation mode: 'mcq' (MCQ only), 'gen' (generative only), or 'both' (default)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter benchmarks based on eval mode
    benchmarks_to_eval = []
    
    # Benchmark format mapping
    benchmark_formats = {
        'pacute': 'mcq',
        'pacute-mcq': 'mcq',
        'pacute-gen': 'gen',
        'pacute-affixation': 'mcq',
        'pacute-affixation-mcq': 'mcq',
        'pacute-affixation-gen': 'gen',
        'pacute-composition': 'mcq',
        'pacute-composition-mcq': 'mcq',
        'pacute-composition-gen': 'gen',
        'pacute-manipulation': 'mcq',
        'pacute-manipulation-mcq': 'mcq',
        'pacute-manipulation-gen': 'gen',
        'pacute-syllabification': 'mcq',
        'pacute-syllabification-mcq': 'mcq',
        'pacute-syllabification-gen': 'gen',
        'cute': 'gen',
        'cute-gen': 'gen',
        'hierarchical': 'mcq',
        'hierarchical-mcq': 'mcq',
        'hierarchical-gen': 'gen',
        'langgame': 'mcq',
        'langgame-mcq': 'mcq',
        'langgame-gen': 'gen',
        'multi-digit-addition': 'gen',
        'multi-digit-addition-gen': 'gen',
        'multi-digit-addition-mcq': 'mcq',
    }
    
    for benchmark in args.benchmarks:
        bench_format = benchmark_formats.get(benchmark, 'both')
        if args.eval_mode == 'both':
            benchmarks_to_eval.append(benchmark)
        elif args.eval_mode == bench_format:
            benchmarks_to_eval.append(benchmark)
        elif args.eval_mode == 'mcq' and bench_format == 'gen':
            print(f"⚠️  Skipping {benchmark} (generative-only benchmark, eval-mode=mcq)")
        elif args.eval_mode == 'gen' and bench_format == 'mcq':
            print(f"⚠️  Skipping {benchmark} (MCQ-only benchmark, eval-mode=gen)")
    
    if not benchmarks_to_eval:
        print(f"❌ No benchmarks to evaluate with eval-mode={args.eval_mode}")
        return
    
    print(f"\n{'='*80}")
    print(f"Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Mode: {args.eval_mode.upper()}")
    print(f"Benchmarks: {', '.join(benchmarks_to_eval)}")
    print(f"{'='*80}")

    # Run evaluations
    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*80}")

        hf_model_name, model_type = MODEL_CONFIGS[model_name]

        try:
            # Initialize evaluator
            evaluator = HuggingFaceEvaluator(
                model_name=model_name,
                hf_model_name=hf_model_name,
                device=args.device
            )

            # Evaluate on each benchmark
            model_results = {}
            for benchmark_name in benchmarks_to_eval:
                results = evaluator.evaluate_benchmark(
                    benchmark_name=benchmark_name,
                    max_samples=args.max_samples
                )

                if results:
                    model_results[benchmark_name] = results

                    # Print results based on format
                    print(f"\n{benchmark_name} Results:")
                    print(f"  Samples: {results['num_samples']}")

                    if results.get('format') == 'generative':
                        print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
                        print(f"  Contains Match Accuracy: {results['contains_match_accuracy']:.4f}")
                        print(f"  Prefix Match Accuracy: {results['prefix_match_accuracy']:.4f}")
                    else:
                        print(f"  Accuracy: {results['accuracy']:.4f}")
                        print(f"  F1 Score: {results['f1_score']:.4f}")
                        print(f"  Precision: {results['precision']:.4f}")
                        print(f"  Recall: {results['recall']:.4f}")
                        print(f"  Path Confidence: {results['path_confidence']:.4f}")
                        print(f"  Normalized Accuracy: {results['normalized_accuracy']:.4f}")

            all_results[model_name] = {
                'hf_model_name': hf_model_name,
                'model_type': model_type,
                'benchmarks': model_results
            }

            # Save results for this model
            model_output_dir = os.path.join("results", model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            model_output_file = os.path.join(
                model_output_dir,
                f"evaluation_results_{timestamp}.json"
            )
            
            model_result = {
                'hf_model_name': hf_model_name,
                'model_type': model_type,
                'benchmarks': model_results,
                'timestamp': timestamp
            }
            
            with open(model_output_file, 'w') as f:
                json.dump(model_result, f, indent=2)
            
            print(f"\n{'='*80}")
            print(f"Results for {model_name} saved to: {model_output_file}")
            print(f"{'='*80}")

            # Clean up GPU memory
            del evaluator
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined results (for backwards compatibility)
    if args.output_dir:
        output_file = os.path.join(
            args.output_dir,
            f"evaluation_results_{timestamp}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Combined results saved to: {output_file}")
        print(f"{'='*80}")

    # Print summary table
    print("\nSummary Table:")
    print(f"{'Model':<25} {'Benchmark':<20} {'Format':<12} {'Primary Metric':<20}")
    print("-" * 85)

    for model_name, model_data in all_results.items():
        for benchmark_name, results in model_data['benchmarks'].items():
            format_type = results.get('format', 'mcq')
            if format_type == 'generative':
                metric_str = f"Exact: {results['exact_match_accuracy']:.4f}"
            else:
                metric_str = f"Acc: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}"

            print(
                f"{model_name:<25} {benchmark_name:<20} "
                f"{format_type:<12} {metric_str:<20}"
            )


if __name__ == "__main__":
    main()
