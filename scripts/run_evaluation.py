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
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.loaders import load_benchmark


def load_model_configs(config_path=None):
    """
    Load model configurations from YAML file.
    
    Args:
        config_path: Path to the YAML config file. If None, uses default location.
    
    Returns:
        Dictionary mapping model names to (path, type) tuples
    """
    if config_path is None:
        # Default to configs/models.yaml
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "configs" / "models.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert YAML format to the expected dictionary format
    model_configs = {}
    for model_name, model_info in config['models'].items():
        model_configs[model_name] = (model_info['path'], model_info['type'])
    
    return model_configs


# Load model configurations from YAML
MODEL_CONFIGS = load_model_configs()


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

    def evaluate_benchmark(self, benchmark_name, max_samples=None, check_existing=True, timestamp=None):
        """
        Evaluate on a benchmark.

        Args:
            benchmark_name: Name of the benchmark (cute, pacute, langgame, etc.)
            max_samples: Maximum number of samples to evaluate (None = all)
            check_existing: Whether to check for existing inference results (default: True)
            timestamp: Timestamp for output files (if None, uses current time)

        Returns:
            Dictionary with results, or None if skipped
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
                # Unpack 4 values: prefix, ground_truth, false_options, sample_id
                prefix, ground_truth, false_options, sample_id = item
                # Generative format has empty false_options
                is_generative = len(false_options) == 0
                format_type = "generative" if is_generative else "MCQ"
                print(f"Detected {format_type} format for {benchmark_name}")
            if max_samples and i >= max_samples - 1:
                break

        if len(benchmark_items) == 0:
            print(f"No samples loaded for {benchmark_name}")
            return None

        # Determine setting for metadata
        setting = "gen" if is_generative else "mcq"
        
        # Check if inference results already exist
        if check_existing:
            inference_dir = os.path.join("results", self.model_name, "inference")
            inference_file = os.path.join(
                inference_dir,
                f"{benchmark_name}.jsonl"
            )
            if os.path.exists(inference_file):
                print(f"\n{'='*80}")
                print(f"⏭️  SKIPPING: {benchmark_name}")
                print(f"{'='*80}")
                print(f"Reason: Inference results already exist")
                print(f"File: {inference_file}")
                print(f"Note: Use --overwrite flag to re-run evaluation")
                print(f"{'='*80}\n")
                return {'skipped': True, 'inference_file': inference_file}

        # Evaluate based on format
        if is_generative:
            return self._evaluate_generative_benchmark(benchmark_items, benchmark_name, setting=setting, timestamp=timestamp)
        else:
            return self._evaluate_mcq_benchmark(benchmark_items, benchmark_name, setting=setting, timestamp=timestamp)

    def _evaluate_mcq_benchmark(self, benchmark_items, benchmark_name, setting=None, timestamp=None):
        """Evaluate MCQ format benchmark."""
        confidences = []
        correct_count = 0
        total_count = 0
        detailed_results = []

        for item_data in tqdm(benchmark_items, desc=benchmark_name):
            prefix, ground_truth, false_options, sample_id = item_data
            
            # Evaluate
            logprobs = self.evaluate_mcq(prefix, ground_truth, false_options)
            confidences.append(logprobs)

            # Check if correct
            predicted_idx = torch.argmax(logprobs).item()
            is_correct = predicted_idx == 0
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store detailed result
            all_options = [ground_truth] + false_options
            detailed_result = {
                'id': sample_id,
                'question': prefix,
                'ground_truth': ground_truth,
                'options': all_options,
                'predicted_idx': predicted_idx,
                'predicted_answer': all_options[predicted_idx] if predicted_idx < len(all_options) else None,
                'is_correct': is_correct,
                'logprobs': logprobs.tolist(),
            }
            detailed_results.append(detailed_result)

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
        results['detailed_results'] = detailed_results
        results['setting'] = setting
        results['timestamp'] = timestamp

        return results

    def _evaluate_generative_benchmark(self, benchmark_items, benchmark_name, setting=None, timestamp=None):
        """Evaluate generative format benchmark."""
        exact_matches = 0
        contains_matches = 0
        prefix_matches = 0
        total_count = 0
        detailed_results = []

        for item_data in tqdm(benchmark_items, desc=benchmark_name):
            prefix, ground_truth, _, sample_id = item_data
            
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
            
            # Store detailed result
            detailed_result = {
                'id': sample_id,
                'question': prefix,
                'ground_truth': ground_truth,
                'generated': result['generated'],
                'exact_match': result['exact_match'],
                'contains_match': result['contains_match'],
                'prefix_match': result['prefix_match'],
            }
            detailed_results.append(detailed_result)

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
            'format': 'generative',
            'detailed_results': detailed_results,
            'setting': setting,
            'timestamp': timestamp
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
        "--model-config",
        type=str,
        default=None,
        help="Path to model configuration YAML file (default: configs/models.yaml)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        help="Models to evaluate (must be defined in model config)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "pacute-affixation-mcq", 
            "pacute-composition-mcq",
            "pacute-manipulation-mcq",
            "pacute-syllabification-mcq",
            "hierarchical-mcq",
            "langgame-mcq",
            "multi-digit-addition-mcq",
            "cute-gen",
            "pacute-affixation-gen",
            "pacute-composition-gen",
            "pacute-manipulation-gen",
            "pacute-syllabification-gen",
            "hierarchical-gen",
            "langgame-gen",
            "multi-digit-addition-gen",
        ],
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference results (default: False, skip if results exist)"
    )

    args = parser.parse_args()
    
    # Load model configurations (reload if custom config specified)
    if args.model_config:
        global MODEL_CONFIGS
        MODEL_CONFIGS = load_model_configs(args.model_config)
        print(f"Loaded custom model config from: {args.model_config}")
    
    # Validate model choices
    invalid_models = [m for m in args.models if m not in MODEL_CONFIGS]
    if invalid_models:
        print(f"Error: Invalid model names: {invalid_models}")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return

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
        'multi-digit-addition': 'mcq',
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
    print(f"Overwrite existing results: {args.overwrite}")
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
                    max_samples=args.max_samples,
                    check_existing=(not args.overwrite),
                    timestamp=timestamp
                )

                if results:
                    # Check if evaluation was skipped
                    if results.get('skipped'):
                        continue
                    
                    # Save detailed inference results
                    detailed_results = results.pop('detailed_results', None)
                    setting = results.pop('setting', None)
                    if detailed_results:
                        inference_dir = os.path.join("results", model_name, "inference")
                        os.makedirs(inference_dir, exist_ok=True)
                        inference_file = os.path.join(
                            inference_dir,
                            f"{benchmark_name}.jsonl"
                        )
                        with open(inference_file, 'w') as f:
                            for result in detailed_results:
                                f.write(json.dumps(result) + '\n')
                        print(f"Saved detailed inference results to: {inference_file}")
                    
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
