"""
Run comprehensive benchmark evaluation on multiple models.

Benchmarks: CUTE, LangGame, PACUTE
Models: GPT2, Gemma, Llama, Qwen, GPT-OSS (PT and IT versions)
Setting: MCQ using log probabilities, reporting F1, precision, recall, accuracy
"""

import sys
from pathlib import Path

# Add project root to path so we can import setup_paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from setup_paths import setup_project_paths

setup_project_paths()

import argparse
import json
import os
from datetime import datetime

import asyncio
import re

import torch
import torch.nn.functional as F
import yaml
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.loaders import load_benchmark

# Add src to path


def load_model_configs(config_paths=None):
    """
    Load model configurations from one or more YAML files, merging all entries.

    Args:
        config_paths: Path or list of paths to YAML config files. If None,
                      defaults to configs/models_pt.yaml and configs/models_it.yaml.

    Returns:
        Dictionary mapping model names to (path, type) tuples
    """
    script_dir = Path(__file__).parent.parent

    if config_paths is None:
        config_paths = [
            script_dir / "configs" / "models_pt.yaml",
            script_dir / "configs" / "models_it.yaml",
        ]
    elif isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    model_configs = {}
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        for model_name, model_info in config["models"].items():
            model_configs[model_name] = (
                model_info["path"],
                model_info["type"],
                model_info.get("tokenizer", model_info["path"]),
                model_info.get("thinking", False),
            )

    return model_configs


# Load model configurations from YAML
MODEL_CONFIGS = load_model_configs()

# Single source of truth for benchmark format ("mcq" or "gen").
# Used both in evaluate_benchmark (format detection) and main() (mode filtering).
BENCHMARK_FORMATS = {
    "pacute": "mcq",
    "pacute-mcq": "mcq",
    "pacute-gen": "gen",
    "pacute-affixation": "mcq",
    "pacute-affixation-mcq": "mcq",
    "pacute-affixation-gen": "gen",
    "pacute-composition": "mcq",
    "pacute-composition-mcq": "mcq",
    "pacute-composition-gen": "gen",
    "pacute-manipulation": "mcq",
    "pacute-manipulation-mcq": "mcq",
    "pacute-manipulation-gen": "gen",
    "pacute-syllabification": "mcq",
    "pacute-syllabification-mcq": "mcq",
    "pacute-syllabification-gen": "gen",
    "cute": "gen",
    "cute-gen": "gen",
    "hierarchical": "mcq",
    "hierarchical-mcq": "mcq",
    "hierarchical-gen": "gen",
    "langgame": "mcq",
    "langgame-mcq": "mcq",
    "langgame-gen": "gen",
    "multi-digit-addition": "mcq",
    "multi-digit-addition-mcq": "mcq",
    "multi-digit-addition-gen": "gen",
}


class VLLMEvaluator:
    """Evaluator that talks to a running vLLM server via the OpenAI-compatible API."""

    def __init__(
        self,
        model_name,
        model_id,
        model_type,
        tokenizer_name,
        thinking=False,
        vllm_url="http://localhost:8000",
        vllm_api_key="token-abc123",
        vllm_model_id=None,
        system_prompt=None,
        benchmark_system_prompts=None,
        benchmark_answer_tags=None,
    ):
        """
        Initialize evaluator pointing at a running vLLM server.

        Args:
            model_name: Short name for logging.
            model_id: HuggingFace model ID that the vLLM server was started with
                (e.g. ``vllm serve Qwen/Qwen2.5-7B-Instruct``).
            model_type: ``"pt"`` (pretrained) or ``"it"`` (instruction-tuned).
                Controls whether chat completions or plain completions are used
                for generative evaluation.
            tokenizer_name: HuggingFace tokenizer to load locally (CPU-only) for
                token counting during MCQ log-prob scoring.
            vllm_url: Base URL of the vLLM server (default: ``"http://localhost:8000"``).
            vllm_api_key: API key sent to the vLLM server (default: ``"token-abc123"``).
            system_prompt: Optional global system prompt for generative evaluation.
                Overrides ``benchmark_system_prompts`` when set.
            benchmark_system_prompts: Optional dict mapping benchmark name to
                instruction string. Loaded from ``configs/evaluation.yaml`` by default.
            benchmark_answer_tags: Optional dict mapping benchmark name to the
                answer-prefix string (e.g. ``"Answer:"``). Used to strip the
                prefix from generated text before comparison.
            thinking: Whether this model uses chain-of-thought / reasoning mode.
                When True, generation uses 8192 max tokens (vs 256) and passes
                ``enable_thinking=True`` via the vLLM chat-template kwargs so that
                the model's native thinking toggle is respected. The ``<think>``
                block is extracted and stored as ``thinking_trace`` in results.
        """
        self.model_name = model_name
        self.model_id = vllm_model_id if vllm_model_id else model_id
        self.model_type = model_type
        self.thinking = thinking
        self.system_prompt = system_prompt
        self.benchmark_system_prompts = benchmark_system_prompts or {}
        self.benchmark_answer_tags = benchmark_answer_tags or {}

        self.client = AsyncOpenAI(
            base_url=f"{vllm_url.rstrip('/')}/v1",
            api_key=vllm_api_key,
        )

        # Load tokenizer locally (CPU-only) for token counting in MCQ logprob scoring.
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        thinking_label = "thinking=ON (max_new_tokens=8192)" if thinking else "thinking=OFF (max_new_tokens=256)"
        if vllm_model_id and vllm_model_id != model_id:
            print(f"Connected to vLLM server at {vllm_url} (registered as: {vllm_model_id}, HF path: {model_id}, {thinking_label})")
        else:
            print(f"Connected to vLLM server at {vllm_url} (model: {self.model_id}, {thinking_label})")

    # -------------------------------------------------------------------------
    # Low-level async helpers
    # -------------------------------------------------------------------------

    async def _compute_logprob(self, prefix: str, option: str) -> float:
        """Score log P(option | prefix) via the vLLM completions endpoint."""
        prefix = str(prefix)
        option = str(option)
        full_text = prefix + " " + option

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        n_prefix = len(prefix_tokens)
        n_full = len(full_tokens)

        if n_full <= n_prefix:
            return -100.0

        response = await self.client.completions.create(
            model=self.model_id,
            prompt=full_text,
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )

        # With echo=True, token_logprobs covers all prompt tokens + 1 generated.
        # token_logprobs[0] is always None.  Option tokens occupy indices [n_prefix, n_full).
        all_token_lps = response.choices[0].logprobs.token_logprobs
        option_token_lps = [lp for lp in all_token_lps[n_prefix:n_full] if lp is not None]
        return sum(option_token_lps) if option_token_lps else -100.0

    async def _score_options(self, prefix: str, ground_truth, false_options):
        """Score all MCQ options in parallel and return a log-prob tensor."""
        all_options = [ground_truth] + list(false_options)
        tasks = [self._compute_logprob(prefix, str(opt)) for opt in all_options]
        logprobs = await asyncio.gather(*tasks)
        return torch.tensor(list(logprobs))

    async def _generate(
        self, prefix: str, ground_truth, max_new_tokens: int = None, system_prompt=None
    ) -> dict:
        """Generate text for one item via chat completions (IT) or completions (PT)."""
        prefix = str(prefix)
        ground_truth = str(ground_truth).strip().lower()

        # Default token budget: thinking models need room for the <think> trace.
        if max_new_tokens is None:
            max_new_tokens = 8192 if self.thinking else 256

        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt

        if self.model_type == "it":
            messages = []
            if effective_prompt:
                messages.append({"role": "system", "content": effective_prompt})
            messages.append({"role": "user", "content": prefix})
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                # Pass enable_thinking so vLLM honours the model's thinking toggle.
                # Silently ignored for models that don't support it.
                extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
            )
            content = response.choices[0].message.content
            generated_answer = content.strip().lower() if content is not None else ""
        else:
            # PT model: plain text completion
            prompt = (effective_prompt + "\n\n" + prefix) if effective_prompt else prefix
            response = await self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            text = response.choices[0].text
            generated_answer = text.strip().lower() if text is not None else ""

        return {
            "generated": generated_answer,
            "ground_truth": ground_truth,
            "exact_match": generated_answer == ground_truth,
            "contains_match": ground_truth in generated_answer,
            "prefix_match": generated_answer.startswith(ground_truth),
        }

    # -------------------------------------------------------------------------
    # Benchmark evaluation
    # -------------------------------------------------------------------------

    def evaluate_benchmark(
        self, benchmark_name, max_samples=None, check_existing=True, timestamp=None
    ):
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

        try:
            benchmark_loader = load_benchmark(benchmark_name)
        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            return None

        benchmark_items = []

        # Determine format from the BENCHMARK_FORMATS registry.
        is_generative = BENCHMARK_FORMATS.get(benchmark_name, "mcq") == "gen"
        format_type = "generative" if is_generative else "MCQ"
        print(f"Detected {format_type} format for {benchmark_name}")

        for i, item in enumerate(benchmark_loader):
            benchmark_items.append(item)
            if max_samples and i >= max_samples - 1:
                break

        if len(benchmark_items) == 0:
            print(f"No samples loaded for {benchmark_name}")
            return None

        setting = "gen" if is_generative else "mcq"

        if check_existing:
            inference_dir = os.path.join("results", self.model_name, "inference")
            inference_file = os.path.join(inference_dir, f"{benchmark_name}.jsonl")
            if os.path.exists(inference_file):
                print(f"\n{'='*80}")
                print(f"⏭️  SKIPPING: {benchmark_name}")
                print(f"{'='*80}")
                print(f"Reason: Inference results already exist")
                print(f"File: {inference_file}")
                print(f"Note: Use --overwrite flag to re-run evaluation")
                print(f"{'='*80}\n")
                return {"skipped": True, "inference_file": inference_file}

        if is_generative:
            return self._evaluate_generative_benchmark(
                benchmark_items, benchmark_name, setting=setting, timestamp=timestamp
            )
        else:
            return self._evaluate_mcq_benchmark(
                benchmark_items, benchmark_name, setting=setting, timestamp=timestamp
            )

    def _evaluate_mcq_benchmark(
        self, benchmark_items, benchmark_name, setting=None, timestamp=None
    ):
        """Evaluate MCQ format benchmark. Scores all options per question in parallel."""
        return asyncio.run(
            self._async_mcq_benchmark(benchmark_items, benchmark_name, setting, timestamp)
        )

    async def _async_mcq_benchmark(
        self, benchmark_items, benchmark_name, setting=None, timestamp=None
    ):
        # Limit to 16 concurrent items; each item scores ~4 options in parallel,
        # giving ~64 in-flight requests — same ceiling as the gen path.
        sem = asyncio.Semaphore(16)
        pbar = tqdm(total=len(benchmark_items), desc=benchmark_name)

        async def score_item(item):
            prefix, ground_truth, false_options, sample_id = item[:4]
            category = item[4] if len(item) > 4 else None
            async with sem:
                logprobs = await self._score_options(prefix, ground_truth, false_options)
            pbar.update(1)

            predicted_idx = torch.argmax(logprobs).item()
            all_options = [ground_truth] + list(false_options)
            return {
                "logprobs": logprobs,
                "detail": {
                    "id": sample_id,
                    "category": category,
                    "question": prefix,
                    "ground_truth": ground_truth,
                    "options": all_options,
                    "predicted_idx": predicted_idx,
                    "predicted_answer": (
                        all_options[predicted_idx] if predicted_idx < len(all_options) else None
                    ),
                    "is_correct": predicted_idx == 0,
                    "logprobs": logprobs.tolist(),
                },
            }

        scored = await asyncio.gather(*[score_item(item) for item in benchmark_items])
        pbar.close()

        confidences = [s["logprobs"] for s in scored]
        detailed_results = [s["detail"] for s in scored]
        total_count = len(scored)

        if total_count == 0:
            print(f"No samples evaluated for {benchmark_name}")
            return None

        max_length = max(len(c) for c in confidences)
        padded = [F.pad(c, (0, max_length - len(c)), value=-1e10) for c in confidences]
        confidences_tensor = torch.stack(padded)

        results = self.calculate_metrics(confidences_tensor)
        results["num_samples"] = total_count
        results["format"] = "mcq"
        results["by_category"] = self._by_category_mcq(detailed_results)
        results["detailed_results"] = detailed_results
        results["setting"] = setting
        results["timestamp"] = timestamp
        return results

    def _evaluate_generative_benchmark(
        self, benchmark_items, benchmark_name, setting=None, timestamp=None
    ):
        """Evaluate generative benchmark. Sends all requests concurrently."""
        return asyncio.run(
            self._async_generative_benchmark(benchmark_items, benchmark_name, setting, timestamp)
        )

    async def _async_generative_benchmark(
        self, benchmark_items, benchmark_name, setting=None, timestamp=None
    ):
        benchmark_prompt = self.benchmark_system_prompts.get(benchmark_name)
        effective_prompt = self.system_prompt if self.system_prompt is not None else benchmark_prompt
        if effective_prompt is not None:
            source = "CLI override" if self.system_prompt is not None else "evaluation config"
            print(f"  Using instruction for {benchmark_name} (source: {source})")

        answer_tag = self.benchmark_answer_tags.get(benchmark_name)

        # Fire generation requests concurrently, limited to 64 in-flight at once
        # to avoid overwhelming vLLM and receiving empty/None responses.
        sem = asyncio.Semaphore(64)

        async def generate_one(prefix, ground_truth):
            async with sem:
                return await self._generate(str(prefix), ground_truth, system_prompt=effective_prompt)

        tasks = [
            generate_one(item[0], item[1])
            for item in benchmark_items
        ]
        print(f"  Sending {len(tasks)} generation requests (max 64 concurrent)...")
        api_responses = await asyncio.gather(*tasks)

        exact_matches = 0
        contains_matches = 0
        prefix_matches = 0
        detailed_results = []

        for item_data, api_response in zip(benchmark_items, api_responses):
            prefix, ground_truth, _, sample_id = item_data[:4]
            category = item_data[4] if len(item_data) > 4 else None

            # Full model output string, preserved verbatim for inspection.
            response = api_response["generated"]

            # Extract <think> block if present (chain-of-thought / reasoning models).
            thinking_trace = None
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking_trace = think_match.group(1).strip()
            # Strip the <think> block so it doesn't pollute answer extraction.
            response_for_answer = (
                re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
                if thinking_trace is not None
                else response
            )

            # Extract <reflection> block if present.
            reflection = None
            reflection_match = re.search(
                r"<reflection>(.*?)</reflection>", response_for_answer, re.DOTALL | re.IGNORECASE
            )
            if reflection_match:
                reflection = reflection_match.group(1).strip()

            # Extract the answer: strip answer-tag and take the first non-empty line.
            if answer_tag and answer_tag.lower() in response_for_answer:
                _, _, after_tag = response_for_answer.partition(answer_tag.lower())
                # Strip the closing XML tag (e.g. </answer>) if present.
                closing_tag = answer_tag.replace("<", "</").rstrip(">") + ">"
                if closing_tag in after_tag:
                    after_tag = after_tag.split(closing_tag)[0]
                lines = after_tag.strip().splitlines()
                answer = lines[0].strip() if lines else ""
            elif "\n" in response_for_answer:
                lines = response_for_answer.splitlines()
                answer = lines[0].strip() if lines else response_for_answer
            else:
                answer = response_for_answer

            expected = str(ground_truth).strip().lower()
            is_exact = answer == expected
            is_contains = expected in answer
            is_prefix = answer.startswith(expected)

            if is_exact:
                exact_matches += 1
            if is_contains:
                contains_matches += 1
            if is_prefix:
                prefix_matches += 1

            detailed_results.append({
                "id": sample_id,
                "category": category,
                "question": prefix,
                "ground_truth": ground_truth,
                "response": response,
                "thinking_trace": thinking_trace,
                "reflection": reflection,
                "answer": answer,
                "exact_match": is_exact,
                "contains_match": is_contains,
                "prefix_match": is_prefix,
            })

        total_count = len(detailed_results)
        if total_count == 0:
            print(f"No samples evaluated for {benchmark_name}")
            return None

        return {
            "num_samples": total_count,
            "exact_match": exact_matches / total_count,
            "contains_match": contains_matches / total_count,
            "prefix_match": prefix_matches / total_count,
            "format": "generative",
            "by_category": self._by_category_gen(detailed_results),
            "detailed_results": detailed_results,
            "setting": setting,
            "timestamp": timestamp,
        }

    @staticmethod
    def _group_by_category(results):
        """Partition a list of per-sample result dicts by their 'category' field."""
        groups = {}
        for r in results:
            cat = r.get("category") or "__all__"
            groups.setdefault(cat, []).append(r)
        return groups

    @staticmethod
    def _by_category_mcq(detailed_results):
        """Group MCQ detailed results by category and compute per-category metrics."""
        out = {}
        for cat, items in sorted(VLLMEvaluator._group_by_category(detailed_results).items()):
            n = len(items)
            correct = sum(1 for r in items if r["is_correct"])
            acc = correct / n
            # FP=0 in single-answer MCQ, so precision=1.0 and recall=accuracy
            tp = correct
            fn = n - correct
            precision = tp / (tp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            # Path confidence: mean softmax prob of the ground-truth option (index 0)
            path_conf_vals = []
            for r in items:
                lps = r.get("logprobs")
                if lps:
                    t = torch.tensor(lps, dtype=torch.float32)
                    path_conf_vals.append(F.softmax(t, dim=0)[0].item())
            path_confidence = sum(path_conf_vals) / len(path_conf_vals) if path_conf_vals else 0.0
            num_opts = len(items[0]["options"]) if items else 4
            norm_acc = (acc * num_opts - 1) / (num_opts - 1)
            out[cat] = {
                "num_samples": n,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "path_confidence": round(path_confidence, 4),
                "normalized_accuracy": round(norm_acc, 4),
            }
        return out

    @staticmethod
    def _by_category_gen(detailed_results):
        """Group generative detailed results by category and compute per-category metrics."""
        out = {}
        for cat, items in sorted(VLLMEvaluator._group_by_category(detailed_results).items()):
            n = len(items)
            out[cat] = {
                "num_samples": n,
                "exact_match": round(sum(r["exact_match"] for r in items) / n, 4),
                "contains_match": round(sum(r["contains_match"] for r in items) / n, 4),
                "prefix_match": round(sum(r["prefix_match"] for r in items) / n, 4),
            }
        return out

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
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "path_confidence": path_confidence,
            "normalized_accuracy": normalized_accuracy,
            "num_options": num_options,
        }


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation on multiple models")
    parser.add_argument(
        "--model-config",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to model configuration YAML file(s) (default: configs/models_pt.yaml + configs/models_it.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        help="Models to evaluate (must be defined in model config)",
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
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum samples per benchmark (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="auto",
        choices=["auto", "mcq", "gen", "both"],
        help=(
            "Evaluation mode. 'auto' (default): MCQ-only for pretrained (pt) models, "
            "both MCQ and generative for instruction-tuned (it) models. "
            "'mcq': MCQ only for all models. "
            "'gen': generative only for all models. "
            "'both': both modes for all models regardless of type."
        ),
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help=(
            "Path to evaluation configuration YAML (default: configs/evaluation.yaml). "
            "Contains per-benchmark system prompts under 'generative_system_prompts'."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Global system prompt that overrides all per-benchmark prompts from the "
            "evaluation config. Applied via the tokenizer's chat template when available "
            "(instruction-tuned models), otherwise prepended as plain text. "
            "Has no effect on MCQ (log-prob) evaluation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference results (default: False, skip if results exist)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default="token-abc123",
        help="API key for the vLLM server (default: token-abc123)",
    )
    parser.add_argument(
        "--vllm-model-id",
        type=str,
        default=None,
        help=(
            "Override the model ID used in API calls. When set, this is used instead of "
            "the HuggingFace path. Useful when vLLM registers the model under a different "
            "name (e.g. 'openai-community/gpt2' instead of 'gpt2')."
        ),
    )

    args = parser.parse_args()

    # Load evaluation configuration (per-benchmark system prompts, etc.)
    eval_config_path = args.eval_config
    if eval_config_path is None:
        default_eval_config = Path(__file__).parent.parent / "configs" / "evaluation.yaml"
        if default_eval_config.exists():
            eval_config_path = str(default_eval_config)

    benchmark_system_prompts = {}
    benchmark_answer_tags = {}
    if eval_config_path:
        with open(eval_config_path, "r") as f:
            eval_config = yaml.safe_load(f)
        raw_instructions = eval_config.get("generative_instructions", {})
        for bench, entry in raw_instructions.items():
            if isinstance(entry, dict):
                benchmark_system_prompts[bench] = entry.get("instruction", "")
                if "answer_tag" in entry:
                    benchmark_answer_tags[bench] = entry["answer_tag"]
            else:
                # Plain string fallback (legacy support)
                benchmark_system_prompts[bench] = entry
        print(f"Loaded evaluation config from: {eval_config_path}")
        print(f"  Instructions defined for: {list(benchmark_system_prompts.keys())}")

    # Load model configurations (reload if custom config specified)
    if args.model_config:
        global MODEL_CONFIGS
        MODEL_CONFIGS = load_model_configs(args.model_config)  # accepts None or list of paths
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

    def filter_benchmarks(benchmarks, mode):
        """Return only the benchmarks that match the requested eval mode."""
        kept = []
        for b in benchmarks:
            fmt = BENCHMARK_FORMATS.get(b, "both")
            if mode == "both" or fmt == "both" or fmt == mode:
                kept.append(b)
            else:
                print(f"⚠️  Skipping {b} ({fmt}-only benchmark, effective mode={mode})")
        return kept

    print(f"\n{'='*80}")
    print(f"Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Eval mode: {args.eval_mode} (auto = MCQ-only for pt, MCQ+gen for it)")
    print(f"Overwrite existing results: {args.overwrite}")
    print(f"{'='*80}")

    # Run evaluations
    all_results = {}

    for model_name in args.models:
        hf_model_name, model_type, tokenizer_name, thinking = MODEL_CONFIGS[model_name]

        # Determine effective eval mode for this model.
        # 'auto' uses model type; an explicit CLI flag overrides for all models.
        if args.eval_mode == "auto":
            effective_mode = "mcq" if model_type == "pt" else "both"
        else:
            effective_mode = args.eval_mode

        benchmarks_for_model = filter_benchmarks(args.benchmarks, effective_mode)

        thinking_label = "thinking=ON" if thinking else "thinking=OFF"
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}  (type: {model_type}, mode: {effective_mode}, {thinking_label})")
        print(f"Benchmarks: {', '.join(benchmarks_for_model)}")
        print(f"{'='*80}")

        if not benchmarks_for_model:
            print(f"No benchmarks to run for {model_name} — skipping.")
            continue

        try:
            # Initialize evaluator
            evaluator = VLLMEvaluator(
                model_name=model_name,
                model_id=hf_model_name,
                model_type=model_type,
                tokenizer_name=tokenizer_name,
                thinking=thinking,
                vllm_url=args.vllm_url,
                vllm_api_key=args.vllm_api_key,
                vllm_model_id=args.vllm_model_id,
                system_prompt=args.system_prompt,
                benchmark_system_prompts=benchmark_system_prompts,
                benchmark_answer_tags=benchmark_answer_tags,
            )

            # Evaluate on each benchmark
            model_results = {}
            for benchmark_name in benchmarks_for_model:
                try:
                    results = evaluator.evaluate_benchmark(
                        benchmark_name=benchmark_name,
                        max_samples=args.max_samples,
                        check_existing=(not args.overwrite),
                        timestamp=timestamp,
                    )
                except Exception as bench_err:
                    print(f"Error on benchmark {benchmark_name}: {bench_err}")
                    import traceback as _tb; _tb.print_exc()
                    continue

                if results:
                    # Check if evaluation was skipped
                    if results.get("skipped"):
                        continue

                    # Save detailed inference results
                    detailed_results = results.pop("detailed_results", None)
                    results.pop("setting", None)
                    if detailed_results:
                        inference_dir = os.path.join("results", model_name, "inference")
                        os.makedirs(inference_dir, exist_ok=True)
                        inference_file = os.path.join(inference_dir, f"{benchmark_name}.jsonl")
                        detailed_results.sort(key=lambda r: r.get("id", ""))
                        with open(inference_file, "w", encoding="utf-8") as f:
                            for result in detailed_results:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        print(f"Saved detailed inference results to: {inference_file}")

                    model_results[benchmark_name] = results

                    # Print results based on format
                    print(f"\n{benchmark_name} Results:")
                    print(f"  Samples: {results['num_samples']}")

                    if results.get("format") == "generative":
                        print(f"  Exact Match: {results['exact_match']:.4f}")
                        print(f"  Contains Match: {results['contains_match']:.4f}")
                        print(f"  Prefix Match: {results['prefix_match']:.4f}")
                    else:
                        print(f"  Accuracy: {results['accuracy']:.4f}")
                        print(f"  F1 Score: {results['f1_score']:.4f}")
                        print(f"  Precision: {results['precision']:.4f}")
                        print(f"  Recall: {results['recall']:.4f}")
                        print(f"  Path Confidence: {results['path_confidence']:.4f}")
                        print(f"  Normalized Accuracy: {results['normalized_accuracy']:.4f}")

            all_results[model_name] = {
                "hf_model_name": hf_model_name,
                "model_type": model_type,
                "benchmarks": model_results,
            }

            # Save results for this model
            model_output_dir = os.path.join("results", model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            model_output_file = os.path.join(
                model_output_dir, f"evaluation_results_{timestamp}.json"
            )

            model_result = {
                "hf_model_name": hf_model_name,
                "model_type": model_type,
                "benchmarks": model_results,
                "timestamp": timestamp,
            }

            with open(model_output_file, "w", encoding="utf-8") as f:
                json.dump(model_result, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*80}")
            print(f"Results for {model_name} saved to: {model_output_file}")
            print(f"{'='*80}")

            del evaluator

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save combined results (for backwards compatibility)
    if args.output_dir:
        output_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Combined results saved to: {output_file}")
        print(f"{'='*80}")

    # Print summary table
    print("\nSummary Table:")
    print(f"{'Model':<25} {'Benchmark':<20} {'Format':<12} {'Primary Metric':<20}")
    print("-" * 85)

    for model_name, model_data in all_results.items():
        for benchmark_name, results in model_data["benchmarks"].items():
            format_type = results.get("format", "mcq")
            if format_type == "generative":
                metric_str = f"Exact Match: {results['exact_match']:.4f}"
            else:
                metric_str = f"Acc: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}"

            print(f"{model_name:<25} {benchmark_name:<20} " f"{format_type:<12} {metric_str:<20}")


if __name__ == "__main__":
    main()
