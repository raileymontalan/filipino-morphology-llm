#!/bin/bash
# Run batch evaluation on all models and benchmarks

# Create output directory
mkdir -p results/benchmark_evaluation

echo "Starting benchmark evaluations..."
echo "================================"

# Small models first (for testing)
echo "Testing with GPT2..."
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute cute \
    --max-samples 100 \
    --output-dir results/benchmark_evaluation

# GPT-2 variants
echo "Running GPT-2 variants..."
python scripts/run_benchmark_evaluation.py \
    --models gpt2 gpt2-medium gpt2-large \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

# Qwen models (smallest first)
echo "Running Qwen models..."
python scripts/run_benchmark_evaluation.py \
    --models qwen-2.5-0.5b qwen-2.5-0.5b-it \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

python scripts/run_benchmark_evaluation.py \
    --models qwen-2.5-1.5b qwen-2.5-1.5b-it \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

# Cerebras GPT models
echo "Running Cerebras GPT models..."
python scripts/run_benchmark_evaluation.py \
    --models cerebras-gpt-111m cerebras-gpt-256m cerebras-gpt-590m \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

# Llama models (if available)
echo "Running Llama models..."
python scripts/run_benchmark_evaluation.py \
    --models llama-3.2-1b llama-3.2-1b-it \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

# Gemma models (if available)
echo "Running Gemma models..."
python scripts/run_benchmark_evaluation.py \
    --models gemma-2b gemma-2b-it \
    --benchmarks pacute cute \
    --output-dir results/benchmark_evaluation

echo "================================"
echo "All evaluations complete!"
echo "Results saved to results/benchmark_evaluation/"
