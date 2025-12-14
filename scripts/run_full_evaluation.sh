#!/bin/bash
# Comprehensive evaluation of all models on all benchmarks
# This script runs evaluations sequentially to avoid GPU memory issues

set -e

# Get the directory where this script is located and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source env/bin/activate

# All benchmarks
BENCHMARKS="pacute hierarchical cute langgame multi-digit-addition"

# Log file
LOG_FILE="results/full_evaluation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "Starting comprehensive evaluation at $(date)" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

# Function to run evaluation for a model
run_eval() {
    local model=$1
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Evaluating: $model at $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    python scripts/run_evaluation.py \
        --models "$model" \
        --benchmarks $BENCHMARKS \
        --eval-mode both \
        2>&1 | tee -a "$LOG_FILE"

    echo "Completed: $model at $(date)" | tee -a "$LOG_FILE"
}

# GPT-2 models (all PT)
echo "=== GPT-2 Family ===" | tee -a "$LOG_FILE"
for model in gpt2 gpt2-medium gpt2-large gpt2-xl; do
    run_eval "$model"
done

# Cerebras GPT-OSS models (all PT)
echo "=== Cerebras GPT-OSS Family ===" | tee -a "$LOG_FILE"
for model in cerebras-gpt-111m cerebras-gpt-256m cerebras-gpt-590m cerebras-gpt-1.3b cerebras-gpt-2.7b; do
    run_eval "$model"
done

# Qwen 2.5 models (PT and IT)
echo "=== Qwen 2.5 Family ===" | tee -a "$LOG_FILE"
for model in qwen-2.5-0.5b qwen-2.5-0.5b-it qwen-2.5-1.5b qwen-2.5-1.5b-it qwen-2.5-3b qwen-2.5-3b-it qwen-2.5-7b qwen-2.5-7b-it qwen-2.5-14b qwen-2.5-14b-it; do
    run_eval "$model"
done

# Llama models (PT and IT)
echo "=== Llama Family ===" | tee -a "$LOG_FILE"
for model in llama-3.2-1b llama-3.2-1b-it llama-3.2-3b llama-3.2-3b-it llama-3.1-8b llama-3.1-8b-it; do
    run_eval "$model"
done

# Gemma models (PT and IT)
echo "=== Gemma Family ===" | tee -a "$LOG_FILE"
for model in gemma-2b gemma-2b-it gemma-7b gemma-7b-it gemma-2-2b gemma-2-2b-it gemma-2-9b gemma-2-9b-it; do
    run_eval "$model"
done

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "Comprehensive evaluation completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved in results/ directory" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
