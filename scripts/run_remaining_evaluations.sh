#!/bin/bash
# Run full evaluation on NEW models not in the original evaluation script
# (SEA-LION and Qwen3 models that were recently added)

WORKSPACE="/home/ubuntu/filipino-morphology-llm"
cd "$WORKSPACE"
source env/bin/activate

LOG_FILE="results/new_models_evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting NEW model evaluations at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "These models were recently added and not in the original script" | tee -a "$LOG_FILE"

# NEW models that need evaluation (not in run_full_evaluation.sh)
MODELS=(
    # SEA-LION v3 (Southeast Asian - high priority)
    "sea-lion-v3-8b-it"      # Re-run with full benchmarks (only had PACUTE MCQ)
    "sea-lion-v3-8b"
    "sea-lion-gemma-v3-9b"
    "sea-lion-gemma-v3-9b-it"

    # Qwen 3 (newest models)
    "qwen3-4b-it"
    "qwen3-4b-thinking"
)

TOTAL=${#MODELS[@]}
CURRENT=0

for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$CURRENT/$TOTAL] Evaluating: $model" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    python scripts/run_evaluation.py \
        --models "$model" \
        --benchmarks pacute hierarchical cute langgame multi-digit-addition \
        --eval-mode both \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$CURRENT/$TOTAL] $model completed successfully at $(date)" | tee -a "$LOG_FILE"
    else
        echo "[$CURRENT/$TOTAL] $model FAILED with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
    fi

    # Clear GPU memory between models
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 5
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "New model evaluations completed at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
