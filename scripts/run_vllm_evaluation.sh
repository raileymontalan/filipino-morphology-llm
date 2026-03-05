#!/bin/bash
# Run evaluation for every model in configs/models_pt.yaml and configs/models_it.yaml.
#
# For each model this script:
#   1. Starts a vLLM server (OpenAI-compatible API)
#   2. Waits for the /health endpoint to respond
#   3. Runs run_evaluation.py against that server
#   4. Stops the server and moves on
#
# Designed for a PBS/qsub interactive job with multiple GPUs.
# Fixes the PBS UUID CUDA_VISIBLE_DEVICES issue automatically.
#
# Usage:
#   bash scripts/run_vllm_evaluation.sh [OPTIONS]
#
# Options:
#   --pt-only        Evaluate only pretrained (PT) models
#   --it-only        Evaluate only instruction-tuned (IT) models
#   --model <name>   Evaluate a single named model (can repeat)
#   --overwrite      Pass --overwrite to run_evaluation.py (re-run existing results)
#   --dry-run        Print what would be done without running anything
#   --port <n>       vLLM server port (default: 8000)
#   --max-samples <n> Limit samples per benchmark (default: all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Source environment (HF_TOKEN, etc.)
# ---------------------------------------------------------------------------
if [[ -f ".env" ]]; then
    source .env
fi
source env/bin/activate

# ---------------------------------------------------------------------------
# Fix CUDA_VISIBLE_DEVICES: PBS/qsub sets UUID strings, vLLM needs integers
# ---------------------------------------------------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
    export CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(map(str,range($N_GPUS))))")
    echo "Remapped CUDA_VISIBLE_DEVICES → $CUDA_VISIBLE_DEVICES  ($N_GPUS GPUs)"
fi
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l | tr -d ' ')

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PT_ONLY=false
IT_ONLY=false
DRY_RUN=false
OVERWRITE_FLAG=""
VLLM_PORT=8000
MAX_SAMPLES_FLAG=""
SINGLE_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt-only)      PT_ONLY=true ;;
        --it-only)      IT_ONLY=true ;;
        --dry-run)      DRY_RUN=true ;;
        --overwrite)    OVERWRITE_FLAG="--overwrite" ;;
        --port)         VLLM_PORT="$2"; shift ;;
        --max-samples)  MAX_SAMPLES_FLAG="--max-samples $2"; shift ;;
        --model)        SINGLE_MODELS+=("$2"); shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

VLLM_URL="http://localhost:${VLLM_PORT}"

# ---------------------------------------------------------------------------
# Determine which config files to read
# ---------------------------------------------------------------------------
CONFIG_FILES=()
if $IT_ONLY; then
    CONFIG_FILES=("configs/models_it.yaml")
elif $PT_ONLY; then
    CONFIG_FILES=("configs/models_pt.yaml")
else
    CONFIG_FILES=("configs/models_pt.yaml" "configs/models_it.yaml")
fi

# ---------------------------------------------------------------------------
# Build model list (from YAML or --model overrides)
# ---------------------------------------------------------------------------
if [[ ${#SINGLE_MODELS[@]} -gt 0 ]]; then
    ALL_MODELS=("${SINGLE_MODELS[@]}")
else
    ALL_MODELS=()
    for cfg in "${CONFIG_FILES[@]}"; do
        while IFS= read -r model; do
            ALL_MODELS+=("$model")
        done < <(python3 -c "
import yaml
with open('${cfg}') as f:
    data = yaml.safe_load(f)
for name in data['models']:
    print(name)
")
    done
fi

# ---------------------------------------------------------------------------
# Set up logging
# ---------------------------------------------------------------------------
LOG_DIR="results/vllm_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

echo "================================================================" | tee -a "$LOG_FILE"
echo "vLLM Evaluation Run — $(date)" | tee -a "$LOG_FILE"
echo "GPUs available: $N_GPUS   Port: $VLLM_PORT" | tee -a "$LOG_FILE"
echo "Config files: ${CONFIG_FILES[*]}" | tee -a "$LOG_FILE"
echo "Models to evaluate (${#ALL_MODELS[@]}):" | tee -a "$LOG_FILE"
for m in "${ALL_MODELS[@]}"; do echo "  $m" | tee -a "$LOG_FILE"; done
echo "================================================================" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Helper: pick tensor-parallel size from model path heuristic
# ---------------------------------------------------------------------------
tp_size_for_model() {
    local model_id="$1"
    # 120B+ (GPT-OSS 120b, Qwen 122B, 397B): all GPUs
    if echo "$model_id" | grep -qiE "120b|122b|397b|685b"; then
        echo "$N_GPUS"
    # 27B–72B (Gemma 27B, SEA-LION 27B/32B, Qwen 72B): 4+ GPUs
    elif echo "$model_id" | grep -qiE "27b|27B|32b|33b|72b|70b"; then
        echo $(( N_GPUS >= 4 ? 4 : N_GPUS ))
    # 7B–14B: 2 GPUs
    elif echo "$model_id" | grep -qiE "7b|7B|8b|8B|9b|9B|12b|12B|14b|14B"; then
        echo $(( N_GPUS >= 2 ? 2 : 1 ))
    # Small / unknown: 1 GPU
    else
        echo 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: wait for vLLM /health to respond
# ---------------------------------------------------------------------------
wait_for_server() {
    local url="$1"
    local timeout=600
    local elapsed=0
    echo -n "  Waiting for vLLM server" | tee -a "$LOG_FILE"
    while ! curl -sf "${url}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "." | tee -a "$LOG_FILE"
        if [[ $elapsed -ge $timeout ]]; then
            echo " TIMEOUT after ${timeout}s" | tee -a "$LOG_FILE"
            return 1
        fi
    done
    echo " ready (${elapsed}s)" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Cleanup: stop vLLM server on exit / signal
# ---------------------------------------------------------------------------
VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID $VLLM_PID)..." | tee -a "$LOG_FILE"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        sleep 3
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Helper: get model path + type from YAMLs
# ---------------------------------------------------------------------------
get_model_info() {
    local name="$1"
    python3 - <<PYEOF
import yaml, sys
for cfg in ['configs/models_pt.yaml', 'configs/models_it.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('${name}')
        if info:
            print(info['path'], info['type'])
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

for model_name in "${ALL_MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "Model: $model_name  [$(date)]" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    # Get model path and type
    model_info=$(get_model_info "$model_name") || {
        echo "  ERROR: '$model_name' not found in any config YAML — skipping" | tee -a "$LOG_FILE"
        FAIL=$((FAIL + 1))
        continue
    }
    model_path=$(echo "$model_info" | awk '{print $1}')
    model_type=$(echo "$model_info" | awk '{print $2}')
    TP=$(tp_size_for_model "$model_path")

    echo "  Path: $model_path" | tee -a "$LOG_FILE"
    echo "  Type: $model_type   TP: ${TP}x" | tee -a "$LOG_FILE"

    if $DRY_RUN; then
        echo "  [DRY RUN] vllm serve $model_path --tensor-parallel-size $TP --port $VLLM_PORT" | tee -a "$LOG_FILE"
        echo "  [DRY RUN] python scripts/run_evaluation.py --models $model_name --vllm-url $VLLM_URL" | tee -a "$LOG_FILE"
        continue
    fi

    # Start vLLM server
    VLLM_LOG="$LOG_DIR/vllm_${model_name}.log"
    echo "  Starting vLLM server → $(basename "$VLLM_LOG")" | tee -a "$LOG_FILE"
    vllm serve "$model_path" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP" \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_server "$VLLM_URL"; then
        echo "  ERROR: server failed to start — see $(basename "$VLLM_LOG")" | tee -a "$LOG_FILE"
        cleanup
        FAIL=$((FAIL + 1))
        continue
    fi

    # Run evaluation
    EVAL_LOG="$LOG_DIR/eval_${model_name}.log"
    echo "  Running evaluation → $(basename "$EVAL_LOG")" | tee -a "$LOG_FILE"

    # Build config file flags
    cfg_flags=""
    for cfg in "${CONFIG_FILES[@]}"; do cfg_flags="$cfg_flags $cfg"; done

    if python3 scripts/run_evaluation.py \
        --models "$model_name" \
        --model-config $cfg_flags \
        --vllm-url "$VLLM_URL" \
        ${OVERWRITE_FLAG} \
        ${MAX_SAMPLES_FLAG} \
        > "$EVAL_LOG" 2>&1; then
        echo "  ✓  PASSED" | tee -a "$LOG_FILE"
        PASS=$((PASS + 1))
    else
        echo "  ✗  FAILED — see $(basename "$EVAL_LOG")" | tee -a "$LOG_FILE"
        FAIL=$((FAIL + 1))
    fi

    cleanup
    sleep 5  # brief cooldown between models
done

echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "All done: $PASS passed, $FAIL failed" | tee -a "$LOG_FILE"
echo "Logs: $LOG_DIR" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
