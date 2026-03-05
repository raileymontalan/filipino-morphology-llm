#!/bin/bash
# Submit one PBS job per model in the model config YAMLs.
#
# Usage:
#   bash scripts/submit_evaluations.sh [OPTIONS]
#
# Options:
#   --pt-only           Only submit PT (base) models
#   --it-only           Only submit IT (instruction-tuned) models
#   --model <name>      Submit only this model (repeatable)
#   --overwrite         Re-run benchmarks that already have results
#   --max-samples <n>   Cap samples per benchmark
#   --port <n>          vLLM server port (default: 8000)
#   --queue <q>         PBS queue (default: AISG_debug)
#   --walltime <hh:mm:ss> Job walltime (default: 12:00:00)
#   --dry-run           Print qsub commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PBS_SCRIPT="$SCRIPT_DIR/eval_model.pbs"

# ── Defaults ─────────────────────────────────────────────────────────────────
PT_ONLY=false
IT_ONLY=false
DRY_RUN=false
OVERWRITE=false
MAX_SAMPLES=""
VLLM_PORT=""          # empty = auto-derive from PBS job ID inside the job
PBS_QUEUE="AISG_debug"
WALLTIME="12:00:00"
SINGLE_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt-only)      PT_ONLY=true ;;
        --it-only)      IT_ONLY=true ;;
        --dry-run)      DRY_RUN=true ;;
        --overwrite)    OVERWRITE=true ;;
        --port)         VLLM_PORT="$2"; shift ;;
        --max-samples)  MAX_SAMPLES="$2"; shift ;;
        --queue)        PBS_QUEUE="$2"; shift ;;
        --walltime)     WALLTIME="$2"; shift ;;
        --model)        SINGLE_MODELS+=("$2"); shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Config files to read ──────────────────────────────────────────────────────
CONFIG_FILES=()
if $IT_ONLY; then
    CONFIG_FILES=("$PROJECT_ROOT/configs/models_it.yaml")
elif $PT_ONLY; then
    CONFIG_FILES=("$PROJECT_ROOT/configs/models_pt.yaml")
else
    CONFIG_FILES=(
        "$PROJECT_ROOT/configs/models_pt.yaml"
        "$PROJECT_ROOT/configs/models_it.yaml"
    )
fi

# ── Build model list ──────────────────────────────────────────────────────────
if [[ ${#SINGLE_MODELS[@]} -gt 0 ]]; then
    ALL_MODELS=("${SINGLE_MODELS[@]}")
else
    ALL_MODELS=()
    for cfg in "${CONFIG_FILES[@]}"; do
        while IFS= read -r model; do
            ALL_MODELS+=("$model")
        done < <(python3 -c "
import yaml
with open('$cfg') as f:
    data = yaml.safe_load(f)
for name in data['models']:
    print(name)
")
    done
fi

# ── Determine GPU count from model path heuristic ────────────────────────────
gpus_for_model() {
    local model_id="$1"
    # Very large / known MoE giants (need all GPUs)
    if   echo "$model_id" | grep -qiE "(120b|122b|397b|685b|deepseek.v3|deepseek.r1)"; then echo 8
    # Large dense (27B – 72B): 4 GPUs
    elif echo "$model_id" | grep -qiE "[-_](27|32|33|35|70|72)[bB]"; then echo 4
    # Medium (7B – 27B): 2 GPUs
    elif echo "$model_id" | grep -qiE "[-_](7|8|9|12|14|20|21)[bB]"; then echo 2
    # Small / unknown: 1 GPU
    else echo 1
    fi
}

# Get model path from YAML for heuristic
get_model_path() {
    local name="$1"
    python3 - <<PYEOF
import yaml, sys
for cfg in ['$PROJECT_ROOT/configs/models_pt.yaml', '$PROJECT_ROOT/configs/models_it.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('$name')
        if info:
            print(info['path'])
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

# ── PBS variable string ───────────────────────────────────────────────────────
pbs_vars() {
    local model="$1"
    local vars="MODEL_NAME=${model}"
    [[ -n "$VLLM_PORT" ]]   && vars="${vars},VLLM_PORT=${VLLM_PORT}"
    $OVERWRITE              && vars="${vars},OVERWRITE=true"
    [[ -n "$MAX_SAMPLES" ]] && vars="${vars},MAX_SAMPLES=${MAX_SAMPLES}"
    echo "$vars"
}

# ── Submit ────────────────────────────────────────────────────────────────────
echo "Submitting ${#ALL_MODELS[@]} model evaluation jobs to queue: $PBS_QUEUE"
echo "PBS script: $PBS_SCRIPT"
echo ""

SUBMITTED=0
SKIPPED=0

for model_name in "${ALL_MODELS[@]}"; do
    model_path=$(get_model_path "$model_name" 2>/dev/null) || {
        echo "  SKIP: '$model_name' not found in config YAMLs"
        SKIPPED=$((SKIPPED + 1))
        continue
    }
    N_GPUS=$(gpus_for_model "$model_path")
    NCPUS=$((N_GPUS * 4))    # 4 CPUs per GPU is a reasonable ratio
    MEM=$((N_GPUS * 64))gb   # 64 GB per GPU

    VARS=$(pbs_vars "$model_name")
    JOB_NAME="fm-eval-$(echo "$model_name" | tr '.' '-')"

    CMD=(
        qsub
        -N "$JOB_NAME"
        -q "$PBS_QUEUE"
        -l "select=1:mem=${MEM}:ncpus=${NCPUS}:ngpus=${N_GPUS}"
        -l "walltime=${WALLTIME}"
        -o "${PROJECT_ROOT}/results/logs/"
        -e "${PROJECT_ROOT}/results/logs/"
        -j oe
        -v "$VARS"
        "$PBS_SCRIPT"
    )

    printf "  %-30s  ngpus=%-2s  " "$model_name" "$N_GPUS"

    if $DRY_RUN; then
        echo "[DRY RUN] ${CMD[*]}"
    else
        JOB_ID=$("${CMD[@]}")
        echo "$JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
echo "Submitted: $SUBMITTED  Skipped: $SKIPPED"
