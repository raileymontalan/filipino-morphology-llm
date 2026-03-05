#!/bin/bash

# ============================================================================
# Convert NeMo/Megatron Checkpoint to HuggingFace and Run Evaluation
# ============================================================================
#
# This script:
#   1. Converts a NeMo/Megatron checkpoint to HuggingFace format
#   2. Adds it to the model config
#   3. Runs evaluation using your existing pipeline
#
# Usage:
#   bash scripts/convert_and_evaluate_checkpoint.sh \
#       --checkpoint /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158931/step=1000.ckpt \
#       --base-model cerebras/Cerebras-GPT-1.3B \
#       --model-name cerebras-1.3b-step1000
#
# Or for all checkpoints in a directory:
#   bash scripts/convert_and_evaluate_checkpoint.sh \
#       --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158931/ \
#       --base-model cerebras/Cerebras-GPT-1.3B \
#       --prefix cerebras-1.3b
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd ${PROJECT_DIR}

# ============================================================================
# Parse Arguments
# ============================================================================

CHECKPOINT=""
CHECKPOINT_DIR=""
BASE_MODEL=""
MODEL_NAME=""
PREFIX=""
SKIP_EVAL=false
BENCHMARKS="pacute-affixation-mcq pacute-composition-mcq hierarchical-mcq langgame-mcq"
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$BASE_MODEL" ]; then
    echo "Error: --base-model is required"
    exit 1
fi

if [ -z "$CHECKPOINT" ] && [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Either --checkpoint or --checkpoint-dir is required"
    exit 1
fi

# ============================================================================
# Helper Functions
# ============================================================================

convert_checkpoint() {
    local ckpt_path=$1
    local base_model=$2
    local output_name=$3
    
    local output_dir="${PROJECT_DIR}/checkpoints/hf/${output_name}"
    
    echo ""
    echo "============================================================================"
    echo "Converting: $(basename $ckpt_path)"
    echo "============================================================================"
    echo "Checkpoint: $ckpt_path"
    echo "Base model: $base_model"
    echo "Output: $output_dir"
    echo ""
    
    # Check if already converted
    if [ -d "$output_dir" ] && [ -f "$output_dir/config.json" ]; then
        echo "⚠ Already converted: $output_dir"
        echo "  Skipping conversion..."
        return 0
    fi
    
    # Run conversion
    python scripts/convert_megatron_to_hf.py \
        --megatron-checkpoint "$ckpt_path" \
        --base-model "$base_model" \
        --output "$output_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ Conversion successful: $output_dir"
        
        # Add to model config if not already present
        add_to_model_config "$output_name" "$output_dir"
        
        return 0
    else
        echo "✗ Conversion failed"
        return 1
    fi
}

add_to_model_config() {
    local model_name=$1
    local model_path=$2
    local config_file="${PROJECT_DIR}/configs/models.yaml"
    
    # Check if model already in config
    if grep -q "^  ${model_name}:" "$config_file" 2>/dev/null; then
        echo "  Model already in config: $model_name"
        return 0
    fi
    
    echo "  Adding to model config: $model_name"
    
    # Append to models.yaml
    cat >> "$config_file" << EOF

  ${model_name}:
    path: ${model_path}
    type: pt
EOF
    
    echo "  ✓ Added to $config_file"
}

extract_step_number() {
    local ckpt_path=$1
    echo "$ckpt_path" | grep -oP 'step=\K\d+' || echo "unknown"
}

# ============================================================================
# Main Logic
# ============================================================================

echo "============================================================================"
echo "Convert and Evaluate NeMo Checkpoints"
echo "============================================================================"

CONVERTED_MODELS=()

# Single checkpoint
if [ -n "$CHECKPOINT" ]; then
    if [ -z "$MODEL_NAME" ]; then
        step=$(extract_step_number "$CHECKPOINT")
        MODEL_NAME="checkpoint-step${step}"
    fi
    
    convert_checkpoint "$CHECKPOINT" "$BASE_MODEL" "$MODEL_NAME"
    CONVERTED_MODELS+=("$MODEL_NAME")

# Directory of checkpoints
elif [ -n "$CHECKPOINT_DIR" ]; then
    if [ -z "$PREFIX" ]; then
        echo "Error: --prefix is required when using --checkpoint-dir"
        exit 1
    fi
    
    # Find all .ckpt files
    shopt -s nullglob
    CKPT_FILES=("$CHECKPOINT_DIR"/*.ckpt)
    
    if [ ${#CKPT_FILES[@]} -eq 0 ]; then
        echo "Error: No .ckpt files found in $CHECKPOINT_DIR"
        exit 1
    fi
    
    echo "Found ${#CKPT_FILES[@]} checkpoint(s)"
    
    # Convert each checkpoint
    for ckpt_file in "${CKPT_FILES[@]}"; do
        step=$(extract_step_number "$ckpt_file")
        model_name="${PREFIX}-step${step}"
        
        if convert_checkpoint "$ckpt_file" "$BASE_MODEL" "$model_name"; then
            CONVERTED_MODELS+=("$model_name")
        fi
    done
fi

# ============================================================================
# Run Evaluation
# ============================================================================

if [ "$SKIP_EVAL" = true ]; then
    echo ""
    echo "============================================================================"
    echo "Skipping evaluation (--skip-eval specified)"
    echo "============================================================================"
    echo ""
    echo "Converted models: ${CONVERTED_MODELS[@]}"
    echo ""
    echo "To evaluate later, run:"
    echo "  ALL_MODELS=\"${CONVERTED_MODELS[*]}\" bash jobs/submit_parallel_evaluation.sh"
    exit 0
fi

if [ ${#CONVERTED_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "✗ No checkpoints converted successfully"
    exit 1
fi

echo ""
echo "============================================================================"
echo "Running Evaluation"
echo "============================================================================"
echo "Models: ${CONVERTED_MODELS[*]}"
echo "Benchmarks: $BENCHMARKS"
[ -n "$MAX_SAMPLES" ] && echo "Max samples: $MAX_SAMPLES"
echo ""

# Export variables for submit_parallel_evaluation.sh
export ALL_MODELS="${CONVERTED_MODELS[*]}"
export BENCHMARKS="$BENCHMARKS"
[ -n "$MAX_SAMPLES" ] && export MAX_SAMPLES="$MAX_SAMPLES"

# Submit evaluation jobs
bash jobs/submit_parallel_evaluation.sh

echo ""
echo "============================================================================"
echo "✓ Conversion and evaluation jobs submitted"
echo "============================================================================"
