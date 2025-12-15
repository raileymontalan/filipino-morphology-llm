#!/bin/bash
# Continued Pretraining Script for Gemma 2 2B with different tokenizations
# Usage: ./scripts/run_cpt_training.sh [vanilla|stochastok|patok] [max_steps] [test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load environment
source .env

# Parse arguments
TOKENIZATION="${1:-vanilla}"
MAX_STEPS="${2:-5000}"
TEST_MODE="${3:-}"  # Set to "test" for 100-step test run

echo "=============================================="
echo "Gemma 2 2B Continued Pretraining (NeMo 2.3.0rc0)"
echo "=============================================="
echo "Tokenization: $TOKENIZATION"
echo "Max Steps: $MAX_STEPS"
echo "=============================================="

# Select data directory based on tokenization
case "$TOKENIZATION" in
    vanilla)
        DATA_DIR="/workspace/data/processed/vanilla"
        FILE_SUFFIX="_text_document"  # vanilla uses _text_document suffix
        WANDB_NAME="gemma2-2b-vanilla-${MAX_STEPS}steps"
        CHECKPOINT_DIR="/logs/checkpoints/gemma2-2b-vanilla"
        ;;
    stochastok)
        DATA_DIR="/workspace/data/processed/stochastok"
        FILE_SUFFIX=""  # stochastok has no suffix
        WANDB_NAME="gemma2-2b-stochastok-${MAX_STEPS}steps"
        CHECKPOINT_DIR="/logs/checkpoints/gemma2-2b-stochastok"
        ;;
    patok)
        DATA_DIR="/workspace/data/processed/patok"
        FILE_SUFFIX=""  # patok has no suffix
        WANDB_NAME="gemma2-2b-patok-${MAX_STEPS}steps"
        CHECKPOINT_DIR="/logs/checkpoints/gemma2-2b-patok"
        ;;
    *)
        echo "Error: Unknown tokenization '$TOKENIZATION'"
        echo "Usage: $0 [vanilla|stochastok|patok] [max_steps] [test]"
        exit 1
        ;;
esac

# Generate data paths for all 20 chunks
DATA_PATHS=""
for i in $(seq -w 1 20); do
    DATA_PATHS="$DATA_PATHS ${DATA_DIR}/chunk_00${i}${FILE_SUFFIX}"
done

# Trim leading space
DATA_PATHS="${DATA_PATHS:1}"

echo "Data directory: $DATA_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "WandB run: $WANDB_NAME"
echo ""

# Configure training parameters
# Using 8x A100-40GB:
#   - Global batch size 256 (reasonable for 8 GPUs)
#   - Micro batch size 4 per GPU (256 / 8 / 8 grad accum = 4)
#   - Sequence length 2048

if [ "$TEST_MODE" = "test" ]; then
    MAX_STEPS=100
    CHECKPOINT_INTERVAL=50
    WARMUP_STEPS=10
    echo "*** TEST MODE: Running 100 steps only ***"
else
    CHECKPOINT_INTERVAL=1000  # Save every 1000 steps (checkpoints are 30-35GB each)
    WARMUP_STEPS=100
fi

# Check if pre-converted checkpoint exists
NEMO_CHECKPOINT="/workspace/checkpoints/nemo/google_gemma-2-2b"
RESUME_ARG=""

if [ -d "$PROJECT_ROOT/checkpoints/nemo/google_gemma-2-2b" ]; then
    echo "✓ Found pre-converted NeMo checkpoint"
    RESUME_ARG="--resume-from $NEMO_CHECKPOINT"
else
    echo "⚠ No pre-converted checkpoint found. Training from scratch."
    echo "  To use pretrained weights, first run:"
    echo "  ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b"
    echo ""
    RESUME_ARG=""
fi

echo "Starting Docker container and training..."
# Use torchrun for multi-GPU distributed training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \
    --data-path $DATA_PATHS \
    --max-steps "$MAX_STEPS" \
    --global-batch-size 64 \
    --micro-batch-size 1 \
    --devices 8 \
    --seq-length 512 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --warmup-steps "$WARMUP_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --wandb-project "filipino-morphology" \
    --wandb-name "$WANDB_NAME" \
    --log-every-n-steps 10 \
    $RESUME_ARG

echo ""
echo "=============================================="
echo "Training completed: $TOKENIZATION"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "=============================================="
