#!/bin/bash
#PBS -N gemma3_1b_cpt
#PBS -l select=1:ncpus=64:ngpus=8:mem=500gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/${PBS_JOBID}.log

# Change to working directory
cd $PBS_O_WORKDIR

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "ERROR: .env file not found!"
    echo "Please create .env with required environment variables"
    exit 1
fi

# ============================================================================
# CONFIGURATION - Modify these variables in .env file
# ============================================================================

CONTAINER_NAME="nemo_framework"
SCRIPT="scripts/run_cpt_gemma3_1b_container.py"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# Verify WandB API Key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set in .env!"
    echo "Please add it to your .env file:"
    echo "  export WANDB_API_KEY='your-key-here'"
    exit 1
fi

# Check if container exists
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container '$CONTAINER_NAME' not found!"
    echo "Please run setup first:"
    echo "  source .env"
    echo "  bash setup_enroot.sh"
    exit 1
fi

# ============================================================================

echo "Starting training job..."
echo "Container: $CONTAINER_NAME"
echo "Script: $SCRIPT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Build mount points from BIND_MOUNTS
MOUNT_ARGS="--mount $PWD:/workspace"

if [ ! -z "$BIND_MOUNTS" ]; then
    IFS=',' read -ra MOUNTS <<< "$BIND_MOUNTS"
    for mount in "${MOUNTS[@]}"; do
        MOUNT_ARGS="$MOUNT_ARGS --mount $mount"
    done
fi

# Run training in container
enroot start \
    --root \
    --rw \
    $MOUNT_ARGS \
    --env HF_HOME=$HF_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$WANDB_DIR \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env NCCL_DEBUG=$NCCL_DEBUG \
    "$CONTAINER_NAME" \
    python /workspace/"$SCRIPT" \
    --devices 8 \
    --max-steps 100
