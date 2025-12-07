#!/bin/bash
#PBS -N gemma3_1b_cpt
#PBS -l select=1:ncpus=64:ngpus=8:mem=500gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/${PBS_JOBID}.log

# Change to working directory
cd $PBS_O_WORKDIR

# Load modules if needed
# module load singularity

# ============================================================================
# CONFIGURATION - Modify these variables before submitting
# ============================================================================

# Container and script paths
SIF_FILE="nemo_framework_25.11.sif"
SCRIPT="scripts/run_cpt_gemma3_1b_container.py"

# Bind mounts - IMPORTANT: Update these paths for your system!
# Format: /host/path:/container/path
# Example: BIND_MOUNTS="$(pwd):/workspace,/scratch/$USER:/scratch"
BIND_MOUNTS="$(pwd):/workspace"
# Uncomment and modify the line below to add your data directories:
# BIND_MOUNTS="$BIND_MOUNTS,/path/to/your/data:/data"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# WandB API Key - REQUIRED: Set this before submitting!
# Option 1: Set in your environment before submitting
# Option 2: Uncomment and set directly (NOT RECOMMENDED for shared code)
# export WANDB_API_KEY="your-wandb-key-here"
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set!"
    echo "Please set it before submitting:"
    echo "  export WANDB_API_KEY='your-key-here'"
    echo "  qsub jobs/submit_cpt_container.sh"
    exit 1
fi

# ============================================================================

# Run training in container
singularity exec \
    --nv \
    --bind "$BIND_MOUNTS" \
    --pwd /workspace \
    "$SIF_FILE" \
    python "$SCRIPT" \
    --devices 8 \
    --max-steps 100
