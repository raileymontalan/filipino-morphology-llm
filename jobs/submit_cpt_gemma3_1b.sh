#!/bin/bash
#PBS -N gemma3-1b-cpt
#PBS -l select=1:ngpus=8
#PBS -l walltime=48:00:00
#PBS -q hopper
#PBS -P SPEC-SF-AISG
#PBS -j oe
#PBS -o logs/gemma3-1b-cpt.log

# Job script for continued pretraining of Gemma 3 1B on Hopper cluster
# Submit with: qsub jobs/submit_cpt_gemma3_1b.sh

echo "=============================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "=============================================="

# Change to project directory
cd $PBS_O_WORKDIR
echo "Working directory: $(pwd)"

# Load conda environment
# Adjust the conda environment name as needed
source env/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Set environment variables
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"  # Replace with your actual key
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Print GPU information
echo "=============================================="
echo "GPU Information:"
nvidia-smi
echo "=============================================="

# Run the training script
echo "Starting training..."
python scripts/run_cpt_gemma3_1b.py \
    --data-path data/corpora/seapile-v2.jsonl \
    --max-steps 10000 \
    --devices 8 \
    --global-batch-size 256 \
    --micro-batch-size 2 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints/gemma3-1b-seapile \
    --wandb-project gemma3-seapile-cpt \
    --wandb-name gemma3-1b-seapile-10k

echo "=============================================="
echo "Job finished at: $(date)"
echo "=============================================="
