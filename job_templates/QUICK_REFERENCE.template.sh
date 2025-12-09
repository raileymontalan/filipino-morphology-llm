#!/bin/bash

# ============================================================================
# PBS Quick Reference: Running CPT with NeMo Framework Container - TEMPLATE
# ============================================================================
# 
# This is a TEMPLATE file with generic paths and queue names.
# Copy to jobs/ and customize for your cluster.
#
# This guide covers everything you need to run Continued Pretraining (CPT) 
# on HPC clusters using PBS and Enroot.
#
# Files:
#   - jobs/run_cpt.pbs       → Main PBS job (multi-GPU, 100 steps default)
#   - jobs/run_cpt_test.pbs  → Test PBS job (single GPU, 10 steps)
#   - jobs/run_cpt.sh        → Bash launcher (called by PBS)
#   - scripts/run_cpt.py     → Python training script (runs in container)
#
# ============================================================================

# SETUP (One-time)
# ----------------
# 1. Configure environment variables
cd /path/to/your/project
source .env

# 2. Setup NeMo container via Enroot (downloads ~15GB)
bash setup_enroot.sh

# 3. Verify container exists
enroot list | grep nemo_framework

# 4. Preprocess data (JSONL → binary format)

# OPTION A: Single large file (slower, but simpler)
qsub jobs/preprocess_data.pbs
# Uses 64 workers, ~4-8 hours depending on size

# OPTION B: Parallel processing (MUCH FASTER! 🚀)
# Step 1: Split into chunks
python scripts/split_jsonl.py \
    --input data/corpora/your_data.jsonl \
    --output-dir data/chunks \
    --num-chunks 20

# Step 2: Process all chunks in parallel (PBS array job)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs
# Each chunk processes independently! 20x faster!

# Monitor: qstat -u $USER
# Check logs: tail -f /path/to/your/logs/<JOB_ID>.OU

# 5. Verify preprocessed data
# Single file:
ls -lh data/corpora/your_data_text_document.*

# Or chunks:
ls -lh data/chunks/chunk_*_text_document.*

# SUBMITTING TRAINING JOBS
# -------------------------
# IMPORTANT: Preprocess data first (step 4 above)!

# 1. Test run first (single GPU, 10 steps, ~5 min)
qsub jobs/run_cpt_test.pbs

# 2. Full training run (4 GPUs, 100 steps, default config)
qsub jobs/run_cpt.pbs

# 3. Custom parameters via qsub -v
qsub -v MAX_STEPS=1000,GBS=512,MBS=4 jobs/run_cpt.pbs
qsub -v SEQ_LENGTH=4096,LR=5e-5,WARMUP_STEPS=100 jobs/run_cpt.pbs
qsub -v DATA_PATH=/workspace/data/my_data_text_document jobs/run_cpt.pbs

# 4. Blended datasets (train on multiple datasets with weights)
# If you processed chunks in parallel, blend them:
qsub -v DATA_PATH="/workspace/data/chunks/chunk_0001_text_document,/workspace/data/chunks/chunk_0002_text_document" jobs/run_cpt.pbs
# Or with custom weights (must add to run_cpt.py call in PBS script)

# 5. Multi-node training (edit run_cpt.pbs: #PBS -l select=2:ncpus=32:ngpus=4:...)
qsub jobs/run_cpt.pbs

# Key parameters you can override:
#   MAX_STEPS, GBS, MBS, SEQ_LENGTH, LR, MIN_LR, WARMUP_STEPS
#   CKPT_INTERVAL, RESUME_FROM, DATA_PATH, WANDB_NAME

# MONITORING JOBS
# ---------------
# List your jobs
qstat -u $USER

# Detailed job info
qstat -f <JOB_ID>

# Watch job output (replace with your log directory)
tail -f /path/to/your/logs/<JOB_ID>.OU

# Training logs (replace with your log directory)
tail -f /path/to/your/logs/training/<JOB_ID>/0_python_master.log

# Check GPU usage (get node name from qstat -f first)
ssh <node_name> nvidia-smi

# CHECKPOINT MANAGEMENT
# ---------------------
# Checkpoints saved to: /path/to/your/checkpoints/gemma3-cpt-<JOB_ID>

# Resume from checkpoint
qsub -v RESUME_FROM=/path/to/your/checkpoints/gemma3-cpt-12345/checkpoint-step-1000 jobs/run_cpt.pbs

# Convert to HuggingFace format
python scripts/convert_nemo_to_hf.py \
    --nemo-checkpoint /path/to/your/checkpoints/gemma3-cpt-12345/checkpoint-step-1000.nemo \
    --output-dir models/my-model-1k

# TROUBLESHOOTING
# ---------------
# "Container not found"
#   → Run setup_enroot.sh first

# "Data not found" 
#   → Preprocess data first (step 4 above)

# "NCCL timeout"
#   → Check network interface in run_cpt.pbs (GLOO_SOCKET_IFNAME)
#   → Common options: ib0, eth0, eno1

# "Out of memory"
#   → Reduce GBS or MBS: qsub -v GBS=128,MBS=1 jobs/run_cpt.pbs
#   → Reduce SEQ_LENGTH: qsub -v SEQ_LENGTH=1024 jobs/run_cpt.pbs

# "Queue limits exceeded"
#   → Check queue limits: qstat -Q
#   → Ask admin about resource availability

# USEFUL PBS COMMANDS
# -------------------
# Delete job: qdel <JOB_ID>
# Hold job: qhold <JOB_ID>
# Release job: qrls <JOB_ID>
# Check queue limits: qstat -Q
# Check node resources: pbsnodes -a
# Check your quota: quota -s

# COMMON QUEUE NAMES (varies by cluster):
# - gpu           → General GPU queue
# - debug         → Debug/test queue (limited resources)
# - batch         → Batch processing queue
# - AISG_debug    → Custom cluster-specific queue
# - YOUR_QUEUE    → Replace with your cluster's queue name

# WANDB TRACKING
# --------------
# View training metrics: https://wandb.ai/your-team/gemma3-seapile-cpt

# Login to WandB (one-time):
wandb login <YOUR_API_KEY>

# Disable WandB logging:
export WANDB_MODE=disabled
qsub jobs/run_cpt.pbs

# EVALUATION
# ----------
# After training completes, evaluate on benchmarks:
qsub -v MODEL_NAME=my-model-1k jobs/run_evaluation_batch.pbs

# Quick test:
qsub jobs/run_evaluation_test.pbs

# ============================================================================
# EXAMPLE WORKFLOW
# ============================================================================

# 1. Setup (one-time)
cd /path/to/your/project
source .env
bash setup_enroot.sh

# 2. Preprocess data
qsub jobs/preprocess_data.pbs
# Wait for completion...

# 3. Test run
qsub jobs/run_cpt_test.pbs
# Check logs to verify everything works

# 4. Full training
qsub -v MAX_STEPS=10000,GBS=512 jobs/run_cpt.pbs

# 5. Monitor
tail -f /path/to/your/logs/training/<JOB_ID>/0_python_master.log

# 6. Convert checkpoint to HuggingFace
python scripts/convert_nemo_to_hf.py \
    --nemo-checkpoint /path/to/checkpoint.nemo \
    --output-dir models/my-model

# 7. Evaluate
qsub -v MODEL_NAME=my-model jobs/run_evaluation_batch.pbs

# ============================================================================
# For more information:
#   - docs/TRAINING.md      → Detailed training documentation
#   - docs/EVALUATION.md    → Evaluation guide
#   - docs/PREPROCESSING.md → Data preprocessing guide
# ============================================================================
