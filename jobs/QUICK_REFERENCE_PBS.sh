#!/bin/bash

# ============================================================================
# PBS Quick Reference: Running CPT with NeMo Framework Container
# ============================================================================
# 
# This guide covers everything you need to run Continued Pretraining (CPT) 
# on the Hopper cluster using PBS and Enroot.
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
cd /scratch_aisg/SPEC-SF-AISG/railey/filipino-morphology-llm
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
    --input data/corpora/seapile-v2.jsonl \
    --output-dir data/chunks \
    --num-chunks 20

# Step 2: Process all chunks in parallel (PBS array job)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs
# Each chunk processes independently! 20x faster!

# Monitor: qstat -u $USER
# Check logs: tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>.OU

# 5. Verify preprocessed data
# Single file:
ls -lh data/corpora/seapile-v2_text_document.*

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

# Watch job output
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>.OU

# Training logs
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/training/<JOB_ID>/0_python_master.log

# Check GPU usage (get node name from qstat -f first)
ssh <node_name> nvidia-smi

# MANAGING JOBS
# -------------
# Delete a job
qdel <JOB_ID>

# Hold a job (prevent it from running)
qhold <JOB_ID>

# Release a held job
qrls <JOB_ID>

# Modify job resources (if supported)
qalter -l walltime=24:00:00 <JOB_ID>

# CHECKING RESULTS
# ----------------
# List checkpoints
ls -lh /scratch_aisg/SPEC-SF-AISG/railey/logs/checkpoints/gemma3-cpt-<JOB_ID>/

# View training summary
grep "Training Completed" /scratch_aisg/SPEC-SF-AISG/railey/logs/training/<JOB_ID>/0_python_master.log

# Check for errors
grep -i error /scratch_aisg/SPEC-SF-AISG/railey/logs/training/<JOB_ID>/0_python_master.log

# WandB dashboard
# Visit: https://wandb.ai/<your-username>/<project-name>

# CLEANUP
# -------
# Remove old checkpoints (WARNING: irreversible!)
rm -rf /scratch_aisg/SPEC-SF-AISG/railey/logs/checkpoints/gemma3-cpt-<OLD_JOB_ID>/

# Remove old logs
rm -rf /scratch_aisg/SPEC-SF-AISG/railey/logs/training/<OLD_JOB_ID>/

# COMMON WORKFLOWS
# ----------------

# Workflow 1: Quick test → Full training
qsub jobs/run_cpt_test.pbs
# Wait for completion, check logs
qsub jobs/run_cpt.pbs

# Workflow 2: Iterative training (resume from checkpoint)
qsub jobs/run_cpt.pbs
# After completion, resume with more steps
qsub -v RESUME_FROM=/logs/checkpoints/gemma3-cpt-<JOB_ID>/last.ckpt,MAX_STEPS=500 jobs/run_cpt.pbs

# Workflow 3: Hyperparameter search
for lr in 1e-4 5e-5 1e-5; do
    qsub -v LR=$lr,WANDB_NAME="lr_${lr}" jobs/run_cpt.pbs
done

# DEBUGGING
# ---------

# Check PBS job script
cat jobs/run_cpt.pbs

# Check bash script  
cat jobs/run_cpt.sh

# Check Python script
cat scripts/run_cpt.py

# Test container interactively
enroot start --rw \
    -e HF_HOME=$HF_HOME \
    --mount $PWD:/workspace \
    nemo_framework \
    bash

# Inside container, test Python imports
python -c "import nemo; print(nemo.__version__)"

# USEFUL PBS COMMANDS
# -------------------

# Show queue status
qstat -Q

# Show available resources
pbsnodes -a

# Show my recent jobs (including completed)
qstat -x -u $USER

# Show job history
qstat -xf <JOB_ID>

# Get job exit status
qstat -xf <JOB_ID> | grep Exit_status

# RESOURCE LIMITS
# ---------------

# Check node specifications
pbsnodes -a | grep -A 20 "hopper"

# Common configurations:
# Small:  1 GPU,  16 cores,  64 GB RAM, 1 hour
# Medium: 4 GPUs, 32 cores, 256 GB RAM, 4 hours  (default)
# Large:  8 GPUs, 64 cores, 512 GB RAM, 24 hours (multi-node)

# ENVIRONMENT VARIABLES
# ---------------------

# Check environment in job
qsub -v VAR1=value1,VAR2=value2 jobs/run_cpt.pbs

# Pass all current environment
qsub -V jobs/run_cpt.pbs

# TROUBLESHOOTING
# ---------------

# "Container not found"
enroot list                          # Check if container exists
bash setup_enroot.sh                 # Re-create if missing

# "Data file not found"
ls -lh data/corpora/seapile-v2.jsonl  # Check data exists
cat .env | grep BIND_MOUNTS           # Check mount config
source .env                           # Reload environment

# "WANDB_API_KEY not found"
echo $WANDB_API_KEY                   # Check if set
source .env                           # Reload if needed

# "CUDA out of memory"
qsub -v MBS=1,SEQ_LENGTH=1024 jobs/run_cpt.pbs

# "NCCL timeout" (multi-GPU)
# Check network interface in run_cpt.pbs: export GLOO_SOCKET_IFNAME=ib0
# View NCCL logs: tail -f logs/training/<JOB_ID>/nccl/*.log

# Job killed without error
qstat -xf <JOB_ID> | grep Exit_status  # Check exit code
qstat -xf <JOB_ID> | grep resources    # Check if exceeded limits

# PERFORMANCE TIPS
# ----------------

# 1. Optimize batch size for GPU memory
#    Rule of thumb: MBS * DEVICES * SEQ_LENGTH should fill ~80% of VRAM

# 2. Use appropriate checkpoint interval
#    Too frequent: wastes time, fills disk
#    Too rare: risk losing progress
#    Recommended: every 500-1000 steps

# 3. Monitor GPU utilization
#    Target: >90% GPU utilization
#    If low: increase batch size or reduce micro-batch size

# 4. Use wandb for experiment tracking
#    Always set meaningful WANDB_NAME

# 5. Multi-node considerations
#    Ensure NCCL communication is working
#    Check network interface (GLOO_SOCKET_IFNAME)
#    Test with single node first
