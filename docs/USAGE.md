# Usage Guide: Training with Enroot Containers

## Quick Reference

| Script | Purpose | Terminal | When to Use |
|--------|---------|----------|-------------|
| `training/nemo/setup/setup_enroot.sh` | Setup container | Login node | Once (first time) |
| `run_in_enroot.sh` | Interactive helper | Any node | Testing/debugging |
| `training/nemo/run_cpt.py` | Distributed training | Inside container | Called by SLURM |
| `jobs/run_cpt.pbs` | SLURM job wrapper | Login node | Production training |

---

## Complete Workflow

### Step 1: One-Time Setup (Run Once)

```bash
# On login node
cd /scratch_aisg/SPEC-SF-AISG/railey/filipino-morphology-llm

# Load environment variables
source .env

# Setup container (downloads ~15GB, takes 10-15 minutes)
bash training/nemo/setup/setup_enroot.sh

# Verify setup
enroot list | grep nemo_framework
```

**Output**: Container `nemo_framework` should be listed.

---

### Step 2: Interactive Testing (Optional but Recommended)

Test your script before submitting a long job:

```bash
# Test with 10 steps (quick validation)
./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 10

# Or get an interactive shell inside container
./run_in_enroot.sh bash
# Inside container:
cd /workspace
python training/nemo/run_cpt.py --max-steps 1
exit
```

---

### Step 3: Production Training

#### Option A: Interactive Node with GPUs

If you have an interactive session on a compute node:

```bash
# Allocate interactive node first
qsub -I -l select=1:ncpus=64:ngpus=8:mem=500gb -l walltime=4:00:00

# Once on compute node:
cd /scratch_aisg/SPEC-SF-AISG/railey/filipino-morphology-llm
source .env

# Run distributed training with run_in_enroot.sh (simple mode)
./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 100

# Or use distributed mode:
PYTHON_SCRIPT=training/nemo/run_cpt.py \
GPUS_PER_NODE=8 \
LOG_DIR=/scratch_aisg/SPEC-SF-AISG/railey/logs/training/test_run \
./run_in_enroot.sh
```

#### Option B: Submit SLURM Job (Recommended)

Submit a job to run in the background:

```bash
# On login node
cd /scratch_aisg/SPEC-SF-AISG/railey/filipino-morphology-llm
source .env

# Submit the job
qsub jobs/run_cpt.pbs

# Check job status
qstat -u $USER

# Monitor logs (once job starts)
tail -f logs/${PBS_JOBID}.log
tail -f $LOG_DIR/0_python_master.log
```

---

## Understanding the Script Stack

### Layer 1: `training/nemo/setup/setup_enroot.sh`
- **Runs**: On host (login node)
- **Does**: Downloads container, creates Enroot image
- **Frequency**: Once

### Layer 2: `run_in_enroot.sh`
- **Runs**: On host (any node)
- **Does**: Wraps `enroot start`, handles mounts and environment
- **Modes**: 
  - Simple: `./run_in_enroot.sh python script.py`
  - Distributed: `PYTHON_SCRIPT=... ./run_in_enroot.sh`

### Layer 3: `training/nemo/run_cpt.py`
- **Runs**: Inside container
- **Does**: Sets up torchrun, coordinates multi-GPU training
- **Called by**: `submit_cpt_enroot.sh` or `run_in_enroot.sh` (distributed mode)

### Layer 4: `jobs/run_cpt.pbs`
- **Runs**: On host (login node)
- **Does**: SLURM wrapper that calls Layers 2+3
- **Frequency**: Once per training run

---

## Common Scenarios

### Scenario 1: First Time User

```bash
# 1. Setup (once)
source .env
bash training/nemo/setup/setup_enroot.sh

# 2. Test (quick)
./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 1

# 3. Submit job
qsub jobs/run_cpt.pbs
```

### Scenario 2: Debug a Failed Training Run

```bash
# Get interactive node
qsub -I -l select=1:ncpus=64:ngpus=8:mem=500gb -l walltime=1:00:00

# Once on node:
cd /scratch_aisg/SPEC-SF-AISG/railey/filipino-morphology-llm
source .env

# Run interactively to see errors
./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 10

# Or get shell to inspect
./run_in_enroot.sh bash
```

### Scenario 3: Multi-Node Training

```bash
# Edit submit_cpt_enroot.sh:
#PBS -l select=2:ncpus=64:ngpus=8:mem=500gb

# In the script, update:
num_node=2
master_addr=$(hostname)  # First node becomes master

# Then run on each node with different node_rank
# SLURM/PBS usually handles this automatically
```

### Scenario 4: Resume from Checkpoint

```bash
# Edit your training script or add arguments:
./run_in_enroot.sh python training/nemo/run_cpt.py \
    --checkpoint-dir /logs/checkpoints/previous_run \
    --resume-from /logs/checkpoints/previous_run/last.ckpt \
    --max-steps 1000
```

---

## Environment Variables Summary

All set in `.env`:

```bash
# Required
export HF_HOME=/path/to/cache/huggingface
export HF_TOKEN="your_token"
export WANDB_API_KEY="your_key"
export LOG_DIR=/path/to/logs/training
export PYTHON_SCRIPT=training/nemo/run_cpt.py

# Container paths
export SQSH_PATH=/path/to/sqsh/
export ENROOT_PATH=/path/to/enroot/
export BIND_MOUNTS=/path/to/cache:/cache,/path/to/logs:/logs
```

---

## Troubleshooting

### "Container not found"
```bash
# Check if container exists
enroot list | grep nemo_framework

# If not, run setup
source .env
bash training/nemo/setup/setup_enroot.sh
```

### "PYTHONPATH: unbound variable"
Already fixed in `run_cpt_enroot.sh`. Make sure you have the latest version.

### "No space left on device"
```bash
# Check disk space
df -h $SQSH_PATH
df -h $ENROOT_PATH

# Container needs ~15GB
```

### Driver version warning
The warning about driver 575 vs 580 is non-fatal. Container should still work.

---

## Best Practices

1. **Always `source .env` first** before running any script
2. **Test with `--max-steps 1`** before long runs
3. **Use `LOG_DIR`** to organize training logs
4. **Monitor logs** with `tail -f` during training
5. **Keep containers** on shared storage (not home directory)
6. **Set `CUDA_VISIBLE_DEVICES`** to control which GPUs to use

---

## File Locations

```
filipino-morphology-llm/
├── .env                          # Your configuration (don't commit!)
├── training/nemo/setup/setup_enroot.sh               # One-time setup
├── run_in_enroot.sh              # Interactive helper
├── jobs/
│   ├── run_cpt_enroot.sh         # Distributed training (inside container)
│   └── submit_cpt_enroot.sh      # SLURM wrapper
├── scripts/
│   └── run_cpt_gemma3_1b_container.py  # Your training script
└── logs/                         # Training logs (created automatically)
```

Container files (not in repo):
```
$SQSH_PATH/nemo_25_11.sqsh       # Container image (~15GB)
$ENROOT_PATH/nemo_framework/     # Container runtime data
```

---

# Part 2: PBS Job Scripts Reference

This directory contains PBS job submission scripts for running continued pretraining with NeMo Framework on Hopper cluster.

## Available PBS Scripts

### Data Preprocessing Scripts

Located in: `jobs/`

#### `preprocess_test_chunk1.pbs`
**Purpose:** Test preprocessing on a single chunk (recommended first step)

**Configuration:**
- 1 node, 32 CPUs, 256GB RAM
- 2 hour walltime
- Processes: `data/chunks/chunk_0001.jsonl`
- Creates: `data/processed/chunk_0001_text_document.{bin,idx}`

**Usage:**
```bash
qsub jobs/preprocess_test_chunk1.pbs
```

#### `preprocess_data_parallel.pbs`
**Purpose:** Process all chunks in parallel (20x faster!)

**Configuration:**
- PBS array job (#PBS -J 1-20)
- 1 node per chunk, 64 CPUs, 256GB RAM
- 4 hour walltime per chunk
- Processes all 20 chunks simultaneously

**Usage:**
```bash
# Edit line 4 first: change #PBS -J 1-1 to #PBS -J 1-20
qsub jobs/preprocess_data_parallel.pbs
```

#### `preprocess_data.pbs`
**Purpose:** Preprocess full dataset in one job (slower, simpler)

**Configuration:**
- 1 node, 64 CPUs, 512GB RAM
- 8 hour walltime
- Processes: `/workspace/data/corpora/seapile-v2.jsonl`

**Usage:**
```bash
qsub jobs/preprocess_data.pbs
```

### Training Scripts

#### `run_cpt_test.pbs`
**Purpose:** Test training run (short, 1 GPU)

**Configuration:**
- 1 node, 1 GPU
- 1 hour walltime
- 10 training steps
- Tests full pipeline

**Usage:**
```bash
qsub jobs/run_cpt_test.pbs
```

#### `run_cpt.pbs`
**Purpose:** Full training run (multi-GPU)

**Configuration:**
- 1+ nodes, 8 GPUs per node
- 48 hour walltime
- Full training

**Usage:**
```bash
qsub jobs/run_cpt.pbs
```

## Monitoring Jobs

### Check Status
```bash
qstat                  # All your jobs
qstat -u $USER         # Your jobs only
qstat -t               # Array job tasks
qstat -f <JOB_ID>      # Detailed job info
```

### Check Logs
```bash
# Standard PBS logs (in /scratch_aisg/SPEC-SF-AISG/railey/logs/)
tail -f logs/<JOB_ID>.OU

# Preprocessing logs (detailed per-chunk)
tail -f logs/preprocessing/<JOB_ID>/chunk_1/preprocessing.log

# Training logs
tail -f logs/training/<JOB_ID>/training.log
```

### Cancel Jobs
```bash
qdel <JOB_ID>          # Cancel single job
qdel <JOB_ID>[]        # Cancel all array tasks
```

## Customizing Jobs

### Override Variables
```bash
# Change input/output for preprocessing
qsub -v INPUT=/path/to/data.jsonl,OUTPUT_PREFIX=/path/to/output jobs/preprocess_data.pbs

# Change tokenizer
qsub -v TOKENIZER=google/gemma-2-9b jobs/preprocess_data.pbs

# Change number of workers
qsub -v WORKERS=32 jobs/preprocess_test_chunk1.pbs
```

### Edit Script Directly
```bash
nano jobs/preprocess_data.pbs

# Key variables to change:
# - #PBS -l walltime=HH:MM:SS
# - #PBS -l select=N:ncpus=C:mem=MGgb
# - export TOKENIZER="model-name"
# - export WORKERS=64
```

## Workflow Example

```bash
# 1. Test single chunk preprocessing
qsub jobs/preprocess_test_chunk1.pbs

# 2. Monitor test
qstat
tail -f logs/<JOB_ID>.OU

# 3. If test succeeds, process all chunks
# Edit jobs/preprocess_data_parallel.pbs: change line 4 to #PBS -J 1-20
qsub jobs/preprocess_data_parallel.pbs

# 4. Monitor array job
qstat -t

# 5. Check all chunks completed
ls -lh data/processed/*.bin | wc -l   # Should show 20

# 6. Test training
qsub jobs/run_cpt_test.pbs

# 7. Monitor training test
tail -f logs/<JOB_ID>.OU

# 8. If test succeeds, run full training
qsub jobs/run_cpt.pbs
```

## Troubleshooting

### Job Stays in Queue (Q state)
**Cause:** No available nodes or job waiting for resources

**Solution:** Check queue position with `qstat -t`

### Job Fails Immediately
**Check:**
1. PBS logs: `cat logs/<JOB_ID>.OU`
2. Environment loaded: `.env` file exists and sourced
3. Container available: `enroot list` shows `nemo_framework`
4. Input files exist: `ls data/chunks/` or `ls data/corpora/`

### Array Job Partial Failures
**Check which tasks failed:**
```bash
# List all task logs
ls logs/preprocessing/<JOB_ID>/

# Check for errors
grep -i error logs/preprocessing/<JOB_ID>/*/preprocessing.log

# Rerun specific chunk (if chunk 5 failed)
qsub -v ARRAY_INDEX=5 jobs/preprocess_test_chunk1.pbs
```

### Out of Memory
**Increase memory allocation:**
```bash
# Edit PBS script
nano jobs/preprocess_data.pbs

# Change line 4:
#PBS -l select=1:ncpus=64:mem=512GB  # Increase from 256GB
```

### Out of Time
**Increase walltime:**
```bash
nano jobs/preprocess_data.pbs

# Change line 5:
#PBS -l walltime=12:00:00  # Increase from 08:00:00
```

## PBS Quick Reference

```bash
# Submit job
qsub script.pbs

# Submit array job
qsub -J 1-20 script.pbs

# Check status
qstat
qstat -t           # Array jobs
qstat -u $USER     # Your jobs only

# Job details
qstat -f <JOB_ID>

# Cancel job
qdel <JOB_ID>

# Hold job
qhold <JOB_ID>

# Release held job
qrls <JOB_ID>

# Check queue info
qstat -Q

# Node info
pbsnodes -a
```

For complete PBS commands cheatsheet, see `jobs/QUICK_REFERENCE_PBS.sh`.
