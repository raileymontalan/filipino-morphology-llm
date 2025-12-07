# Environment Setup

Complete step-by-step guide to set up the Filipino Morphology LLM training environment.

## TL;DR - Quick Commands

```bash
# 1. Configure (do this first!)
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY, ENROOT_PATH, HF_HOME, WANDB_DIR, BIND_MOUNTS
source .env

# 2. Install (~15 minutes) - Using Enroot (recommended)
bash setup_enroot.sh

# 3. Verify
./run_in_enroot.sh python -c "import torch, nemo; print(f'PyTorch {torch.__version__}, NeMo {nemo.__version__}')"

# 4. Train
./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100
```

**Alternative (Singularity/Apptainer):**
```bash
# 2. Install with Singularity
bash setup_singularity.sh $CONTAINER_CACHEDIR

# 3-4. Use ./run_in_singularity.sh instead of ./run_in_enroot.sh
```

---

**Detailed Navigation:**
1. [Configuration](#step-1-configure-environment-variables) - **START HERE** (5 min)
2. [Installation](#step-2-install-container) - Pull NeMo container (15 min)
3. [Verification](#step-3-verify-installation) - Test your setup (2 min)
4. [Next Steps](#next-steps) - Run training
5. [Common Issues](#common-issues) - Troubleshooting
6. [Sharing](#sharing-this-code) - Collaboration guidelines

---

## Prerequisites

Before starting, ensure you have:
- Access to HPC cluster with CUDA 12.1+ and **Enroot** (or Singularity/Apptainer as alternative)
- At least 20GB free disk space on `/scratch`
- Weights & Biases account ([sign up free](https://wandb.ai))
- Your W&B API key from https://wandb.ai/authorize

---

## Step 1: Configure Environment Variables

**This must be done FIRST before any installation.**

### 1.1. Create your environment file

```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
nano .env
```

### 1.2. Set required variables in `.env`

Update these values in the `.env` file:

```bash
# ===== Required: HuggingFace =====
# Where HF models/datasets are cached
export HF_HOME=/scratch/$USER/cache/huggingface
# Your HF token from https://huggingface.co/settings/tokens (if using gated models)
export HF_TOKEN="your_huggingface_token_here"

# ===== Required: Weights & Biases =====
# Your W&B API key from https://wandb.ai/authorize
export WANDB_API_KEY="your_actual_wandb_api_key_here"
# Where W&B logs are stored
export WANDB_DIR=/scratch/$USER/logs/wandb

# ===== Required: Enroot Configuration (Recommended) =====
# Where Enroot stores container images and runtime data
export ENROOT_PATH=/scratch/$USER/enroot/
export ENROOT_DATA_PATH=$ENROOT_PATH/data
export ENROOT_RUNTIME_PATH=$ENROOT_PATH/runtime
export ENROOT_CACHE_PATH=$ENROOT_PATH/cache
export ENROOT_TEMP_PATH=$ENROOT_PATH/tmp

# ===== Alternative: Singularity/Apptainer (if not using Enroot) =====
# Uncomment these if using Singularity/Apptainer instead of Enroot
# export CONTAINER_CACHEDIR=/scratch/$USER/container_cache
# export APPTAINER_CACHEDIR=$CONTAINER_CACHEDIR
# export SINGULARITY_CACHEDIR=$CONTAINER_CACHEDIR

# ===== Required: Container Mounts =====
# Mount shared directories inside the container
# - /cache for HuggingFace models (already exists in your structure)
# - /logs for training outputs (already exists in your structure)
export BIND_MOUNTS=/scratch/$USER/cache:/cache,/scratch/$USER/logs:/logs
```

**Important variables:**
- `HF_HOME`: Where HuggingFace caches downloaded models (saves space, reused across projects)
- `HF_TOKEN`: Needed for gated models (Llama, Gemma, etc.) - get from https://huggingface.co/settings/tokens
- `WANDB_API_KEY`: Required for experiment tracking
- `WANDB_DIR`: Where W&B stores local logs before syncing
- `ENROOT_PATH`: Where Enroot stores container images and runtime data (use `/scratch`, NOT home!)
- `ENROOT_*`: Additional Enroot paths for data, runtime, cache, and temp files
- `CONTAINER_CACHEDIR`: (Singularity only) Where the 15GB container downloads
- `BIND_MOUNTS`: Directories accessible inside container
  - Your existing `cache/` → `/cache` (for HF models)
  - Your existing `logs/` → `/logs` (for training outputs)
  - Current project dir is auto-mounted as `/workspace`

### 1.3. Load the environment

```bash
# Load variables for current session
source .env

# Verify they're set
echo "HF_HOME: $HF_HOME"
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."  # Shows first 10 chars
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:10}..."
echo "WANDB_DIR: $WANDB_DIR"
echo "ENROOT_PATH: $ENROOT_PATH"
echo "BIND_MOUNTS: $BIND_MOUNTS"
```

### 1.4. (Optional) Auto-load on login

To automatically load these variables every time you log in:

```bash
# Add to your ~/.bashrc
echo "source /scratch/$USER/filipino-morphology-llm/.env" >> ~/.bashrc
```

---

## Step 2: Install Container

Now that your environment is configured, pull the NeMo Framework container.

### 2.1. Pull and setup container (Enroot - Recommended)

The container includes everything for training:
- PyTorch 2.5+ with CUDA 12.6
- NeMo Framework 2.1+
- Megatron-Core
- TransformerEngine
- All NVIDIA optimizations

```bash
# Run the setup (uses ENROOT_PATH from your .env)
bash setup_enroot.sh
```

This will (takes ~10-15 minutes):
1. Import NeMo Framework container from NGC (`nvcr.io#nvidia/nemo:25.11`)
2. Create `.sqsh` image (~15GB) at $ENROOT_PATH
3. Create container named `nemo_framework`
4. Verify PyTorch, CUDA, and NeMo installations
5. Show helper commands

**Troubleshooting**: If you see errors, make sure you ran `source .env` first.

### 2.1b. Alternative: Singularity/Apptainer

If your system uses Singularity/Apptainer instead of Enroot:

```bash
# Run the setup (uses $CONTAINER_CACHEDIR from your .env)
bash setup_singularity.sh $CONTAINER_CACHEDIR
```

This will create a `.sif` file instead of `.sqsh` and use `run_in_singularity.sh` helper script.

### 2.2. (Optional) Local preprocessing environment

For data preprocessing and analysis (NOT training):

```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install preprocessing tools only
pip install -r requirements.txt
```

**Note**: Training must use the container. Local environment is only for data prep.

---

## Step 3: Verify Installation

### 3.1. Check container and helper scripts

**For Enroot:**
```bash
# Check if container exists
enroot list | grep nemo_framework

# Container image should exist (~15GB)
ls -lh $ENROOT_PATH/nemo_25_11.sqsh

# Helper script should exist
ls -lh run_in_enroot.sh
```

**For Singularity:**
```bash
# Container file should exist (~15GB)
ls -lh nemo_framework_25.11.sif

# Helper script should exist
ls -lh run_in_singularity.sh
```

### 3.2. Test container execution

**For Enroot:**
```bash
# Test Python version
./run_in_enroot.sh python --version

# Test PyTorch and CUDA
./run_in_enroot.sh python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test NeMo
./run_in_enroot.sh python -c "import nemo; print(f'NeMo: {nemo.__version__}')"

# Test NeMo LLM collections
./run_in_enroot.sh python -c "import nemo.collections.llm as llm; print('✓ NeMo LLM ready')"
```

**For Singularity:**
```bash
# Use ./run_in_singularity.sh instead of ./run_in_enroot.sh for all commands above
./run_in_singularity.sh python --version
```

**Expected output:**
```
PyTorch: 2.5.x
CUDA available: True
NeMo: 2.1.x
✓ NeMo LLM ready
```

### 3.3. Check mounted directories (inside container)

**For Enroot:**
```bash
# List what's accessible inside container
./run_in_enroot.sh ls -la /

# Should see:
# /workspace  (your current project directory)
# /cache      (if BIND_MOUNTS includes cache)
# /logs       (if BIND_MOUNTS includes logs)
```

**For Singularity:**
```bash
./run_in_singularity.sh ls -la /
```

### 3.4. (Optional) Verify local environment

If you created the local preprocessing environment:

```bash
source env/bin/activate
python -c "import pandas, transformers; print('✓ Data tools ready')"
```

---

## Next Steps

Your environment is ready! Here's what to do next:

**For Enroot (Recommended):**
```bash
# 1. Prepare training data (if needed)
python scripts/prepare_data.py

# 2. Test training with 100 steps (interactive)
./run_in_enroot.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100

# 3. Submit full training job to cluster
source .env  # Make sure environment is loaded
qsub jobs/submit_cpt_enroot.sh

# 4. Monitor training
tail -f logs/wandb/latest/output.log
```

**For Singularity:**
```bash
# 2. Use ./run_in_singularity.sh instead
./run_in_singularity.sh python scripts/run_cpt_gemma3_1b_container.py --max-steps 100

# 3. Submit Singularity job
qsub jobs/submit_cpt_singularity.sh
```

---

## Common Issues

### Issue 1: Environment variables not loaded

**Cause**: Forgot to source .env file

**Solution**:
```bash
# Make sure you sourced the .env file first
source .env

# Verify it's set
echo $ENROOT_PATH          # For Enroot
echo $CONTAINER_CACHEDIR   # For Singularity

# Then run setup
bash setup_enroot.sh                          # For Enroot
# or
bash setup_singularity.sh $CONTAINER_CACHEDIR # For Singularity
```

### Issue 2: "No space left on device" during container import

**Cause**: Insufficient disk space or wrong directory

**Solution**:
```bash
# Check available space on scratch
df -h /scratch/$USER

# Make sure ENROOT_PATH points to scratch (not home!)
# Edit .env and change to:
export ENROOT_PATH=/scratch/$USER/enroot/

# For Singularity:
# export CONTAINER_CACHEDIR=/scratch/$USER/container_cache

# Reload and retry
source .env
bash setup_enroot.sh                          # For Enroot
# or
bash setup_singularity.sh $CONTAINER_CACHEDIR # For Singularity
```

### Issue 3: Container import fails / network errors

**Cause**: Network issues or NGC authentication problems

**Solution**:
```bash
# Check if you can reach NGC
curl -I https://nvcr.io

# For Enroot, NGC credentials are usually not required for public images
# For Singularity, if authentication is needed:
singularity remote login
```

### Issue 4: Container can't find my data files

**Cause**: Directories not mounted or wrong paths

**Solution**:
```bash
# Check what's mounted inside container (replace with your helper script)
./run_in_enroot.sh ls -la /         # For Enroot
# or
./run_in_singularity.sh ls -la /    # For Singularity

# Your project data is at /workspace/data/
./run_in_enroot.sh ls /workspace/data/

# Shared directories are at /cache and /logs (if configured)
./run_in_enroot.sh ls /cache
./run_in_enroot.sh ls /logs

# To add more mounts, edit .env:
export BIND_MOUNTS="/scratch/$USER/cache:/cache,/scratch/$USER/logs:/logs,/scratch/$USER/other_data:/data"

# Reload and test
source .env
./run_in_enroot.sh ls /data
```

### Issue 5: CUDA not available in container

**Cause**: GPU not detected or container not configured correctly

**Solution**:
```bash
# Check GPU on host first
nvidia-smi

# For Enroot: GPUs are automatically available (no special flag needed)
./run_in_enroot.sh python -c "import torch; print(torch.cuda.is_available())"

# For Singularity: The run_in_singularity.sh already includes --nv flag
./run_in_singularity.sh python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 6: "Permission denied" when running scripts

**Cause**: Helper scripts not executable

**Solution**:
```bash
chmod +x run_in_enroot.sh setup_enroot.sh          # For Enroot
chmod +x run_in_singularity.sh setup_singularity.sh # For Singularity
```

## Sharing This Code

This repository is designed to be shareable without exposing secrets or personal paths.

### Before Sharing - Security Checklist

Ensure you have NOT committed:
- ❌ Your actual `.env` file (use `.env.example` only)
- ❌ API keys or tokens
- ❌ Personal paths (like `/scratch_aisg/SPEC-SF-AISG/your_name/`)
- ❌ Container files (`.sqsh`, `.sif` - large, user-specific)
- ❌ Generated scripts (`run_in_enroot.sh`, `run_in_singularity.sh`, job files)

### For Recipients - Quick Start

When you receive this code, follow the setup steps:

1. **Configure your environment** (see [Step 1](#step-1-configure-environment-variables)):
   ```bash
   cp .env.example .env
   nano .env  # Add your WANDB_API_KEY, HF_TOKEN, ENROOT_PATH, and BIND_MOUNTS
   source .env
   ```

2. **Install container** (see [Step 2](#step-2-install-container)):
   ```bash
   # For Enroot (recommended)
   bash setup_enroot.sh
   
   # For Singularity
   bash setup_singularity.sh $CONTAINER_CACHEDIR
   ```

3. **Verify installation** (see [Step 3](#step-3-verify-installation)):
   ```bash
   # For Enroot
   ./run_in_enroot.sh python -c "import nemo; print(nemo.__version__)"
   
   # For Singularity
   ./run_in_singularity.sh python -c "import nemo; print(nemo.__version__)"
   ```

4. **Submit training job**:
   ```bash
   source .env
   qsub jobs/submit_cpt_enroot.sh      # For Enroot
   # or
   qsub jobs/submit_cpt_singularity.sh # For Singularity
   ```

**Git Security Checklist** (before `git push`):
- [ ] `.env` is in `.gitignore` (keep `.env.example`)
- [ ] No hardcoded API keys
- [ ] No personal paths in version-controlled files
- [ ] Container files (`.sqsh`, `.sif`) and generated scripts ignored

---

## Container vs Local Environment

| Component | Container | Local Env |
|-----------|-----------|-----------|
| **Training (NeMo/Megatron)** | ✅ Required | ❌ Not needed |
| **Data Preprocessing** | ✅ Can use | ✅ Recommended |
| **Analysis/Visualization** | ✅ Can use | ✅ Recommended |
| **PyTorch + CUDA** | ✅ Pre-installed | ❌ Not needed |
| **Jupyter Notebooks** | ✅ Can use | ✅ Recommended |

**Best Practice**: 
- Use **container** for all training/inference
- Use **local environment** for data prep and analysis

## Container Details

- **Image**: `nvcr.io/nvidia/nemo:25.11` (or `nvcr.io#nvidia/nemo:25.11` for Enroot)
- **Python**: 3.10 (inside container)
- **PyTorch**: 2.5+ with CUDA 12.6 support
- **NeMo**: 2.1+ (latest stable)
- **Megatron-Core**: Latest compatible version
- **GPU**: NVIDIA H100/H200 (Hopper architecture)
- **Size**: ~15-20 GB
- **Format**: 
  - Enroot: `.sqsh` (squashfs image)
  - Singularity: `.sif` (Singularity Image Format)
