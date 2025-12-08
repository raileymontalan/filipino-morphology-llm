# Training Guide

Complete guide for training models with NeMo continued pretraining (CPT).

## Quick Start

```bash
# 1. Preprocess data first (see training/nemo/data/DATA_PREPROCESSING.md)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# 2. Submit training job
qsub jobs/run_cpt.pbs

# 3. Monitor
qstat -u $USER
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/*.OU
```

---

## Complete Workflow

### Phase 1: Environment Setup (One-Time)

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY, ENROOT_PATH, etc.
source .env

# 2. Setup container (~15 minutes)
bash training/nemo/setup/setup_enroot.sh

# 3. Verify
enroot list | grep nemo_framework
```

See `SETUP.md` for detailed setup instructions.

### Phase 2: Data Preprocessing

```bash
# Test preprocessing
qsub jobs/preprocess_test_chunk1.pbs

# Full preprocessing (parallel, faster)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# Monitor
qstat -u $USER
```

See `training/nemo/data/DATA_PREPROCESSING.md` for complete preprocessing guide.

### Phase 3: Training

```bash
# Run training with defaults
qsub jobs/run_cpt.pbs

# Or customize hyperparameters
qsub -v MAX_STEPS=1000,GBS=512,LR=5e-5 jobs/run_cpt.pbs
```

---

## Training Configurations

### Available PBS Jobs

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `jobs/run_cpt_test.pbs` | Quick test (10 steps) | Verify setup works |
| `jobs/run_cpt.pbs` | Production training | Full training runs |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STEPS` | `100` | Total training steps |
| `GBS` | `256` | Global batch size |
| `MBS` | `2` | Micro batch size |
| `LR` | `1e-4` | Learning rate |
| `RESUME_FROM` | `google/gemma-3-1b-pt` | Model checkpoint to start from |
| `DATA_PATH` | Auto-generated | Path(s) to preprocessed data |

### Example Configurations

**Quick test:**
```bash
qsub jobs/run_cpt_test.pbs
```

**Full training:**
```bash
qsub -v MAX_STEPS=10000,GBS=256 jobs/run_cpt.pbs
```

**Custom hyperparameters:**
```bash
qsub -v MAX_STEPS=5000,GBS=512,LR=5e-5,MBS=4 jobs/run_cpt.pbs
```

**Resume from checkpoint:**
```bash
qsub -v RESUME_FROM=/workspace/nemo_experiments/baseline/checkpoints/step_1000.nemo jobs/run_cpt.pbs
```

---

## Experimental Pipeline

### Comparing Tokenization Methods

The research compares three tokenization approaches:

1. **Baseline (Vanilla BPE)**
2. **Stochastok (Token Expansion)**
3. **Patok (Affix-Aware)** - Future work

#### Experiment 1: Baseline

```bash
# Preprocess with vanilla tokenization
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# Train
export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh 20 google/gemma-3-1b-pt)
qsub -v EXPERIMENT_NAME=baseline jobs/run_cpt.pbs
```

Output: `nemo_experiments/baseline/`

#### Experiment 2: Stochastok

```bash
# Preprocess with stochastok tokenization
qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data_parallel.pbs

# Train
export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh 20 google-gemma-3-1b-pt_stochastok_0.1)
qsub -v EXPERIMENT_NAME=stochastok_0.1 jobs/run_cpt.pbs
```

Output: `nemo_experiments/stochastok_0.1/`

#### Experiment 3: Patok (Future)

```bash
# Preprocess with patok tokenization (to be implemented)
qsub -J 1-20 -v TOKENIZATION_MODE=patok,EXPAND_PROP=0.3,CONTRACT_PROP=0.3 jobs/preprocess_data_parallel.pbs

# Train
export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh 20 google-gemma-3-1b-pt_patok)
qsub -v EXPERIMENT_NAME=patok jobs/run_cpt.pbs
```

Output: `nemo_experiments/patok/`

---

## Monitoring Training

### Check Job Status

```bash
# View running jobs
qstat -u $USER

# Detailed job info
qstat -f <job_id>

# View logs
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<job_id>.OU
```

### Monitor Training Progress

```bash
# View training logs
tail -f nemo_experiments/<experiment_name>/nemo_log_globalrank-0_localrank-0.txt

# Check checkpoints
ls -lh nemo_experiments/<experiment_name>/checkpoints/
```

### Weights & Biases Dashboard

View real-time metrics at: https://wandb.ai

Training automatically logs:
- Loss curves
- Learning rate
- Throughput (tokens/sec)
- GPU utilization

---

## Output Structure

```
nemo_experiments/
└── <experiment_name>/          # e.g., "baseline", "stochastok_0.1"
    ├── checkpoints/            # Model checkpoints
    │   ├── step_100.nemo
    │   ├── step_200.nemo
    │   └── ...
    ├── nemo_log_*.txt          # Training logs
    └── hparams.yaml            # Hyperparameters used
```

---

## Interactive Testing (Optional)

Test scripts interactively before submitting jobs:

```bash
# Get interactive shell in container
./run_in_enroot.sh bash

# Inside container:
cd /workspace

# Test training with 10 steps
python training/nemo/run_cpt.py --max-steps 10

# Exit container
exit
```

---

## Common Training Issues

### Issue: "Container not found"
**Solution:** Run setup first:
```bash
bash training/nemo/setup/setup_enroot.sh
```

### Issue: "Data path not found"
**Solution:** Verify preprocessed data exists:
```bash
ls data/processed/google-gemma-3-1b-pt/
```

### Issue: Out of memory
**Solutions:**
- Reduce `MBS` (micro batch size)
- Reduce `GBS` (global batch size)
- Use gradient checkpointing

### Issue: Training too slow
**Solutions:**
- Increase `MBS` for better GPU utilization
- Check GPU usage: `nvidia-smi`
- Verify data loading isn't bottleneck

### Issue: Loss not decreasing
**Solutions:**
- Check learning rate (try lower: `1e-5`)
- Verify data quality
- Check for data preprocessing errors

---

## PBS Job Configuration

### Resource Requirements

Default in `jobs/run_cpt.pbs`:
- **Nodes**: 1
- **GPUs**: 4x A100 (40GB each)
- **CPUs**: 32
- **Memory**: 256GB
- **Walltime**: 24:00:00

### Modifying Resources

Edit `jobs/run_cpt.pbs`:
```bash
#PBS -l select=1:ncpus=32:ngpus=4:mem=256GB
#PBS -l walltime=24:00:00
```

---

## Next Steps

After training:
1. **Evaluate models** - See `docs/EVALUATION.md`
2. **Compare results** - Analyze which tokenization works best
3. **Document findings** - Record observations

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `run_in_enroot.sh` | Run commands in container interactively |
| `training/nemo/data/generate_chunk_paths.sh` | Generate data paths for training |
| `training/nemo/data/preprocessing_reference.sh` | View preprocessing commands |
| `jobs/QUICK_REFERENCE_PBS.sh` | View PBS commands |

---

## Tips & Best Practices

✅ **Test with `run_cpt_test.pbs` first** (10 steps, quick validation)  
✅ **Monitor disk space** - checkpoints can be large  
✅ **Use Weights & Biases** for tracking experiments  
✅ **Save checkpoints frequently** in case of interruptions  
✅ **Name experiments clearly** (e.g., "baseline", "stochastok_0.1")  
✅ **Document hyperparameters** used for each run  

---

## Advanced: Multi-GPU Training

NeMo automatically uses all available GPUs via PyTorch DDP:

```python
# In run_cpt.py
trainer = Trainer(
    devices=4,  # Use 4 GPUs
    strategy="ddp",  # Distributed Data Parallel
    ...
)
```

No manual configuration needed - just request GPUs in PBS script.

---

## Advanced: Custom Data Paths

```bash
# Multiple chunks
export DATA_PATH="/workspace/data/processed/chunk_0001 /workspace/data/processed/chunk_0002"

# Single file
export DATA_PATH="/workspace/data/processed/seapile-v2"

# Mixed sources (not recommended)
export DATA_PATH="/workspace/data/processed/source1 /workspace/data/processed/source2"

qsub jobs/run_cpt.pbs
```

---

For research background and experimental design, see `docs/RESEARCH.md`.  
For evaluation procedures, see `docs/EVALUATION.md`.
