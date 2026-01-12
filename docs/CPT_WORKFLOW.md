# Continued Pretraining (CPT) Workflow

This directory contains scripts and jobs for continued pretraining of language models on Filipino data with different tokenization strategies.

## Overview

The CPT workflow trains models on a **50/50 mix of SEA-PILE v2 and FineWeb** data to avoid catastrophic forgetting from out-of-domain data. It supports multiple base models and tokenization methods.

## Key Features

- **Multiple Base Models**: gemma-3-1b-pt, gpt2-xl, Cerebras-GPT-1.3B, Qwen3-1.7B-Base
- **Three Tokenization Methods**: vanilla (BPE), stochastok, patok
- **Balanced Dataset Mix**: 50/50 seapile-v2/fineweb to prevent domain shift
- **Long Training**: 100k steps with proper warmup (500 steps)
- **Organized Checkpoints**: Saved by model/method/job_id

## Directory Structure

### Preprocessed Data
```
data/processed/
├── <tokenizer>/
│   ├── seapile-v2/
│   │   ├── vanilla/
│   │   │   ├── chunk_0001.bin
│   │   │   ├── chunk_0001.idx
│   │   │   └── ...
│   │   ├── stochastok/
│   │   │   ├── expand_0.1/
│   │   │   ├── expand_0.2/
│   │   │   └── ...
│   │   └── patok/
│   │       ├── expand_0.1_contract_0.9_affix_0.95_overlap_0.75/
│   │       └── ...
│   └── fineweb/
│       └── (same structure as seapile-v2)
```

### Checkpoints
```
checkpoints/
├── gemma-3-1b-pt/
│   ├── vanilla/
│   │   └── <job_id>/
│   ├── stochastok-0.1/
│   │   └── <job_id>/
│   └── patok-expand_0.1_contract_0.9_affix_0.95_overlap_0.75/
│       └── <job_id>/
├── gpt2-xl/
│   └── ...
└── ...
```

## Usage

### 1. Preprocess Data (First Time Only)

Generate tokenizations for all datasets, models, and methods:

```bash
# Full sweep: all datasets × tokenizers × modes × parameters
bash scripts/submit_tokenizations.sh sweep
```

This creates preprocessed data for both `seapile-v2` and `fineweb` with all tokenization configurations.

### 2. Submit CPT Jobs

#### Single Job
```bash
# Vanilla tokenization
bash scripts/submit_cpt.sh google/gemma-3-1b-pt seapile-v2/google-gemma-3-1b-pt/vanilla

# Stochastok tokenization
bash scripts/submit_cpt.sh google/gemma-3-1b-pt seapile-v2/google-gemma-3-1b-pt/stochastok/expand_0.1

# Patok tokenization
bash scripts/submit_cpt.sh google/gemma-3-1b-pt seapile-v2/google-gemma-3-1b-pt/patok/expand_0.1_contract_0.9_affix_0.95_overlap_0.75
```

#### Full Sweep
Submit CPT jobs for all model × tokenization combinations:

```bash
bash scripts/submit_cpt.sh sweep
```

This submits:
- 4 models × (1 vanilla + 10 stochastok + 36 patok) = **188 CPT jobs**
- Patok: 2 expand × 3 contract × 3 affix × 2 overlap = 36 configurations

### 3. Monitor Training

```bash
# Check job status
qstat

# View logs
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/training/<job_id>/training.log

# Monitor WandB
# Project: filipino-morphology-cpt
# Run name: <model>-<method>-<job_id>
```

## Training Configuration

### Default Hyperparameters
```bash
MAX_STEPS=100000          # 100k steps (vs previous 10k)
WARMUP_STEPS=500          # 0.5% of total steps
SEQ_LENGTH=2048           # 2048 tokens
GBS=256                   # Global batch size
MBS=2                     # Micro batch size
LR=1e-4                   # Learning rate
MIN_LR=1e-5              # Minimum LR
CKPT_INTERVAL=20000      # Save every 20k steps (5 checkpoints total)
VAL_CHECK_INTERVAL=1000  # Validate every 1k steps
RUN_EVAL_ON_CHECKPOINT=true  # Run evaluation after each checkpoint save
```

### Override Parameters
```bash
# Custom hyperparameters
qsub -v MODEL_NAME="google/gemma-3-1b-pt",\
TOKENIZATION_PATH="seapile-v2/google-gemma-3-1b-pt/vanilla",\
MAX_STEPS=50000,\
LR=5e-5 \
jobs/run_cpt.pbs
```

## Quick Test

Test the pipeline with minimal steps:

```bash
# Uses first chunk only, 10 steps
qsub -v MODEL_NAME="google/gemma-3-1b-pt",\
TOKENIZATION_PATH="seapile-v2/google-gemma-3-1b-pt/vanilla" \
jobs/run_cpt_test.pbs
```

## Environment Variables

### Required (in .env file)
```bash
HF_TOKEN=<your_huggingface_token>
WANDB_API_KEY=<your_wandb_key>
SQSH_PATH=/path/to/container/images/
BIND_MOUNTS=/cache:/cache,/data:/data
```

### Job-Specific
```bash
MODEL_NAME              # Base model to continue training
TOKENIZATION_PATH       # Path to tokenization (dataset/tokenizer/method/params)
MAX_STEPS              # Total training steps
GBS                    # Global batch size
MBS                    # Micro batch size
LR                     # Learning rate
WARMUP_STEPS           # Warmup steps
CKPT_INTERVAL          # Checkpoint save interval
```

## Data Mix Strategy

The training data is automatically mixed 50/50 from both datasets:

1. **SEA-PILE v2** (Filipino-specific):
   - Path: `data/processed/<tokenizer>/seapile-v2/<method>/`
   - In-domain Filipino data

2. **FineWeb** (General web data):
   - Path: `data/processed/<tokenizer>/fineweb/<method>/`
   - Out-of-domain general knowledge

This mix prevents catastrophic forgetting while adapting to Filipino morphology.

## Checkpoint Organization

Checkpoints are saved to:
```
checkpoints/<model>/<method>-<params>/<job_id>/
```

Examples:
- `checkpoints/gemma-3-1b-pt/vanilla/153487/`
- `checkpoints/gemma-3-1b-pt/stochastok-0.1/153488/`
- `checkpoints/gpt2-xl/patok-expand_0.1_contract_0.9_affix_0.95_overlap_0.75/153489/`

This structure makes it easy to:
- Compare methods for the same model
- Track experiments by job ID
- Resume training from specific checkpoints

## WandB Logging

All runs are logged to WandB:

- **Project**: `filipino-morphology-cpt`
- **Run Name**: `<model>-<method>-<job_id>`
  - Example: `gemma-3-1b-pt-vanilla-153487`
  - Example: `gpt2-xl-stochastok-0.1-153488`

Logged metrics:
- Training loss
- Learning rate schedule
- Gradient norms
- Throughput (tokens/sec)
- Validation perplexity (every 1k steps)

## Troubleshooting

### No chunks found
```bash
# Verify preprocessed data exists
ls data/processed/<tokenizer>/seapile-v2/<method>/
ls data/processed/<tokenizer>/fineweb/<method>/

# Re-run preprocessing if missing
bash scripts/submit_tokenizations.sh <dataset> <tokenizer> <mode> [params...]
```

### Container not found
```bash
# Setup enroot container first
bash training/nemo/setup/setup_enroot.sh
enroot list | grep nemo_framework
```

### CUDA errors
Check compute node has GPUs:
```bash
# In PBS job
nvidia-smi
```

### OOM (Out of Memory)
Reduce batch sizes:
```bash
qsub -v GBS=128,MBS=1 jobs/run_cpt.pbs
```

## Related Scripts

- `scripts/submit_tokenizations.sh` - Preprocess data with different tokenizations
- `scripts/submit_cpt.sh` - Submit CPT jobs (this workflow)
- `jobs/run_cpt.pbs` - Main CPT PBS job script
- `jobs/run_cpt.sh` - Container launch script
- `scripts/run_cpt.py` - Python training script

## Notes

- **100k steps** @ GBS=256 ≈ 25.6M tokens (with seq_len=2048)
- **Warmup (500 steps)** = 0.5% of training
- **Checkpoint interval (20k steps)** = 5 checkpoints total (at 20k, 40k, 60k, 80k, 100k)
- **Evaluation after each checkpoint** - Performance metrics tracked over time
- **Mixed data** prevents catastrophic forgetting from Filipino-only training
- **Job IDs** track individual experiments for reproducibility

### Evaluation After Checkpoints

After each checkpoint is saved (every 20k steps), the system automatically:
1. Loads the checkpoint
2. Runs evaluation on **all MCQ benchmarks**:
   - PACUTE (Affixation, Composition, Manipulation, Syllabification)
   - Hierarchical benchmark
   - LangGame
   - Multi-digit Addition
3. Saves results to `checkpoints/<model>/<method>/<job_id>/eval_step_<N>.json`
4. Logs metrics to WandB for tracking improvement over time

**Why MCQ only?** MCQ evaluation using log probabilities is much faster than generative evaluation (~5-10 minutes vs hours), making it practical to run after each checkpoint during training.

This allows you to:
- Track model performance across all benchmarks throughout training
- Identify optimal checkpoint (may not be the final one)
- Detect overfitting or degradation
- Compare learning curves across tokenization methods
- See which linguistic capabilities improve at different training stages
