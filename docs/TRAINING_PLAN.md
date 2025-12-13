# Training Plan for Filipino Morphology Research

This document outlines the complete training plan for the morphology-aware tokenization research paper.

## Research Question

**Can tokenization that preserves morpheme boundaries improve LLM understanding of agglutinative morphology?**

We compare three tokenization approaches on Filipino continued pretraining:
1. **Baseline**: Standard BPE tokenization (vanilla Gemma tokenizer)
2. **Stochastok**: Stochastic token expansion (~10% expansion rate)
3. **Patok**: Morphology-aware expand-contract with Filipino affix awareness

---

## Hardware Resources

**Available GPUs**: 8x NVIDIA A100-SXM4-40GB (320GB total VRAM)

This is sufficient for:
- Gemma 3 1B model training without model parallelism
- Global batch size of 256 with gradient accumulation
- Full bf16 mixed precision training

---

## Training Overview

### Phase 1: Data Preparation

| Step | Command | Time Estimate |
|------|---------|---------------|
| 1.1 Download SEA-PILE v2 | `python scripts/download_seapile.py` | ~2 hours |
| 1.2 Split into chunks | `python training/nemo/data/split_jsonl.py` | ~30 min |
| 1.3 Preprocess baseline | `qsub jobs/preprocess_data_vanilla.pbs` | ~4 hours |
| 1.4 Preprocess stochastok | `qsub jobs/preprocess_data_stochastok.pbs` | ~6 hours |
| 1.5 Preprocess patok | `qsub jobs/preprocess_data_patok.pbs` | ~8 hours |

**Total Data Prep**: ~20 hours (can run preprocessing in parallel)

### Phase 2: Model Training (3 runs)

| Run | Tokenization | Output Path | Time Estimate |
|-----|-------------|-------------|---------------|
| 2.1 Baseline | vanilla | `/logs/checkpoints/gemma3-1b-baseline/` | ~24-48 hours |
| 2.2 Stochastok | stochastok | `/logs/checkpoints/gemma3-1b-stochastok/` | ~24-48 hours |
| 2.3 Patok | patok | `/logs/checkpoints/gemma3-1b-patok/` | ~24-48 hours |

**Total Training**: ~72-144 hours (run sequentially or with separate clusters)

### Phase 3: Evaluation

| Step | Command | Time Estimate |
|------|---------|---------------|
| 3.1 Generate benchmarks | `python scripts/generate_benchmarks.py` | ~10 min |
| 3.2 Evaluate all models | `python scripts/run_evaluation.py --models <model_paths>` | ~4-8 hours |
| 3.3 Analyze results | `python scripts/analyze_inference_results.py` | ~30 min |

---

## Detailed Instructions

### Step 1: Environment Setup

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY

# 2. Install NeMo container
source .env
bash training/nemo/setup/setup_enroot.sh

# 3. Verify setup
enroot list | grep nemo_framework
python scripts/verify_setup.py
```

### Step 2: Download SEA-PILE v2 Corpus

```bash
# Download Filipino subset of SEA-PILE v2 (~7.4GB)
python scripts/download_seapile.py \
    --output data/corpora/seapile-v2-filipino.jsonl \
    --language fil

# Verify download
wc -l data/corpora/seapile-v2-filipino.jsonl
head -1 data/corpora/seapile-v2-filipino.jsonl | python -m json.tool
```

### Step 3: Split Corpus into Chunks

```bash
# Split into 20 chunks for parallel preprocessing
python training/nemo/data/split_jsonl.py \
    --input data/corpora/seapile-v2-filipino.jsonl \
    --output-dir data/chunks \
    --num-chunks 20
```

### Step 4: Preprocess Data (3 versions)

**4a. Baseline (vanilla tokenization)**
```bash
# Generate PBS jobs from templates
bash job_templates/setup_jobs.sh

# Submit parallel preprocessing
qsub -J 1-20 jobs/preprocess_data_vanilla.pbs
```

**4b. Stochastok (10% expansion)**
```bash
# Submit parallel preprocessing with stochastok
qsub -J 1-20 jobs/preprocess_data_stochastok.pbs
```

**4c. Patok (morphology-aware)**
```bash
# Submit parallel preprocessing with patok
qsub -J 1-20 jobs/preprocess_data_patok.pbs
```

### Step 5: Train Models

**5a. Baseline Model**
```bash
qsub jobs/run_cpt_baseline.pbs
# Or interactively:
# ./run_in_enroot.sh python training/nemo/run_cpt.py \
#     --data-path /workspace/data/processed/vanilla/seapile \
#     --checkpoint-dir /logs/checkpoints/gemma3-1b-baseline \
#     --wandb-name gemma3-1b-baseline \
#     --max-steps 10000
```

**5b. Stochastok Model**
```bash
qsub jobs/run_cpt_stochastok.pbs
```

**5c. Patok Model**
```bash
qsub jobs/run_cpt_patok.pbs
```

### Step 6: Evaluate Models

```bash
# Generate evaluation benchmarks
python scripts/generate_benchmarks.py

# Evaluate all three trained models
python scripts/run_evaluation.py \
    --models /logs/checkpoints/gemma3-1b-baseline \
             /logs/checkpoints/gemma3-1b-stochastok \
             /logs/checkpoints/gemma3-1b-patok \
    --benchmarks pacute hierarchical \
    --eval-mode both

# Analyze results
python scripts/analyze_inference_results.py \
    --results-dir results/
```

---

## Training Configuration

### Model: Gemma 3 1B

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-3-1b-pt` |
| Parameters | 1B |
| Sequence length | 2048 |
| Vocab size | 262,144 |

### Optimizer Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Min learning rate | 1e-5 |
| Warmup steps | 500 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Precision | bf16-mixed |

### Batch Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Global batch size | 256 | Across all 8 GPUs |
| Micro batch size | 2 | Per GPU |
| Gradient accumulation | 16 | 256 / (8 * 2) = 16 |

### Training Duration

| Parameter | Value |
|-----------|-------|
| Max steps | 10,000 |
| Checkpoint interval | 1,000 steps |
| Validation interval | 500 steps |
| Log interval | 10 steps |

---

## Tokenization Parameters

### Stochastok
```
expand_prop: 0.1     # 10% of tokens expanded
```

### Patok (MorphologyAwarePatokProcessor)
```
contract_prop: 0.9      # 90% of token positions processed
expand_prop: 0.1        # 10% final expansion
affix_awareness: 0.95   # 95% probability of preserving affixes
num_toks_to_cont: [2, 3, 4]  # Contract 2-4 tokens at a time
contract_prob: [0.35, 0.35, 0.3]  # Probability weights
```

---

## Expected Results

Based on preliminary experiments:

| Method | PACUTE Affixation | Hierarchical Level 2 | Notes |
|--------|-------------------|---------------------|-------|
| Baseline | 40-50% | 30-40% | Standard BPE |
| Stochastok | 50-65% | 45-55% | +10-15% improvement |
| Patok | 60-70% | 55-70% | +20-30% improvement |

**Key Insight**: Level 2 (Morpheme Decomposition) is the critical bottleneck. Improvements there cascade through Levels 3-5.

---

## PBS Job Templates

The following PBS job templates need to be created/updated:

1. `jobs/preprocess_data_vanilla.pbs` - Baseline preprocessing
2. `jobs/preprocess_data_stochastok.pbs` - Stochastok preprocessing
3. `jobs/preprocess_data_patok.pbs` - Patok preprocessing
4. `jobs/run_cpt_baseline.pbs` - Baseline training
5. `jobs/run_cpt_stochastok.pbs` - Stochastok training
6. `jobs/run_cpt_patok.pbs` - Patok training

Use `bash job_templates/setup_jobs.sh` to generate from templates.

---

## Monitoring

### WandB Dashboard

All training runs log to Weights & Biases:
- Project: `gemma3-seapile-cpt`
- Runs: `gemma3-1b-baseline`, `gemma3-1b-stochastok`, `gemma3-1b-patok`

Key metrics to monitor:
- `train/loss` - Training loss
- `val/loss` - Validation loss
- `train/learning_rate` - Learning rate schedule
- `train/consumed_samples` - Training progress

### GPU Utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f /logs/wandb/latest-run/files/output.log
```

---

## Troubleshooting

### Common Issues

1. **OOM Error**: Reduce `micro-batch-size` from 2 to 1
2. **Slow preprocessing**: Increase `--workers` parameter
3. **WandB connection issues**: Set `WANDB_MODE=offline`
4. **Container not found**: Re-run `setup_enroot.sh`

### Checkpoint Recovery

If training crashes, it auto-resumes from the latest checkpoint:
```bash
# Training will automatically resume from --checkpoint-dir
qsub jobs/run_cpt_baseline.pbs
```

---

## Timeline Summary

| Phase | Duration | Parallelizable |
|-------|----------|----------------|
| Data download | 2 hours | No |
| Data preprocessing | 8 hours | Yes (3 versions in parallel) |
| Training (3 models) | 72-144 hours | Partially (if multiple clusters) |
| Evaluation | 8 hours | Yes |
| **Total** | **~90-160 hours** | |

With 8x A100 GPUs and parallel preprocessing, expect **4-7 days** end-to-end.

---

## Next Steps After Training

1. Export models to HuggingFace format
2. Run full evaluation suite
3. Generate analysis plots
4. Write paper sections on methodology and results
