# Data Preprocessing Guide

Complete guide for preprocessing text data for continued pretraining with NeMo.

## Quick Start

```bash
# Test on single chunk (recommended first)
qsub jobs/preprocess_test_chunk1.pbs

# Preprocess all chunks in parallel (fastest)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# View all commands
./training/nemo/data/preprocessing_reference.sh
```

---

## Overview

Preprocessing converts JSONL text → Megatron binary format (`.bin` + `.idx`) required by NeMo for efficient training.

**Two tokenization modes supported:**
1. **Vanilla** (default): Standard tokenization
2. **Stochastok**: Token expansion for improved morphology

---

## Tokenization Modes

### Vanilla Tokenization (Default)

Standard tokenization without modifications. Fast and simple.

```bash
qsub jobs/preprocess_data.pbs
```

### Stochastok Tokenization

Applies stochastic token expansion by randomly splitting tokens into sub-tokens. This creates training data with varying token granularities, potentially improving:
- Morphological awareness
- Robustness to tokenization variations
- Performance on morphologically rich languages

```bash
qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data.pbs
```

**How it works:**
1. Tokenize text normally
2. Randomly select ~10% of tokens
3. Split selected tokens into sub-tokens using merge vocabulary
4. Result: Longer sequences with finer-grained tokens

**Example:**
```
Original: "kumain" → [token_123]
Expanded: "kumain" → [token_12, token_3]  (split into sub-tokens)
```

---

## Preprocessing Methods

### Method 1: Single File (Simple)

Best for smaller datasets.

```bash
# Inside container
python /workspace/training/nemo/data/preprocess_data.py \
    --input /workspace/data/corpora/seapile-v2.jsonl \
    --output-prefix /workspace/data/processed/seapile-v2 \
    --tokenizer-model google/gemma-3-1b-pt \
    --workers 32
```

### Method 2: Parallel Chunks (Recommended)

**Much faster** for large datasets. Process chunks in parallel.

#### Step 1: Split JSONL into chunks

```bash
python /workspace/training/nemo/data/split_jsonl.py \
    --input /workspace/data/corpora/seapile-v2.jsonl \
    --output-dir /workspace/data/chunks \
    --num-chunks 20 \
    --tokenizer google/gemma-3-1b-pt
```

Creates: `data/chunks/google-gemma-3-1b-pt/chunk_0001.jsonl` ... `chunk_0020.jsonl`

#### Step 2: Preprocess in parallel

```bash
# Vanilla
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# Stochastok (10% expansion)
qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data_parallel.pbs
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOKENIZER` | `google/gemma-3-1b-pt` | HuggingFace tokenizer model |
| `WORKERS` | `64` | Number of worker processes |
| `JSON_KEY` | `text` | JSON key containing text |
| `TOKENIZATION_MODE` | `vanilla` | `vanilla` or `stochastok` |
| `EXPAND_PROP` | `0.1` | Stochastok: proportion to expand (10%) |
| `SEED` | `42` | Random seed for reproducibility |

**Override parameters:**
```bash
qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.15,SEED=99 jobs/preprocess_data_parallel.pbs
```

---

## Output Organization

Data is automatically organized by tokenization mode:

```
data/processed/
├── google-gemma-3-1b-pt/              # Vanilla tokenization
│   ├── chunk_0001_text_document.bin
│   ├── chunk_0001_text_document.idx
│   └── ...
└── google-gemma-3-1b-pt_stochastok_0.1/   # Stochastok 10%
    ├── chunk_0001_text_document.bin
    ├── chunk_0001_text_document.idx
    └── ...
```

This organization allows:
- Running experiments with different tokenization approaches
- Comparing results easily
- Managing multiple experiments

---

## Complete Workflow

### 1. Test First (Recommended)

```bash
# Test vanilla
qsub jobs/preprocess_test_chunk1.pbs

# Test stochastok
qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_test_chunk1.pbs

# Monitor
qstat -u $USER
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/*.OU
```

### 2. Full Preprocessing

```bash
# Vanilla (all 20 chunks)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# Stochastok (all 20 chunks)
qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data_parallel.pbs
```

### 3. Verify Output

```bash
# Check files created
ls data/processed/google-gemma-3-1b-pt/ | wc -l
# Should show 40 files (20 .bin + 20 .idx)

# Check sizes
du -sh data/processed/google-gemma-3-1b-pt/
du -sh data/processed/google-gemma-3-1b-pt_stochastok_0.1/
```

### 4. Use in Training

```bash
# Generate data paths for training
export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh 20 google/gemma-3-1b-pt)

# Or for stochastok
export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh 20 google-gemma-3-1b-pt_stochastok_0.1)

# Submit training
qsub jobs/run_cpt.pbs
```

---

## Monitoring Jobs

```bash
# Check job status
qstat -u $USER

# View array job details
qstat -t <job_id>

# Check logs
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<job_id>.OU

# Check preprocessing progress
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/preprocessing/<job_id>/chunk_*/preprocessing.log
```

---

## Expected Results

### Vanilla Preprocessing
- Processing time: ~30 min per GB (64 workers)
- Output: Standard binary files

### Stochastok Preprocessing
- Processing time: ~60 min per GB (2x slower)
- Output: Binary files ~10-15% larger
- Logs show expansion statistics:
  ```
  Expansion statistics:
    Original tokens:  1,234,567
    Expanded tokens:  1,358,024
    Expansion ratio:  110.00%
  ```

---

## Troubleshooting

### Issue: "Input file not found"
**Solution:** Check chunk files exist:
```bash
ls data/chunks/google-gemma-3-1b-pt/
```

### Issue: "StochastokProcessor not found"
**Solution:** Verify processor exists in container:
```bash
ls /workspace/src/tokenization/stochastok_processor.py
```

### Issue: Low expansion ratio
**Cause:** Not all tokens can be expanded (e.g., single-byte tokens).  
**Solution:** This is normal. Actual expansion will be less than target `EXPAND_PROP`.

### Issue: Out of memory
**Solution:** Reduce `WORKERS` parameter or process smaller chunks.

### Issue: Job failed
**Check logs:**
```bash
grep -i "error\|failed" /scratch_aisg/SPEC-SF-AISG/railey/logs/<job_id>.OU
```

---

## Tips & Best Practices

✅ **Always test on single chunk first** (`preprocess_test_chunk1.pbs`)  
✅ **Use parallel processing** for large datasets  
✅ **Monitor disk space** - preprocessing needs 2-3x input size  
✅ **Keep different modes separate** - output directories prevent conflicts  
✅ **Use consistent seeds** for reproducible stochastok results  
✅ **Check expansion statistics** to verify stochastok is working  

---

## Command Reference

View all commands with examples:
```bash
./training/nemo/data/preprocessing_reference.sh
```

For questions about tokenization modes, see the CHANGELOG in `CHANGES_SUMMARY.md`.
