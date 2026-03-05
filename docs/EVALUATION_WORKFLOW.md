# Evaluation Workflow for NeMo Checkpoints

## Overview

Training and evaluation are now **separated** for better reliability:
1. **Training** saves checkpoints every N steps
2. **After training**, convert checkpoints to HuggingFace format
3. **Run evaluation** using your existing evaluation pipeline

## Why This Approach?

✅ **No callback complexity** - Training code is simpler and more stable  
✅ **Parallel evaluation** - Evaluate multiple checkpoints simultaneously  
✅ **Retry-friendly** - Failed evaluations don't affect training  
✅ **Flexible** - Evaluate anytime, even while training continues  

## Conversion Requirement

**Yes, you need to convert NeMo/Megatron weights to HuggingFace** because:
- Your `run_evaluation.py` uses `AutoModelForCausalLM` and `AutoTokenizer`
- These expect HuggingFace model format, not Megatron's stacked tensor format

## Workflow

### Option 1: Automated (Recommended)

After training completes, run the conversion + evaluation script:

```bash
# For a single checkpoint
bash scripts/convert_and_evaluate_checkpoint.sh \
    --checkpoint /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942/step=1000.ckpt \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --model-name cerebras-1.3b-step1000

# For all checkpoints in a directory
bash scripts/convert_and_evaluate_checkpoint.sh \
    --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942/ \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --prefix cerebras-1.3b-158942

# Convert only (skip evaluation)
bash scripts/convert_and_evaluate_checkpoint.sh \
    --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942/ \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --prefix cerebras-1.3b-158942 \
    --skip-eval
```

This script:
1. Converts checkpoint(s) to HuggingFace format → `checkpoints/hf/`
2. Adds them to `configs/models.yaml`
3. Submits parallel evaluation jobs (one per checkpoint)

### Option 2: Python Wrapper

```bash
# Convert and evaluate
python scripts/evaluate_checkpoint.py \
    --checkpoint /workspace/checkpoints/.../step=1000.ckpt \
    --base-model cerebras/Cerebras-GPT-1.3B

# Latest checkpoint only
python scripts/evaluate_checkpoint.py \
    --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942/ \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --latest-only

# Custom benchmarks
python scripts/evaluate_checkpoint.py \
    --checkpoint-dir /workspace/checkpoints/.../158942/ \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --benchmarks pacute-affixation-mcq hierarchical-mcq \
    --max-samples 100
```

### Option 3: Manual Steps

```bash
# 1. Convert checkpoint to HuggingFace
python scripts/convert_megatron_to_hf.py \
    --megatron-checkpoint /workspace/checkpoints/.../step=1000.ckpt \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --output checkpoints/hf/cerebras-1.3b-step1000

# 2. Add to configs/models.yaml
cat >> configs/models.yaml << EOF

  cerebras-1.3b-step1000:
    path: checkpoints/hf/cerebras-1.3b-step1000
    type: pt
EOF

# 3. Run evaluation using your existing pipeline
ALL_MODELS="cerebras-1.3b-step1000" \
BENCHMARKS="pacute-affixation-mcq pacute-composition-mcq hierarchical-mcq langgame-mcq" \
bash jobs/submit_parallel_evaluation.sh
```

## Training Configuration

Your training script (`run_cpt.py`) now:
- **Saves checkpoints** every N steps (configurable via `CHECKPOINT_INTERVAL`)
- **No inline evaluation** - callback removed for stability
- **Checkpoints saved to**: `/workspace/checkpoints/{model_name}/{variant}/{job_id}/`

Example checkpoint path:
```
/workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942.hopper-m-02/step=1000-consumed=8192000.ckpt
```

## Checkpoint Conversion

The conversion script (`convert_megatron_to_hf.py`) handles:
- **Embedding layers** - Word embeddings + LM head (weight tying)
- **Layer norms** - Pre/post attention and feedforward norms
- **Attention weights** - QKV projection splitting for multi-query attention
- **MLP weights** - Gated feedforward networks (gate + up + down projections)
- **Stacked tensors** - Megatron stacks all layers, script splits them

Supports:
- ✅ Cerebras-GPT models
- ✅ GPT-2 variants
- ✅ Qwen models
- ✅ Gemma 2/3 models (with GQA)

## Evaluation

Uses your existing evaluation pipeline:
- **Benchmarks**: PACUTE, CUTE, LangGame, Hierarchical, Multi-digit Addition
- **Modes**: MCQ (log probability) and Generative
- **Metrics**: Accuracy, F1, Precision, Recall, Normalized Accuracy
- **Output**: Results saved to `results/benchmark_evaluation/`

## Monitoring Progress

```bash
# Check training job
qstat | grep 158942

# View training log
tail -f logs/158942.hopper-m-02.OU

# Check for saved checkpoints
ls -lh /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942.hopper-m-02/

# Check evaluation jobs
qstat -u $USER | grep eval

# View evaluation results
ls results/benchmark_evaluation/
```

## Troubleshooting

**Training fails with callback error**
→ Make sure you're using the latest `run_cpt.py` without `EvaluationCallback`

**Conversion fails**
→ Check that checkpoint file exists and base model is correct

**Evaluation fails to find model**
→ Verify model was added to `configs/models.yaml`

**Out of memory during evaluation**
→ Reduce batch size or use smaller model for testing

## Example End-to-End

```bash
# 1. Submit training job
qsub jobs/run_cpt_test.pbs

# 2. Wait for checkpoints (training saves every 200 steps)
# Monitor: tail -f logs/158942.hopper-m-02.OU

# 3. After training, convert and evaluate all checkpoints
bash scripts/convert_and_evaluate_checkpoint.sh \
    --checkpoint-dir /workspace/checkpoints/cerebras-Cerebras-GPT-1.3B/vanilla/158942.hopper-m-02/ \
    --base-model cerebras/Cerebras-GPT-1.3B \
    --prefix cerebras-1.3b-cpt

# 4. Monitor evaluation jobs
qstat -u $USER

# 5. View results
cat results/benchmark_evaluation/summary.json
```
