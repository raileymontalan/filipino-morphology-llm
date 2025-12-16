# Llama 3.1 8B Adaptation Plan

## Overview
Adapt the CPT (Continued Pre-Training) pipeline to train Llama 3.1 8B with the three tokenization methods (vanilla, stochastok, patok) on the same SEA-PILE Filipino corpus.

**Goal:** Show that morphology-aware tokenization generalizes across model architectures and scales.

---

## 1. Model Setup & Conversion

### Download Base Model
```bash
# Option A: Direct download
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir checkpoints/hf/llama-3.1-8b

# Option B: Via transformers
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.1-8B', cache_dir='checkpoints/hf')"
```

**Requirements:**
- HuggingFace access token with Llama access approved
- ~16GB disk space for HF checkpoint

### Convert to NeMo Format
```bash
./run_in_docker.sh python scripts/convert_hf_to_nemo.py \
    --model meta-llama/Llama-3.1-8B \
    --output checkpoints/nemo/llama-3.1-8b
```

**Action Items:**
- [ ] Verify `convert_hf_to_nemo.py` supports Llama 3.1 architecture
- [ ] May need to update conversion script for Llama-specific layers
- [ ] Test conversion produces valid NeMo checkpoint

---

## 2. Tokenizer Differences

### Gemma vs Llama Tokenizers

| Aspect | Gemma 2 2B | Llama 3.1 8B |
|--------|------------|--------------|
| Type | SentencePiece | Tiktoken-based (BPE) |
| Vocab Size | 256,000 | 128,000 |
| Special Tokens | `<bos>`, `<eos>`, `<pad>` | `<|begin_of_text|>`, `<|end_of_text|>` |
| Tokenization Style | Unigram LM | Byte-level BPE |

### Tokenizer Integration

**Current code:**
```python
# In preprocess_data.py
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

**Needs to become:**
```python
# Model-agnostic approach
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
```

**Action Items:**
- [ ] Parameterize tokenizer loading in `preprocess_data.py`
- [ ] Add `--model-name` argument to preprocessing scripts
- [ ] Test that patok/stochastok processors work with Llama tokenizer
- [ ] Verify affix detection works on Llama's BPE tokens

---

## 3. Data Preprocessing

### Preprocess with Llama Tokenizer

**New preprocessing runs needed:**
```bash
# Vanilla tokenization with Llama
python training/nemo/data/preprocess_data.py \
    --input data/chunks/chunk_*.jsonl \
    --output-prefix data/processed/llama_vanilla/chunk_ \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --tokenization-mode vanilla

# Stochastok with Llama
python training/nemo/data/preprocess_data.py \
    --input data/chunks/chunk_*.jsonl \
    --output-prefix data/processed/llama_stochastok/chunk_ \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --tokenization-mode stochastok \
    --stochastok-p 0.1

# Patok with Llama
python training/nemo/data/preprocess_data.py \
    --input data/chunks/chunk_*.jsonl \
    --output-prefix data/processed/llama_patok/chunk_ \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --tokenization-mode patok \
    --expand-prop 0.3 \
    --contract-prop 0.3
```

**Storage Requirements:**
- Gemma preprocessed: ~44GB (3 versions × ~14-15GB each)
- Llama preprocessed: ~44GB (similar size expected)
- **Total:** ~88GB for all preprocessed data

**Action Items:**
- [ ] Update `preprocess_data.py` to accept `--tokenizer-model` argument
- [ ] Batch preprocessing scripts for Llama (similar to `preprocess_all_*.sh`)
- [ ] Verify Megatron binary format is model-agnostic
- [ ] Test that processors work correctly with Llama's smaller vocabulary

**Time Estimate:** ~1-2 hours per tokenization method (parallel processing)

---

## 4. Model Configuration Changes

### Architecture Differences

| Parameter | Gemma 2 2B | Llama 3.1 8B |
|-----------|------------|--------------|
| Parameters | 2.6B | 8.0B |
| Layers | 26 | 32 |
| Hidden Size | 2304 | 4096 |
| Num Heads | 8 | 32 |
| MLP Hidden | 9216 | 14336 |
| Vocab Size | 256,128 | 128,000 |
| Context Length | 8192 | 8192 (up to 128K with RoPE) |

### Training Script Updates

**Current (Gemma-specific):**
```python
# In run_cpt.py
config = llm.GemmaConfig2(
    num_layers=26,
    hidden_size=2304,
    # ...
)
model = llm.GemmaModel2(config, tokenizer=tokenizer)
```

**Needs to become:**
```python
# Model-agnostic approach
if args.model_family == "gemma":
    config = llm.GemmaConfig2(...)
    model = llm.GemmaModel2(config, tokenizer=tokenizer)
elif args.model_family == "llama":
    config = llm.Llama31Config8B(...)  # or llm.LlamaConfig()
    model = llm.Llama31Model8B(config, tokenizer=tokenizer)
```

**Action Items:**
- [ ] Add `--model-family` argument to `run_cpt.py`
- [ ] Check NeMo 2.x API for Llama 3.1 support
- [ ] May need to use `llm.LlamaModel()` with custom config
- [ ] Test model initialization loads correctly

---

## 5. Training Configuration Adjustments

### Memory & Batch Size Considerations

**Gemma 2B (current):**
- Global batch size: 64
- Micro batch size: 1
- Gradient accumulation: 8 steps
- Memory per GPU: ~35GB (with distributed optimizer)

**Llama 8B (estimated):**
- Model is ~3x larger → ~3x memory
- With 8x A100 40GB GPUs:
  - **Option 1:** Reduce batch size to fit
    - Global batch size: 64 (keep same)
    - Micro batch size: 1 (keep same)
    - Should fit with bf16-mixed precision
  - **Option 2:** Use gradient accumulation
    - Global batch size: 128
    - Micro batch size: 1
    - Gradient accumulation: 16 steps
  - **Option 3:** Enable model parallelism
    - Tensor parallel: 2 (split model across 2 GPUs)
    - Effective GPUs: 4 pairs = 4-way data parallel

**Recommended Configuration for Llama 8B:**
```bash
# Conservative approach - should fit in memory
--global-batch-size 64 \
--micro-batch-size 1 \
--devices 8 \
--seq-length 512 \
--gradient-accumulation-steps 8
```

**Action Items:**
- [ ] Test memory requirements with dry run
- [ ] May need to enable tensor parallelism if OOM
- [ ] Monitor GPU memory during initial training steps
- [ ] Adjust if needed based on actual usage

### Training Hyperparameters

Keep same as Gemma for fair comparison:
- Learning rate: 1e-4
- Min LR: 1e-5
- Warmup steps: 100
- Max steps: 5000
- Optimizer: Adam (β1=0.9, β2=0.95)
- Weight decay: 0.1

**No changes needed** - these should transfer well.

---

## 6. Training Scripts & Commands

### Create Llama-specific Training Script

**New script: `scripts/run_cpt_training_llama.sh`**
```bash
#!/bin/bash
# Llama 3.1 8B CPT Training Script

TOKENIZATION="${1:-vanilla}"
MAX_STEPS="${2:-5000}"

case "$TOKENIZATION" in
    vanilla)
        DATA_DIR="/workspace/data/processed/llama_vanilla"
        CHECKPOINT_DIR="/logs/checkpoints/llama-3.1-8b-vanilla"
        ;;
    stochastok)
        DATA_DIR="/workspace/data/processed/llama_stochastok"
        CHECKPOINT_DIR="/logs/checkpoints/llama-3.1-8b-stochastok"
        ;;
    patok)
        DATA_DIR="/workspace/data/processed/llama_patok"
        CHECKPOINT_DIR="/logs/checkpoints/llama-3.1-8b-patok"
        ;;
esac

# Generate data paths
DATA_PATHS=""
for i in $(seq -w 1 20); do
    DATA_PATHS="$DATA_PATHS ${DATA_DIR}/chunk_00${i}"
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \
    --model-family llama \
    --data-path $DATA_PATHS \
    --max-steps "$MAX_STEPS" \
    --global-batch-size 64 \
    --micro-batch-size 1 \
    --devices 8 \
    --seq-length 512 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval 1000 \
    --resume-from /workspace/checkpoints/nemo/llama-3.1-8b
```

**Action Items:**
- [ ] Create `run_cpt_training_llama.sh` script
- [ ] Create `run_all_trainings_llama.sh` for sequential runs
- [ ] Test with short run first (100 steps)

---

## 7. Checkpoint Conversion & Evaluation

### Convert Llama Checkpoints to HF Format

**Same process as Gemma:**
```bash
# After training completes
./run_in_docker.sh python scripts/convert_megatron_to_hf.py \
    --megatron-path /workspace/logs/checkpoints/llama-3.1-8b-vanilla/val_loss=X.XX-step=5000-consumed_samples=320000.0 \
    --output-path /workspace/checkpoints/hf/llama-3.1-8b-vanilla-step5000 \
    --model-type llama \
    --base-model meta-llama/Llama-3.1-8B
```

**Action Items:**
- [ ] Update `convert_megatron_to_hf.py` to support Llama architecture
- [ ] Handle Llama-specific weight naming (different from Gemma)
- [ ] Test conversion produces loadable HF checkpoint

### Evaluation

**Same benchmarks, same evaluation script:**
```bash
python scripts/evaluate_converted_models.py \
    --model-path checkpoints/hf/llama-3.1-8b-vanilla-step5000 \
    --model-name llama-vanilla-step5000 \
    --benchmarks pacute hierarchical cute langgame multi-digit-addition
```

**No changes needed** - evaluation is model-agnostic.

---

## 8. Implementation Checklist

### Phase 1: Setup & Conversion (Est: 1-2 hours)
- [ ] Request Llama 3.1 access on HuggingFace (if not already)
- [ ] Download Llama 3.1 8B base model
- [ ] Test/update `convert_hf_to_nemo.py` for Llama
- [ ] Convert Llama to NeMo format
- [ ] Verify converted checkpoint loads

### Phase 2: Data Preprocessing (Est: 3-4 hours)
- [ ] Update `preprocess_data.py` with `--tokenizer-model` arg
- [ ] Create batch preprocessing scripts for Llama
- [ ] Preprocess vanilla data (1-2 hours)
- [ ] Preprocess stochastok data (1-2 hours)
- [ ] Preprocess patok data (1-2 hours)
- [ ] Verify all preprocessed files exist and are valid

### Phase 3: Training Script Updates (Est: 2-3 hours)
- [ ] Add `--model-family` argument to `run_cpt.py`
- [ ] Implement Llama model initialization
- [ ] Test training script with 100-step run
- [ ] Verify checkpoints save correctly
- [ ] Monitor memory usage, adjust batch size if needed
- [ ] Create Llama-specific training wrapper scripts

### Phase 4: Full Training (Est: 5-8 hours)
- [ ] Train Llama vanilla (1.5-2.5 hours)
- [ ] Train Llama stochastok (1.5-2.5 hours)
- [ ] Train Llama patok (1.5-2.5 hours)
- [ ] Convert all checkpoints to HF format

### Phase 5: Evaluation (Est: 6-9 hours)
- [ ] Evaluate all 3 Llama models on 5 benchmarks (2-3 hours each)
- [ ] Compare results with Gemma baseline
- [ ] Analyze cross-architecture consistency

---

## 9. Resource Requirements

### Disk Space
- Llama base model (HF): ~16GB
- Llama NeMo checkpoint: ~10GB
- Preprocessed data (3 versions): ~45GB
- Training checkpoints (3 models × 5 checkpoints): ~75GB
- **Total additional:** ~146GB

**Current available:** 130GB → **Need to free ~16GB or expand storage**

### GPU Time
- Preprocessing: ~3-4 hours
- Training: ~5-8 hours (3 models)
- Evaluation: ~6-9 hours
- **Total:** ~14-21 hours

### Cost Estimate (if on cloud)
- 8x A100 40GB: ~$20-30/hour
- Total time: ~14-21 hours
- **Estimated cost:** $280-630

---

## 10. Key Differences to Watch

### Tokenizer Behavior
- Llama's BPE may tokenize Filipino differently than Gemma's SentencePiece
- Affix detection might behave differently with Llama tokens
- Need to verify patok processor identifies morpheme boundaries correctly

### Training Dynamics
- Larger model (8B) may:
  - Learn slower (more parameters to update)
  - Learn faster (more capacity)
  - Benefit more/less from morphology-aware tokenization
- Monitor loss curves to compare with Gemma

### Performance Comparison
- Key question: Does patok improve Llama by similar margin as Gemma?
- If yes → strong evidence for generalization
- If no → investigate why (tokenizer? model size? architecture?)

---

## 11. Rollback Plan

If Llama adaptation encounters issues:

**Option A: Quick fixes**
- Memory issues → reduce batch size / enable tensor parallelism
- Preprocessing issues → debug tokenizer integration
- Training instability → adjust learning rate

**Option B: Defer Llama work**
- Focus on completing Gemma analysis first
- Llama can be future work if time-constrained
- Current Gemma results still valuable on their own

---

## Next Steps

**Immediate (while Gemma trains):**
1. Download Llama 3.1 8B model
2. Test HF → NeMo conversion
3. Update preprocessing scripts

**After Gemma completes:**
1. Decide: Continue with Llama or focus on Gemma analysis?
2. If continuing: Start Llama preprocessing
3. If deferring: Focus on Gemma evaluation and paper writing

---

## Notes
- This plan assumes NeMo 2.x supports Llama 3.1 natively
- If not, may need to use earlier Llama 3.0 or add custom model definition
- Check NeMo documentation/examples for Llama 3.1 support before starting
