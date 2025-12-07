# Benchmark Evaluation Guide

## Overview

This directory contains scripts for evaluating language models on Filipino morphology and character understanding benchmarks.

## Benchmarks

### 1. PACUTE (Philippine Annotated Corpus for Understanding Tagalog Entities)
- **Total**: 560 tasks across 4 categories
- **Affixation** (140 tasks): Filipino affix identification and application
- **Composition** (180 tasks): Character counting and word formation
- **Manipulation** (160 tasks): Character operations (insert, delete, swap)
- **Syllabification** (80 tasks): Syllable counting and extraction

### 2. CUTE (Character Understanding Test Evaluation)
- **Total**: 14,000 tasks across 14 types
- Character-level: spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char
- Word-level: contains_word, ins_word, del_word, swap_word, sub_word
- Semantic/Orthographic: orth, sem

### 3. LangGame (Subword Understanding)
- Novel subword understanding benchmark
- 6 question types: most/contains/starts/ends/longest/shortest
- Tests understanding of token composition

## Models

### GPT-2 Family
- `gpt2` (124M parameters) - PT
- `gpt2-medium` (355M) - PT
- `gpt2-large` (774M) - PT

### Qwen Family
- `qwen-2.5-0.5b` - PT
- `qwen-2.5-0.5b-it` - IT
- `qwen-2.5-1.5b` - PT
- `qwen-2.5-1.5b-it` - IT
- `qwen-2.5-3b` - PT
- `qwen-2.5-3b-it` - IT

### Llama Family
- `llama-3.2-1b` - PT
- `llama-3.2-1b-it` - IT
- `llama-3.2-3b` - PT
- `llama-3.2-3b-it` - IT

### Gemma Family
- `gemma-2b` - PT
- `gemma-2b-it` - IT
- `gemma-7b` - PT
- `gemma-7b-it` - IT

### Cerebras GPT (Open Source GPT)
- `cerebras-gpt-111m` - PT
- `cerebras-gpt-256m` - PT
- `cerebras-gpt-590m` - PT
- `cerebras-gpt-1.3b` - PT

**PT** = Pre-trained | **IT** = Instruction-tuned

## Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Path Confidence**: Average softmax probability on correct answer
- **Normalized Accuracy**: Accuracy normalized to account for random chance

## Usage

### Quick Test (10 samples)
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute \
    --max-samples 10
```

### Single Model on All Benchmarks
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute cute
```

### Multiple Models
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 qwen-2.5-0.5b cerebras-gpt-111m \
    --benchmarks pacute cute
```

### Full Evaluation (All Models, All Benchmarks)
```bash
bash scripts/run_eval_batch.sh
```

### Custom Output Directory
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute \
    --output-dir my_results
```

## Output Format

Results are saved as JSON with this structure:

```json
{
  "model_name": {
    "hf_model_name": "gpt2",
    "model_type": "pt",
    "benchmarks": {
      "pacute": {
        "num_samples": 560,
        "accuracy": 0.4521,
        "f1_score": 0.4521,
        "precision": 0.4521,
        "recall": 0.4521,
        "path_confidence": 0.3124,
        "normalized_accuracy": 0.2695
      }
    }
  }
}
```

## Requirements

```bash
pip install torch transformers tqdm datasets
```

## Implementation Details

### MCQ Evaluation
- Models are evaluated using log-probability scoring
- For each question, we compute log P(answer | question) for all options
- The option with highest log probability is selected
- Metrics are computed by comparing predictions to ground truth

### Model Loading
- Models are loaded from HuggingFace using `AutoModelForCausalLM`
- FP16 precision is used on GPU for efficiency
- Models run in evaluation mode (no dropout)

### Benchmarks
- PACUTE: Filipino-specific morphology tasks (MCQ format)
- CUTE: English character understanding (generative format, adapted to MCQ)
- LangGame: Subword understanding (MCQ format)

## Notes

### LangGame Data
LangGame requires pre-generated data files in `data/data_as_datasets/langgame/`. If not available, you can:
1. Generate using StochasTok data generation scripts
2. Download from the StochasTok repository
3. Skip LangGame and evaluate on PACUTE and CUTE only

### GPU Memory
Larger models (7B+) require significant GPU memory:
- 7B models: ~14GB VRAM (FP16)
- 3B models: ~6GB VRAM (FP16)
- 1B models: ~2GB VRAM (FP16)

Use `--device cpu` if GPU memory is insufficient (will be slower).

### Model Access
Some models (Llama, Gemma) may require HuggingFace authentication:
```bash
huggingface-cli login
```

## Example Results

Expected performance ranges (approximate):

| Model Size | PACUTE Accuracy | CUTE Accuracy |
|------------|----------------|---------------|
| <500M | 30-45% | 25-40% |
| 500M-2B | 40-55% | 35-50% |
| 2B-7B | 50-65% | 45-60% |
| 7B+ | 60-75% | 55-70% |

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install torch transformers
```

### CUDA out of memory
```bash
# Use smaller models or CPU
python scripts/run_benchmark_evaluation.py --models gpt2 --device cpu
```

### Model not found
Some models require authentication. Log in to HuggingFace:
```bash
huggingface-cli login
```

### LangGame FileNotFoundError
LangGame data not available. Skip it:
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute cute  # Skip langgame
```
