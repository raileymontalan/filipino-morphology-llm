# Benchmark Formats

This document describes the format (MCQ vs GEN) for each benchmark and their corresponding dataset files.

## Format Types

- **MCQ (Multiple Choice Question)**: Model chooses from options based on log probabilities
- **GEN (Generative)**: Model generates the answer directly

## Benchmark Overview

| Benchmark | MCQ | GEN | Dataset Files | Samples |
|-----------|-----|-----|---------------|---------|
| **PACUTE** | ✓ | ✓ | affixation_mcq.jsonl, affixation_gen.jsonl | 140 |
| | | | composition_mcq.jsonl, composition_gen.jsonl | 900/500 |
| | | | manipulation_mcq.jsonl, manipulation_gen.jsonl | 800 |
| | | | syllabification_mcq.jsonl, syllabification_gen.jsonl | 400 |
| **CUTE** | ✗ | ✓ | cute_gen.jsonl | 1,400 (100 per task × 14) |
| **LangGame** | ✓ | ✓ | langgame_mcq.jsonl, langgame_gen.jsonl | 1,000 |
| **Multi-digit Addition** | ✓ | ✓ | multi_digit_addition_mcq.jsonl, multi_digit_addition_gen.jsonl | 1,000 |

## File Naming Convention

All benchmark files follow this naming pattern:
```
{benchmark_name}_{format}.jsonl
```

Where:
- `benchmark_name`: pacute category, cute, langgame, multi_digit_addition
- `format`: `mcq` or `gen`

Note: Train splits have been removed - only evaluation/test data is provided.

### Examples:
- `affixation_mcq.jsonl` - PACUTE Affixation in MCQ format
- `cute_gen.jsonl` - CUTE in generative format
- `langgame_mcq.jsonl` - LangGame in MCQ format
- `multi_digit_addition_gen.jsonl` - Multi-digit addition in generative format

## JSON Format

### MCQ Format
```json
{
  "question": "Which option contains 'a'?",
  "answer": " cat",
  "options": [" cat", " dog", " bird", " fish"]
}
```
Note: First option in `options` array is always the correct answer.

### GEN Format
```json
{
  "question": "What is 2+2?",
  "answer": "4"
}
```

## Registry Keys

Use these keys with `load_benchmark()`:

### PACUTE
- `pacute` or `pacute-mcq` - All categories, MCQ format (default)
- `pacute-gen` - All categories, GEN format
- `pacute-affixation` or `pacute-affixation-mcq` - Affixation, MCQ
- `pacute-affixation-gen` - Affixation, GEN
- `pacute-composition` or `pacute-composition-mcq` - Composition, MCQ
- `pacute-composition-gen` - Composition, GEN
- `pacute-manipulation` or `pacute-manipulation-mcq` - Manipulation, MCQ
- `pacute-manipulation-gen` - Manipulation, GEN
- `pacute-syllabification` or `pacute-syllabification-mcq` - Syllabification, MCQ
- `pacute-syllabification-gen` - Syllabification, GEN

### CUTE
- `cute` or `cute-gen` - Generative format (1,400 samples)

### LangGame
- `langgame` or `langgame-mcq` - MCQ format (default)
- `langgame-gen` - GEN format

### Multi-digit Addition
- `multi-digit-addition` or `multi-digit-addition-gen` - GEN format (default)
- `multi-digit-addition-mcq` - MCQ format

## Default Benchmarks for Evaluation

When running `python scripts/run_evaluation.py` without specifying benchmarks, these are evaluated by default:
1. `pacute` - 2,240 samples (MCQ)
2. `cute` - 1,400 samples (GEN)
3. `langgame` - 1,000 samples (MCQ)

## Subsampling

For efficiency, some benchmarks are subsampled:
- **CUTE**: 100 samples per task (from 1,000) = 1,400 total
- **Multi-digit Addition MCQ**: 1k train, 1k val (subsampled from 810k/90k GEN datasets)
- **Multi-digit Addition GEN**: Full datasets available (810k train, 90k val)

The loader applies additional subsampling during evaluation (default: 1,000 samples for multi-digit addition).

## Evaluation Mode

The `run_evaluation.py` script supports filtering benchmarks by format:

```bash
# Evaluate only MCQ benchmarks
python scripts/run_evaluation.py --models gpt2 --eval-mode mcq

# Evaluate only generative benchmarks
python scripts/run_evaluation.py --models gpt2 --eval-mode gen

# Evaluate both (default)
python scripts/run_evaluation.py --models gpt2 --eval-mode both
```

When using `--eval-mode mcq` or `--eval-mode gen`, benchmarks that don't match the specified format will be automatically skipped.

## Generation Scripts

To regenerate benchmark variants:
```bash
python scripts/generate_benchmark_variants.py
```

This will:
1. Create LangGame GEN versions from MCQ
2. Create Multi-digit Addition MCQ versions with distractor options (limited to 2k train, 1k val)
3. Save CUTE locally from HuggingFace (with subsampling)
