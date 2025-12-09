# Benchmark Generation Scripts

This directory contains all scripts for generating evaluation benchmarks for the Filipino Morphology LLM project.

## Overview

All benchmark generation scripts have been consolidated here from various locations:
- Previously in `tests/` (PACUTE benchmarks)
- Previously in `scripts/` (Hierarchical benchmarks)
- Previously in `training/stochastok/data_processing/` (LangGame, Multi-digit Addition)

## Available Benchmarks

### 1. PACUTE Benchmarks
**Script:** `generate_pacute_benchmarks.py`

Generates core Filipino morphological understanding tasks:
- **Affixation** (280 MCQ + 280 Gen): Understanding prefixes, suffixes, infixes
- **Composition** (280 MCQ + 280 Gen): Combining morphemes to form words
- **Manipulation** (320 MCQ + 320 Gen): String operations on words
- **Syllabification** (160 MCQ + 160 Gen): Breaking words into syllables

**Total:** 1,040 MCQ + 1,040 generative tasks

**Usage:**
```bash
cd /path/to/filipino-morphology-llm
python src/evaluation/benchmark_generation/generate_pacute_benchmarks.py
```

**Output:**
- `data/benchmarks/affixation_mcq.jsonl`
- `data/benchmarks/affixation_gen.jsonl`
- `data/benchmarks/composition_mcq.jsonl`
- `data/benchmarks/composition_gen.jsonl`
- `data/benchmarks/manipulation_mcq.jsonl`
- `data/benchmarks/manipulation_gen.jsonl`
- `data/benchmarks/syllabification_mcq.jsonl`
- `data/benchmarks/syllabification_gen.jsonl`

**Naming Convention:** All PACUTE benchmarks follow the pattern `{category}_{mcq|gen}.jsonl`

### 2. Hierarchical Benchmarks
**Script:** `generate_hierarchical_benchmark.py`

Generates multi-level linguistic understanding tasks across 6 hierarchical levels:
- **Level 0:** Character Recognition
- **Level 1:** Character Manipulation
- **Level 2:** Morpheme Decomposition
- **Level 3:** Morpheme Manipulation
- **Level 4:** Morpheme Composition
- **Level 5:** Complex Morphological Reasoning

**Usage:**
```bash
cd /path/to/filipino-morphology-llm
python src/evaluation/benchmark_generation/generate_hierarchical_benchmark.py
```

**Output:**
- `data/benchmarks/hierarchical_mcq.jsonl`
- `data/benchmarks/hierarchical_gen.jsonl`

### 3. LangGame Dataset
**Script:** `generate_langgame_benchmark.py`

Generates language reasoning tasks for testing linguistic pattern recognition.

**Usage:**
```bash
cd /path/to/filipino-morphology-llm
python src/evaluation/datasets/scripts/generate_langgame_benchmark.py
```

**Output:**
- `data/benchmarks/langgame_mcq.jsonl` (1,000 samples)

**Note:** Use `scripts/generate_benchmark_variants.py` to create the generative version (`langgame_gen.jsonl`).

### 4. Multi-digit Addition Dataset
**Script:** `generate_math_benchmark.py`

Generates mathematical reasoning benchmarks using multi-digit addition problems.

**Usage:**
```bash
cd /path/to/filipino-morphology-llm
python src/evaluation/datasets/scripts/generate_math_benchmark.py
```

**Output:**
- `data/benchmarks/multi_digit_addition_gen.jsonl` (1,000 samples)

**Note:** Use `scripts/generate_benchmark_variants.py` to create the MCQ version (`multi_digit_addition_mcq.jsonl`).

## Generate All Benchmarks

To generate all benchmarks at once (where possible):

```bash
cd /path/to/filipino-morphology-llm
python scripts/generate_evaluation_datasets.py
```

This master script will:
1. Generate PACUTE benchmarks
2. Generate Hierarchical benchmarks
3. Attempt LangGame (skip if dependencies missing)
4. Attempt Multi-digit Addition (skip if dependencies missing)

## Data Requirements

The scripts expect the following data files to exist:

### For PACUTE Benchmarks:
- `data/corpora/pacute_data/inflections.xlsx` (for affixation)
- `data/corpora/pacute_data/syllables.jsonl` (for composition, manipulation, syllabification)

### For Hierarchical Benchmarks:
- `data/corpora/pacute_data/syllables.jsonl`
- `data/corpora/affix_annotations.jsonl`

### For LangGame and Multi-digit Addition:
- Various stochastok components and model files
- See individual scripts for specific requirements

## Available Benchmarks Summary

| Benchmark | MCQ File | Gen File | MCQ Samples | Gen Samples |
|-----------|----------|----------|-------------|-------------|
| **Affixation** | `affixation_mcq.jsonl` | `affixation_gen.jsonl` | 140 | 140 |
| **Composition** | `composition_mcq.jsonl` | `composition_gen.jsonl` | 900 | 500 |
| **Manipulation** | `manipulation_mcq.jsonl` | `manipulation_gen.jsonl` | 800 | 800 |
| **Syllabification** | `syllabification_mcq.jsonl` | `syllabification_gen.jsonl` | 400 | 400 |
| **LangGame** | `langgame_mcq.jsonl` | `langgame_gen.jsonl` | 1,000 | 1,000 |
| **Multi-digit Addition** | `multi_digit_addition_mcq.jsonl` | `multi_digit_addition_gen.jsonl` | 1,000 | 1,000 |
| **CUTE** | - | `cute_gen.jsonl` | - | 1,400 |
| **TOTAL** | - | - | **4,240** | **5,240** |

**Naming Convention:** All benchmarks follow the pattern `{benchmark_name}_{mcq|gen}.jsonl`

## Output Location

All benchmarks are saved to two locations:

### JSONL Format (for evaluation)
`data/benchmarks/` - Human-readable JSONL files
- Each line is a JSON object representing one task
- Used for model evaluation and analysis

### Memmap Format (for efficient training)
`data/memmaps/` - Binary memory-mapped files
- Tokenized and padded sequences in `.bin` format
- Metadata in corresponding `.json` files
- Efficient for large-scale training

## Converting Benchmarks to Memmaps

All JSONL benchmarks can be converted to memory-mapped format:

```bash
cd /path/to/filipino-morphology-llm
python src/evaluation/benchmark_generation/convert_benchmarks_to_memmaps.py
```

This will:
1. Load all `*.jsonl` files from `data/benchmarks/`
2. Tokenize each file using a character-level tokenizer
3. Save as `.bin` (memmap) + `.json` (metadata) in `data/memmaps/`

**Note:** The default conversion uses a simple character-level tokenizer. For production training, you may want to modify the script to use your model's actual tokenizer.

### Manual Conversion

You can also convert specific files programmatically:

```python
from src.evaluation.benchmark_generation.conversion_utils import jsonl_to_memmap

# With your own tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model")

paths = jsonl_to_memmap(
    "data/benchmarks/mcq_affixation.jsonl",
    "data/memmaps/affixation",
    tokenizer,
    text_field="question"
)
```

## Evaluation

To use these benchmarks for evaluation, see:
- `src/evaluation/benchmarks/` - Evaluation loaders
- `scripts/run_evaluation.py` - Evaluation runner

## Migration Notes

- ✓ PACUTE generation consolidated from individual test files
- ✓ Hierarchical generation moved from `scripts/`
- ✓ LangGame and Multi-digit Addition copied from `training/stochastok/data_processing/`
- All scripts now output to standardized `data/benchmarks/` directory
- Path handling updated to use `pathlib` and be robust to execution location

## Troubleshooting

### Module Import Errors
If you encounter import errors, ensure you're running from the project root:
```bash
cd /path/to/filipino-morphology-llm
python -m src.evaluation.benchmark_generation.generate_pacute_benchmarks
```

### Missing Data Files
Ensure all required data files are present in `data/corpora/pacute_data/`.

### Stochastok Dependencies
For LangGame and Multi-digit Addition, you may need to activate the stochastok environment first.
