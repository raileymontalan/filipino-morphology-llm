# Evaluation Guide

Complete guide for generating benchmarks and evaluating models on Filipino morphology tasks.

## Quick Start

```bash
# 1. Generate all benchmarks
python scripts/generate_benchmarks.py

# 2. Evaluate a model
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute

# 3. View results
cat results/benchmark_evaluation/*.json
```

---

## Benchmark Overview

Total: **10,678 evaluation tasks** across multiple benchmark suites

| Benchmark | MCQ Tasks | GEN Tasks | Description |
|-----------|-----------|-----------|-------------|
| **PACUTE** | 2,240 | 1,840 | Filipino morphology (affixes, composition, manipulation, syllabification) |
| **Hierarchical** | 600 | 598 | Diagnostic tasks across 6 compositional levels |
| **CUTE** | - | 1,400 | Character Understanding Tasks Evaluation (14 task types) |
| **LangGame** | 1,000 | 1,000 | Subword understanding (word games) |
| **Multi-digit Addition** | 1,000 | 1,000 | Numerical reasoning |

**Format Types:**
- **MCQ (Multiple Choice)**: Log probability-based selection from 4 options
- **GEN (Generative)**: Free-form text generation with exact match scoring

---

## 1. PACUTE Benchmark

**P**ilipino **A**ffix and **C**haracter-Level **U**nderstanding of **T**okens **E**valuation

### Tasks (4,080 total)

#### Affixation (280 tasks: 140 MCQ + 140 GEN)
Tests understanding of Filipino morphology:
- Identify prefixes: "What is the prefix in 'kumain'?" → "k-"
- Identify infixes: "What is the infix in 'kumain'?" → "-um-"
- Identify suffixes: "What is the suffix in 'matulog'?" → "-log"
- Apply affixes: "Add 'mag-' to 'luto'" → "magluto"

#### Composition (1,400 tasks: 900 MCQ + 500 GEN)
Tests character-level understanding:
- Count characters: "How many letters in 'kumain'?" → 6
- Count specific chars: "How many 'a's in 'banana'?" → 3
- Identify diacritics: "Does 'kumâin' have diacritics?" → Yes
- Normalize text: "Remove diacritics from 'kumâin'" → "kumain"

#### Manipulation (1,600 tasks: 800 MCQ + 800 GEN)
Tests character operations:
- Insert: "Insert 'l' at position 3 in 'kumain'" → "kulmain"
- Delete: "Delete character at position 2 in 'kumain'" → "kuain"
- Swap: "Swap positions 1 and 3 in 'kumain'" → "mukakinan"
- Replace: "Replace 'm' with 'l' in 'kumain'" → "kulain"

#### Syllabification (800 tasks: 400 MCQ + 400 GEN)
Tests syllable understanding:
- Count syllables: "How many syllables in 'kumain'?" → 3
- Extract syllables: "What is the first syllable in 'kumain'?" → "ku"
- Identify stress: "Where is the stress in 'kumáin'?" → 2
- Handle reduplication: "Reduplicate 'takbo'" → "tatakbo"

### Generate PACUTE

```bash
python scripts/generate_benchmarks.py  # Generates all benchmarks

# Or generate PACUTE only
python src/evaluation/datasets/scripts/generate_pacute_benchmarks.py
```

Output: `data/benchmarks/{affixation,composition,manipulation,syllabification}_{mcq,gen}.jsonl`

---

## 2. Hierarchical Benchmark

Diagnostic tasks organized into 6 compositional levels to identify where models fail.

### Levels (1,798 total tasks)

| Level | Capability | Example Task | Dependency |
|-------|------------|--------------|------------|
| **0** | Character Recognition | "What is char 3 in 'kumain'?" → 'm' | None |
| **1** | Character Manipulation | "Delete char 3 in 'kumain'" → "kuain" | Level 0 |
| **2** | Morpheme Decomposition | "Extract root from 'kumain'" → "kain" | Level 0 |
| **3** | Morpheme Manipulation | "Change 'um' to 'mag' in 'kumain'" → "magkain" | Levels 1+2 |
| **4** | Morpheme Composition | "Combine 'ka-' + 'alis' + '-an'" → "kaalisan" | Level 2 |
| **5** | Complex Reasoning | "Apply focus markers to 'kain'" → "kainin" | Levels 2-4 |

### Diagnostic Use

Identify bottlenecks:
- ✅ Level 0 (95%) → Model can see characters
- ✅ Level 1 (75%) → Model can manipulate strings  
- ❌ Level 2 (40%) → **Bottleneck: morpheme boundaries**
- ❌ Level 3 (25%) → Expected failure (needs Level 2)

**Diagnosis:** Tokenization doesn't align with morphology  
**Solution:** Use affix-aware tokenization (Stochastok/Patok)

### Generate Hierarchical

```bash
python src/evaluation/datasets/scripts/generate_hierarchical_benchmark.py
```

Output: `data/benchmarks/hierarchical_{mcq,gen}.jsonl`

---

## 3. LangGame Benchmark

Tests subword understanding through word games.

### Tasks (2,000 tasks: 1,000 MCQ + 1,000 GEN)

6 question types:
- **most**: "Which word has the most 'a's?"
- **contains**: "Which word contains 'ing'?"
- **starts**: "Which word starts with 'ka'?"
- **ends**: "Which word ends with 'an'?"
- **longest**: "Which word is longest?"
- **shortest**: "Which word is shortest?"

**MCQ Format**: Select from 4 word options
**GEN Format**: Directly output the correct word

### Generate LangGame

```bash
python src/evaluation/datasets/scripts/generate_langgame_benchmark.py
```

Output: `data/benchmarks/langgame_mcq.jsonl` (1,000 samples)

Use `scripts/generate_benchmark_variants.py` to create generative version: `langgame_gen.jsonl`

---

## 4. CUTE Benchmark

**C**haracter **U**nderstanding **T**asks **E**valuation

Tests character-level understanding across diverse operations.

### Tasks (1,400 tasks: GEN only)

14 task types (100 samples each):
- **spell**: Spell out characters: "kumain" → "k-u-m-a-i-n"
- **spell_inverse**: Combine spelled chars: "k-u-m-a-i-n" → "kumain"
- **contains_char**: Check if char exists: "Does 'kumain' contain 'k'?" → "Yes"
- **contains_word**: Check if substring exists
- **orth**: Orthographic neighbors
- **sem**: Semantic similarity
- **ins_char**: Insert character at position
- **ins_word**: Insert word/substring
- **del_char**: Delete character at position
- **del_word**: Delete word/substring
- **sub_char**: Substitute character
- **sub_word**: Substitute word/substring
- **swap_char**: Swap two characters
- **swap_word**: Swap two words/substrings

### Generate CUTE

CUTE is loaded from HuggingFace and saved locally:

```bash
python scripts/generate_benchmark_variants.py
```

Output: `data/benchmarks/cute_gen.jsonl` (1,400 samples, subsampled to 100 per task type)

---

## 5. Multi-digit Addition Benchmark

Tests numerical reasoning with multi-digit addition.

### Tasks (2,000 tasks: 1,000 MCQ + 1,000 GEN)

3-digit addition: "123 + 456 = ?" → "579"

**MCQ Format**: Select from 4 numerical options with strategic distractors
**GEN Format**: Directly output the correct sum

Distractor strategies for MCQ:
- Off by small amounts (±1 to ±20)
- Digit manipulation (swapped, added, removed)
- Common arithmetic errors (carry errors, ±10, ±100)

### Generate Math

```bash
python src/evaluation/datasets/scripts/generate_math_benchmark.py
```

Output: `data/benchmarks/multi_digit_addition_gen.jsonl` (1,000 samples)

Use `scripts/generate_benchmark_variants.py` to create MCQ version: `multi_digit_addition_mcq.jsonl`

---

## Benchmark Format Variants

Many benchmarks are available in both MCQ and GEN formats. Use `scripts/generate_benchmark_variants.py` to convert between formats or generate missing variants:

### What it Does

1. **LangGame GEN**: Converts MCQ format to generative by removing options
2. **Multi-digit Addition MCQ**: Creates MCQ from GEN with strategic distractors
3. **CUTE GEN**: Downloads from HuggingFace and saves locally (subsampled)

### Usage

```bash
python scripts/generate_benchmark_variants.py
```

### Output

Creates/updates these files:
- `data/benchmarks/langgame_gen.jsonl` (1,000 samples)
- `data/benchmarks/multi_digit_addition_mcq.jsonl` (1,000 samples)
- `data/benchmarks/cute_gen.jsonl` (1,400 samples)

### Format Comparison

**MCQ (Multiple Choice Questions)**
- 4 options per question
- Model selects based on log probabilities
- More controlled evaluation
- Less sensitive to formatting

**GEN (Generative)**
- Free-form text generation
- Exact match scoring
- Tests true generation capability
- More sensitive to output format

---

## Evaluation Workflow

### Step 1: Generate All Benchmarks

```bash
# Generate base benchmarks
python scripts/generate_benchmarks.py

# Generate additional format variants (MCQ/GEN conversions)
python scripts/generate_benchmark_variants.py
```

This creates all benchmark files in `data/benchmarks/`:
- PACUTE: All categories in both MCQ and GEN formats
- LangGame: MCQ and GEN formats (1,000 samples each)
- Multi-digit Addition: MCQ and GEN formats (1,000 samples each)
- CUTE: GEN format only (1,400 samples, 100 per task type)

### Step 2: Evaluate Models

```bash
# Evaluate single model on all benchmarks
python scripts/run_evaluation.py --models gpt2

# Evaluate multiple models
python scripts/run_evaluation.py --models gpt2 google/gemma-3-1b-pt

# Evaluate specific benchmarks only
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute cute langgame

# Filter by evaluation mode (MCQ or GEN)
python scripts/run_evaluation.py --models gpt2 --eval-mode mcq    # Only MCQ benchmarks
python scripts/run_evaluation.py --models gpt2 --eval-mode gen    # Only GEN benchmarks
python scripts/run_evaluation.py --models gpt2 --eval-mode both   # All benchmarks (default)

# Limit samples for quick testing
python scripts/run_evaluation.py --models gpt2 --max-samples 100
```

### Step 3: View Results

```bash
# View JSON results
cat results/benchmark_evaluation/gpt2_pacute_results.json

# View summary
python scripts/analyze_results.py
```

---

## Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--models` | Required | Model IDs to evaluate (space-separated) |
| `--benchmarks` | pacute, cute, langgame | Benchmarks to run (or "all") |
| `--eval-mode` | both | Filter by format: mcq, gen, or both |
| `--max-samples` | None | Limit samples per benchmark (for testing) |
| `--batch-size` | 8 | Inference batch size |
| `--device` | auto | Device: cuda, cpu, or auto |

### Benchmark Format Filtering

The `--eval-mode` parameter allows you to filter benchmarks by their evaluation format:

- **mcq**: Only run Multiple Choice Question benchmarks (log probability-based selection)
- **gen**: Only run Generative benchmarks (text generation and exact match)
- **both**: Run all available benchmarks (default)

This is useful when you want to:
- Compare MCQ vs GEN performance on the same tasks
- Run quick evaluations on just one format
- Debug format-specific issues

---

## Output Structure

```
results/benchmark_evaluation/
├── gpt2_pacute_affixation_mcq.json       # Detailed results per task
├── gpt2_pacute_affixation_gen.json
├── gpt2_hierarchical_mcq.json
├── gpt2_summary.json                     # Summary statistics
└── comparison.json                       # Cross-model comparison
```

---

## Evaluation Metrics

### Multiple Choice Questions (MCQ)
- **Accuracy**: Percentage of correct answers
- **Per-option analysis**: Distribution of selected options

### Generative Tasks (Gen)
- **Exact match**: Percentage of exact matches
- **Edit distance**: Average edit distance from correct answer
- **Partial credit**: Fuzzy matching score

### Hierarchical Analysis
- **Per-level accuracy**: Performance at each level
- **Dependency analysis**: Cascading failures
- **Bottleneck identification**: Which level causes failures

---

## Comparing Models

### Example: Baseline vs. Stochastok

```bash
# Evaluate baseline model
python scripts/run_evaluation.py \
    --models /path/to/baseline/checkpoint.nemo \
    --benchmarks pacute hierarchical

# Evaluate stochastok model
python scripts/run_evaluation.py \
    --models /path/to/stochastok/checkpoint.nemo \
    --benchmarks pacute hierarchical

# Compare results
python scripts/compare_models.py \
    --model1 results/baseline_summary.json \
    --model2 results/stochastok_summary.json
```

---

## Batch Evaluation (PBS)

For large-scale evaluation on cluster:

```bash
# Submit batch evaluation job
qsub jobs/run_evaluation_batch.pbs

# Or test with single model first
qsub jobs/run_evaluation_test.pbs
```

Edit `jobs/run_evaluation_batch.pbs` to specify:
- `MODELS`: Space-separated model paths
- `BENCHMARKS`: Which benchmarks to run
- `MAX_SAMPLES`: Limit for testing

---

## Understanding Results

### Good Performance Profile
- PACUTE Affixation: 70-90% (understands morphology)
- PACUTE Manipulation: 60-80% (can manipulate strings)
- Hierarchical Level 0-1: 85-95% (sees characters)
- Hierarchical Level 2: 60-80% (understands morphemes)

### Poor Performance Profile  
- PACUTE Affixation: <50% (doesn't understand morphology)
- Hierarchical Level 2: <40% (morpheme boundary issues)
- Cascading failures in Levels 3-5

### Expected Improvements with Stochastok/Patok
- **+10-20%** on PACUTE Affixation
- **+15-25%** on Hierarchical Level 2
- **+5-10%** on PACUTE Manipulation
- Reduced cascading failures in Levels 3-5

---

## Troubleshooting

### Issue: "Benchmark files not found"
**Solution:** Generate benchmarks first:
```bash
python scripts/generate_benchmarks.py
```

### Issue: Out of memory during evaluation
**Solutions:**
- Reduce `--batch-size`
- Use smaller `--max-samples`
- Evaluate on CPU: `--device cpu`

### Issue: Evaluation too slow
**Solutions:**
- Increase `--batch-size`
- Use GPU: `--device cuda`
- Limit samples: `--max-samples 1000`

### Issue: Model loading failed
**Solution:** Check model path and ensure it's accessible:
```bash
ls /path/to/model/checkpoint.nemo
```

---

## Custom Benchmarks

### Adding New Tasks

1. Create generator in `src/evaluation/datasets/generators/`
2. Add to `scripts/generate_evaluation_datasets.py`
3. Create evaluator in `src/evaluation/evaluators/`
4. Add to `scripts/run_evaluation.py`

Example structure:
```python
# Generator
def generate_my_benchmark():
    tasks = []
    # Generate tasks...
    return tasks

# Evaluator
def evaluate_my_benchmark(model, tasks):
    results = []
    # Evaluate...
    return results
```

---

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_analysis.py` | Detailed analysis of results |
| `scripts/compare_models.py` | Cross-model comparison |
| `scripts/visualize_results.py` | Generate plots |
| `scripts/demo_hierarchical_tasks.py` | Demo hierarchical evaluation |

---

## Tips & Best Practices

✅ **Generate benchmarks once** - they don't change per model  
✅ **Test with `--max-samples 100` first** for quick validation  
✅ **Use batch evaluation** for multiple models  
✅ **Focus on hierarchical levels** to diagnose issues  
✅ **Compare across tokenization modes** to measure impact  
✅ **Save results with clear names** for tracking experiments  

---

For research background, see `docs/RESEARCH.md`.  
For training models to evaluate, see `docs/TRAINING.md`.
