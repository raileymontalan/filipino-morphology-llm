# Evaluation Guide

Complete guide for generating benchmarks and evaluating models on Filipino morphology tasks.

## Quick Start

```bash
# 1. Generate all benchmarks
python scripts/generate_evaluation_datasets.py

# 2. Evaluate a model
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute

# 3. View results
cat results/benchmark_evaluation/*.json
```

---

## Benchmark Overview

Total: **15,023 evaluation tasks** across 4 benchmark suites

| Benchmark | Total Tasks | Description |
|-----------|-------------|-------------|
| **PACUTE** | 11,225 | Filipino morphology (affixes, composition, manipulation, syllabification) |
| **Hierarchical** | 1,798 | 6 diagnostic levels (character → morphology) |
| **LangGame** | 3,000 | Subword understanding (word games) |
| **Math** | 3,000 | Multi-digit addition (numerical reasoning) |

---

## 1. PACUTE Benchmark

**P**ilipino **A**ffix and **C**haracter-Level **U**nderstanding of **T**okens **E**valuation

### Tasks (11,225 total)

#### Affixation (280 tasks)
Tests understanding of Filipino morphology:
- Identify prefixes: "What is the prefix in 'kumain'?" → "k-"
- Identify infixes: "What is the infix in 'kumain'?" → "-um-"
- Identify suffixes: "What is the suffix in 'matulog'?" → "-log"
- Apply affixes: "Add 'mag-' to 'luto'" → "magluto"

#### Composition (3,905 tasks)
Tests character-level understanding:
- Count characters: "How many letters in 'kumain'?" → 6
- Count specific chars: "How many 'a's in 'banana'?" → 3
- Identify diacritics: "Does 'kumâin' have diacritics?" → Yes
- Normalize text: "Remove diacritics from 'kumâin'" → "kumain"

#### Manipulation (5,120 tasks)
Tests character operations:
- Insert: "Insert 'l' at position 3 in 'kumain'" → "kulmain"
- Delete: "Delete character at position 2 in 'kumain'" → "kuain"
- Swap: "Swap positions 1 and 3 in 'kumain'" → "mukakinan"
- Replace: "Replace 'm' with 'l' in 'kumain'" → "kulain"

#### Syllabification (1,280 tasks)
Tests syllable understanding:
- Count syllables: "How many syllables in 'kumain'?" → 3
- Extract syllables: "What is the first syllable in 'kumain'?" → "ku"
- Identify stress: "Where is the stress in 'kumáin'?" → 2
- Handle reduplication: "Reduplicate 'takbo'" → "tatakbo"

### Generate PACUTE

```bash
python scripts/generate_evaluation_datasets.py  # Generates all benchmarks

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

### Tasks (3,000 tasks)

6 question types:
- **most**: "Which word has the most 'a's?"
- **contains**: "Which word contains 'ing'?"
- **starts**: "Which word starts with 'ka'?"
- **ends**: "Which word ends with 'an'?"
- **longest**: "Which word is longest?"
- **shortest**: "Which word is shortest?"

### Generate LangGame

```bash
python src/evaluation/datasets/scripts/generate_langgame_benchmark.py
```

Output: `data/benchmarks/langgame_{train,val}.jsonl`

---

## 4. Math Benchmark

Tests numerical reasoning with multi-digit addition.

### Tasks (3,000 tasks)

3-digit addition: "123 + 456 = ?" → "579"

### Generate Math

```bash
python src/evaluation/datasets/scripts/generate_math_benchmark.py
```

Output: `data/benchmarks/multi_digit_addition_{train,val}.jsonl`

---

## Evaluation Workflow

### Step 1: Generate All Benchmarks

```bash
python scripts/generate_evaluation_datasets.py
```

This creates all benchmark files in `data/benchmarks/`.

### Step 2: Evaluate Models

```bash
# Evaluate single model on all benchmarks
python scripts/run_evaluation.py --models gpt2

# Evaluate multiple models
python scripts/run_evaluation.py --models gpt2 google/gemma-3-1b-pt

# Evaluate specific benchmarks only
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute hierarchical

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
| `--benchmarks` | All | Benchmarks to run (pacute, hierarchical, langgame, math) |
| `--max-samples` | None | Limit samples per benchmark (for testing) |
| `--batch-size` | 8 | Inference batch size |
| `--device` | auto | Device: cuda, cpu, or auto |

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
python scripts/generate_evaluation_datasets.py
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
