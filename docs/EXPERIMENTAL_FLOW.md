# Experimental Flow Diagram

This document visualizes the complete experimental pipeline from data to results.

---

## High-Level Overview

```
Raw Data → Preprocessing → Training → Evaluation → Analysis → Paper
   ↓            ↓             ↓           ↓           ↓         ↓
seapile    Tokenization   NeMo CPT   Benchmarks  Metrics   Findings
```

---

## Detailed Pipeline

### Phase 1: Data Preparation

```
data/corpora/seapile-v2.jsonl (7.4GB)
           ↓
    [Split into 20 chunks]
           ↓
data/chunks/chunk_000*.jsonl
           ↓
┌──────────┴──────────┬──────────────┐
│                     │              │
│ Baseline            │ StochasTok   │ Patok
│ (vanilla BPE)       │ (expand 10%) │ (expand 30% + contract 30%)
│                     │              │
│ GPT-2 tokenizer     │ + expand()   │ + expand_contract()
│                     │              │   + affix_preference=0.7
↓                     ↓              ↓
Megatron binary     Megatron       Megatron
data/processed/     + expansion    + Patok
*.bin + *.idx       metadata       metadata
```

### Phase 2: Training (3 Parallel Experiments)

```
Experiment 1: BASELINE
├── Model: Gemma 3 1B (google/gemma-3-1b-pt)
├── Data: data/processed/ (vanilla tokenization)
├── Training: training/nemo/run_cpt.py
├── Job: jobs/run_cpt.pbs
├── Steps: 10,000
├── Checkpoints: nemo_experiments/baseline/checkpoints/
└── Logs: logs/baseline_*.log

Experiment 2: STOCHASTOK
├── Model: Gemma 3 1B (same initialization)
├── Data: data/processed_stochastok/ (expanded tokens)
├── Training: training/nemo/run_cpt.py --tokenizer stochastok
├── Job: jobs/run_cpt_stochastok.pbs
├── Steps: 10,000
├── Checkpoints: nemo_experiments/stochastok/checkpoints/
└── Logs: logs/stochastok_*.log

Experiment 3: PATOK
├── Model: Gemma 3 1B (same initialization)
├── Data: data/processed_patok/ (Patok-processed)
├── Training: training/nemo/run_cpt.py --tokenizer patok
├── Job: jobs/run_cpt_patok.pbs
├── Steps: 10,000
├── Checkpoints: nemo_experiments/patok/checkpoints/
└── Logs: logs/patok_*.log
```

### Phase 3: Evaluation (5 Benchmarks × 3 Models)

```
For each model checkpoint:
├── nemo_experiments/baseline/checkpoints/step_10000.ckpt
├── nemo_experiments/stochastok/checkpoints/step_10000.ckpt
└── nemo_experiments/patok/checkpoints/step_10000.ckpt

Run 5 benchmark suites:

1. PACUTE (1,040 tasks)
   ├── data/benchmarks/mcq_affixation.jsonl → Accuracy
   ├── data/benchmarks/mcq_composition.jsonl → Accuracy
   ├── data/benchmarks/mcq_manipulation.jsonl → Accuracy
   └── data/benchmarks/mcq_syllabification.jsonl → Accuracy

2. Hierarchical (1,196 tasks)
   ├── Level 0: Character recognition → Accuracy
   ├── Level 1: Character manipulation → Accuracy
   ├── Level 2: Morpheme decomposition → Accuracy
   ├── Level 3: Morpheme manipulation → Accuracy
   ├── Level 4: Morpheme composition → Accuracy
   └── Level 5: Complex reasoning → Accuracy

3. LangGame (1,000 tasks)
   ├── Contains → Accuracy
   ├── Starts with → Accuracy
   ├── Ends with → Accuracy
   ├── Longest → Accuracy
   ├── Shortest → Accuracy
   └── Most of letter → Accuracy

4. Multi-Digit Addition (1,000 tasks)
   └── 3-digit addition → Exact match accuracy

5. Morphological Analysis (472 annotated words)
   ├── MorphScore → 0-1 (boundary alignment)
   ├── Boundary F1 → Precision/Recall
   ├── Fragmentation → Tokens per morpheme
   └── Affix coverage → % in vocabulary
```

### Phase 4: Statistical Analysis

```
Results aggregation:
├── results/baseline/
│   ├── pacute_scores.json
│   ├── hierarchical_scores.json
│   ├── langgame_scores.json
│   ├── math_scores.json
│   └── morphological_metrics.json
│
├── results/stochastok/
│   └── [same structure]
│
└── results/patok/
    └── [same structure]

Statistical tests:
├── Paired t-test (Patok vs Baseline)
├── Paired t-test (Patok vs StochasTok)
├── Effect size (Cohen's d)
├── Confidence intervals
└── Significance: p < 0.05

Visualizations:
├── Bar charts (accuracy by benchmark)
├── Heatmaps (accuracy by task type)
├── Level-wise performance (hierarchical)
├── MorphScore comparison
└── Error analysis
```

---

## Code Mapping to Pipeline

### Data Preparation
```python
# Baseline preprocessing
training/nemo/data/preprocess_data.py

# StochasTok preprocessing (to be implemented)
training/nemo/data/preprocess_data_stochastok.py
# Uses: src/tokenization/stochastok_processor.py

# Patok preprocessing (to be implemented)
training/nemo/data/preprocess_data_patok.py
# Uses: src/tokenization/patok_processor.py
```

### Training
```python
# All experiments use:
training/nemo/run_cpt.py
# With different data paths and tokenizer configs
```

### Evaluation
```python
# PACUTE evaluation
src/evaluation/benchmarks/mcqs/mcq_evaluator.py
# Loads: data/benchmarks/mcq_*.jsonl

# Hierarchical evaluation
src/evaluation/hierarchical_analysis.py
# Loads: data/benchmarks/hierarchical_*.jsonl

# LangGame evaluation (to be implemented)
src/evaluation/benchmarks/mcqs/benchmarks/langgame.py

# Math evaluation (to be implemented)
src/evaluation/benchmarks/generation_evaluator_math.py

# Morphological analysis
src/analysis/morphological_metrics.py
# Loads: data/corpora/affix_annotations.jsonl
```

### Analysis
```python
# Aggregate results
scripts/analyze_results.py

# Generate comparison tables
scripts/compare_results.py

# Generate paper figures
scripts/generate_paper_tables.py
```

---

## Timeline Estimate

```
Week 1-2: Data Preprocessing
├── Complete baseline preprocessing (20 chunks)
├── Implement StochasTok preprocessing for NeMo
├── Implement Patok preprocessing for NeMo
└── Validate all three preprocessed datasets

Week 3-4: Training
├── Train baseline model (10K steps, ~3 days)
├── Train StochasTok model (10K steps, ~3 days)
├── Train Patok model (10K steps, ~3 days)
└── Verify checkpoints and training logs

Week 5-6: Evaluation
├── Implement evaluation pipeline
├── Run PACUTE on all 3 models
├── Run Hierarchical on all 3 models
├── Run LangGame on all 3 models
├── Run Math on all 3 models
└── Compute morphological metrics for all 3

Week 7-8: Analysis & Writing
├── Statistical significance tests
├── Ablation studies (expand_prop, contract_prop)
├── Error analysis
├── Generate all figures and tables
└── Write paper draft

Total: ~2 months
```

---

## Expected Result Structure

### Per-Model Results
```json
{
  "model": "patok",
  "checkpoint": "nemo_experiments/patok/checkpoints/step_10000.ckpt",
  "benchmarks": {
    "pacute": {
      "affixation": {"accuracy": 0.72, "n": 280},
      "composition": {"accuracy": 0.68, "n": 280},
      "manipulation": {"accuracy": 0.65, "n": 320},
      "syllabification": {"accuracy": 0.70, "n": 160},
      "overall": {"accuracy": 0.69, "n": 1040}
    },
    "hierarchical": {
      "level_0": {"accuracy": 0.95, "n": 200},
      "level_1": {"accuracy": 0.85, "n": 200},
      "level_2": {"accuracy": 0.70, "n": 200},
      "level_3": {"accuracy": 0.65, "n": 200},
      "level_4": {"accuracy": 0.60, "n": 200},
      "level_5": {"accuracy": 0.55, "n": 196},
      "overall": {"accuracy": 0.72, "n": 1196}
    },
    "langgame": {"accuracy": 0.78, "n": 1000},
    "math": {"accuracy": 0.42, "n": 1000},
    "morphological": {
      "morph_score": 0.52,
      "boundary_f1": 0.38,
      "fragmentation": 1.35,
      "affix_coverage": 0.446
    }
  }
}
```

### Comparison Table (Expected)
```
| Benchmark        | Baseline | StochasTok | Patok  | Δ (Patok-Base) |
|------------------|----------|------------|--------|----------------|
| PACUTE Overall   | 0.58     | 0.62       | 0.69   | +0.11***       |
| - Affixation     | 0.52     | 0.59       | 0.72   | +0.20***       |
| - Composition    | 0.62     | 0.64       | 0.68   | +0.06**        |
| - Manipulation   | 0.60     | 0.62       | 0.65   | +0.05*         |
| - Syllabification| 0.58     | 0.63       | 0.70   | +0.12***       |
|                  |          |            |        |                |
| Hierarchical     | 0.68     | 0.70       | 0.72   | +0.04**        |
| - Level 0-1      | 0.90     | 0.90       | 0.90   | 0.00           |
| - Level 2-3      | 0.62     | 0.66       | 0.68   | +0.06**        |
| - Level 4-5      | 0.52     | 0.56       | 0.58   | +0.06**        |
|                  |          |            |        |                |
| LangGame         | 0.75     | 0.76       | 0.78   | +0.03          |
| Math             | 0.41     | 0.41       | 0.42   | +0.01          |
|                  |          |            |        |                |
| MorphScore       | 0.235    | 0.38       | 0.52   | +0.285***      |
| Boundary F1      | 0.165    | 0.28       | 0.38   | +0.215***      |
| Fragmentation    | 1.574    | 1.42       | 1.35   | -0.224***      |

*** p < 0.001, ** p < 0.01, * p < 0.05
```

---

## Key Hypotheses Tested

1. **H1**: Patok > Baseline on PACUTE morphological tasks
   - **Mechanism**: Affix-aware tokenization preserves morpheme boundaries
   - **Expected effect**: Large (Cohen's d > 0.8)

2. **H2**: Patok > StochasTok on PACUTE affixation specifically
   - **Mechanism**: Linguistic guidance outperforms generic expansion
   - **Expected effect**: Medium (Cohen's d > 0.5)

3. **H3**: Patok ≈ StochasTok on LangGame
   - **Mechanism**: Character-level tasks don't require morphological awareness
   - **Expected effect**: Small or null (Cohen's d < 0.2)

4. **H4**: All models ≈ on Math
   - **Mechanism**: Numerical reasoning is invariant to tokenization
   - **Expected effect**: Null (Cohen's d ≈ 0)

5. **H5**: MorphScore improvement correlates with PACUTE accuracy
   - **Mechanism**: Boundary alignment → morphological understanding
   - **Expected correlation**: r > 0.7

---

## Related Documentation

- **[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)** - Complete research design
- **[SETUP.md](SETUP.md)** - Environment setup
- **[USAGE.md](USAGE.md)** - Training workflow
- **[AFFIX_PROCESSING.md](AFFIX_PROCESSING.md)** - Patok implementation
- **[HIERARCHICAL_TASKS.md](HIERARCHICAL_TASKS.md)** - Benchmark design
