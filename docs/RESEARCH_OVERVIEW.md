# Research Overview: Affix-Aware Continued Pretraining for Filipino

## Research Question

**Can continued pretraining (CPT) with affix-aware tokenization improve language model performance on morphological understanding tasks in Filipino?**

---

## Hypothesis

Standard BPE tokenization (e.g., GPT-2, LLaMA) systematically destroys morpheme boundaries in agglutinative languages like Filipino. By applying **affix-aware tokenization** during continued pretraining, we hypothesize that models will:

1. Better preserve morpheme boundaries
2. Show improved performance on morphological understanding tasks
3. Develop stronger representation of Filipino linguistic structure

---

## Experimental Design

### Training Systems

We compare **three tokenization approaches** during continued pretraining:

| Approach | Method | Implementation |
|----------|--------|----------------|
| **Baseline** | Standard BPE (GPT-2 tokenizer) | Current NeMo implementation |
| **StochasTok** | Stochastic token expansion | `src/tokenization/stochastok_processor.py` |
| **Patok** (New) | Affix-aware expand-contract | `src/tokenization/patok_processor.py` |

#### 1. Baseline (Vanilla BPE)
- Uses pretrained tokenizer as-is (e.g., GPT-2, Gemma)
- No morphological awareness
- **Current status**: Implemented in `training/nemo/run_cpt.py`

#### 2. StochasTok (Original Method)
- **What it does**: Stochastically expands tokens by splitting them into smaller subtokens
- **How**: During training, randomly splits ~10% of tokens using tokenizer's merge operations
- **Goal**: Forces model to learn fine-grained subword structure
- **Implementation**: `StochastokProcessor.expand()`
- **Preprocessing**: `training/stochastok/data_processing/stochastok_expand_dataset.py`

**Example**:
```
Token:  "matulog" → [mat][ul][og]
Expand: "matulog" → [ma][t][ul][og]  (randomly split "mat" → "ma" + "t")
```

#### 3. Patok (Our Innovation)
- **What it does**: Affix-aware expand-contract with preference for Filipino morpheme boundaries
- **How**: 
  - Expand: Split tokens, preferring splits that form valid Filipino affixes
  - Contract: Merge tokens, preferring merges that preserve affix boundaries
- **Goal**: Guide model toward morphologically-meaningful tokenization
- **Implementation**: `PatokProcessor.expand_contract()`
- **Key innovation**: Uses affix list (`data/affixes/filipino_affixes.txt`) to guide split/merge decisions

**Example**:
```
Token:    "matulog" → [mat][ul][og]
Expand:   "matulog" → [ma][tulog]      (prefer "ma-" prefix split)
Contract: [ma][t][ulog] → [ma][tulog]  (prefer keeping "ma-" together)
```

### Model Scale Comparison

We test on two scales to understand how model capacity affects morphological learning:

| Scale | Original StochasTok | Our Experiments |
|-------|---------------------|-----------------|
| **Small** | GPT-2 (117M params) | GPT-2 (117M) |
| **Large** | Not tested | Gemma 3 1B (1B params) |

**Why two scales?**
- Small models may benefit more from explicit morphological guidance (limited capacity)
- Large models may learn morphology implicitly (more capacity)
- Testing both reveals whether Patok is scale-dependent

### Training Infrastructure

| Component | Small-Scale (StochasTok) | Large-Scale (Our Work) |
|-----------|--------------------------|------------------------|
| **Code** | `training/stochastok/` | `training/nemo/` |
| **Framework** | Pure PyTorch | NeMo Framework + Megatron |
| **Hardware** | Single GPU | Multi-GPU (distributed) |
| **Training** | Custom loop | NeMo autoregressive language modeling |
| **Data format** | Memmap arrays | Megatron binary (.bin + .idx) |

**Why NeMo for large models?**
- Handles distributed training automatically
- Memory-efficient tensor parallelism
- Production-ready for billion-parameter models
- Used by NVIDIA for training large LLMs

---

## Evaluation Framework

### 1. LangGame (String Operations)
- **Source**: `training/stochastok/data_processing/make_langgame_dataset.py`
- **Tasks**: 6 types of character-level operations
  - Most of letter: "Which word has the most 'a's?"
  - Contains: "Which word contains 'ng'?"
  - Starts with / Ends with
  - Longest / Shortest
- **Dataset**: 1M train + 1K validation
- **Purpose**: Tests fine-grained character understanding

### 2. Multi-Digit Addition
- **Source**: `training/stochastok/data_processing/make_multi_digit_addition_dataset.py`
- **Tasks**: 3-digit addition problems (123 + 456 = ?)
- **Format**: Three tokenization variants (base, character, stochastok)
- **Purpose**: Tests tokenization's effect on numerical reasoning

### 3. PACUTE (Filipino Morphology)
- **Source**: `src/evaluation/` (affixation.py, composition.py, manipulation.py, syllabification.py)
- **Tasks**: 1,040 Filipino-specific morphological tasks
  - **Affixation** (280 items): Identify prefixes, infixes, suffixes
  - **Composition** (280 items): Character counting, word formation
  - **Manipulation** (320 items): Insert/delete/swap characters
  - **Syllabification** (160 items): Count/extract syllables
- **Data**: `data/benchmarks/mcq_*.jsonl` and `gen_*.jsonl`
- **Purpose**: Core evaluation of Filipino morphological understanding

### 4. Hierarchical Benchmark (Diagnostic Levels)
- **Source**: `src/evaluation/hierarchical_tasks.py`, `scripts/generate_hierarchical_benchmark.py`
- **Tasks**: 1,196 tasks across 6 diagnostic levels
  - **Level 0**: Character recognition (basic perception)
  - **Level 1**: Character manipulation (edit operations)
  - **Level 2**: Morpheme decomposition (identify boundaries)
  - **Level 3**: Morpheme manipulation (edit morphemes)
  - **Level 4**: Morpheme composition (combine morphemes)
  - **Level 5**: Complex reasoning (multi-step morphological tasks)
- **Data**: `data/benchmarks/hierarchical_mcq.jsonl` and `hierarchical_gen.jsonl`
- **Purpose**: Diagnose at which complexity level models fail

### 5. Information-Theoretic Analysis
- **Source**: `src/analysis/information_theory.py`, `morphological_metrics.py`
- **Metrics**:
  - **MorphScore**: Alignment between token and morpheme boundaries (0-1)
  - **Boundary F1**: Precision/recall of morpheme boundary detection
  - **Fragmentation**: Average tokens per morpheme (lower = better)
  - **Affix coverage**: Percentage of affixes in tokenizer vocabulary
- **Purpose**: Quantify morphological alignment of tokenization

---

## Current Implementation Status

### ✅ Completed

1. **Baseline training infrastructure**
   - NeMo CPT pipeline working (`training/nemo/`)
   - Data preprocessing for Megatron format
   - PBS job scripts for cluster deployment
   - Gemma 3 1B model integration

2. **Tokenization implementations**
   - `StochastokProcessor` (original method)
   - `PatokProcessor` (our affix-aware method)
   - Both processors generate valid expansions from GPT-2 tokenizer

3. **Evaluation framework**
   - All benchmark datasets generated
   - PACUTE: 1,040 tasks ready
   - Hierarchical: 1,196 tasks ready
   - LangGame: 1M examples ready
   - Multi-digit addition: Ready with 3 tokenization variants

4. **Analysis tools**
   - Morphological metrics implementation
   - Baseline analysis (GPT-2 BPE: MorphScore = 0.235)
   - Oracle analysis (Perfect boundaries: MorphScore = 0.990)
   - Affix coverage analysis (GPT-2: 44.6% coverage)

### ⏳ In Progress

1. **Data preprocessing**
   - Converting seapile-v2.jsonl to Megatron binary format
   - Test preprocessing completed (chunk 1)
   - Full preprocessing: 20 chunks in parallel

2. **Baseline training**
   - Ready to train Gemma 3 1B with vanilla tokenization
   - Awaiting preprocessing completion

### ❌ Not Yet Implemented

1. **StochasTok integration with NeMo**
   - Need to integrate `StochastokProcessor` with NeMo data pipeline
   - Requires preprocessing seapile with stochastok expansion
   - Training loop modification to apply expansion during training

2. **Patok integration with NeMo**
   - Need to integrate `PatokProcessor` with NeMo data pipeline
   - Requires preprocessing seapile with Patok expansion
   - Training loop modification with expand-contract operations

3. **Evaluation pipeline**
   - Need scripts to evaluate trained models on all benchmarks
   - Integration with NeMo checkpoint loading
   - Batch evaluation across all 2,236 tasks

4. **Comparison experiments**
   - Train 3 models: Baseline, StochasTok, Patok
   - Run all evaluations on all 3 models
   - Statistical significance testing
   - Ablation studies (expand_prop, contract_prop, affix_preference)

---

## Expected Results

### Research Outcomes

If our hypothesis is correct, we expect:

1. **Patok > StochasTok > Baseline** on PACUTE morphological tasks
2. **Patok ≈ StochasTok** on LangGame (general string operations)
3. **All ≈ Baseline** on multi-digit addition (numbers are invariant to tokenization)
4. **MorphScore improvement**: 0.235 (Baseline) → 0.4-0.6 (Patok)
5. **Affix coverage**: Model learns to respect affix boundaries even with 44.6% vocab coverage

### Diagnostic Hypotheses (Hierarchical Benchmark)

- **Levels 0-1** (Character-level): All models perform similarly
- **Levels 2-3** (Morpheme-level): Patok shows advantage
- **Levels 4-5** (Complex reasoning): Largest gap between Patok and Baseline

### Scale Effects

- **Small models (GPT-2 117M)**: Large benefit from explicit morphological guidance
- **Large models (Gemma 3 1B)**: Smaller but still significant benefit
- **Finding**: Morphological awareness helps at all scales, but effect may diminish with capacity

---

## Repository Structure Mapping

### Training Systems

```
training/
├── stochastok/              # Original StochasTok implementation
│   ├── models/              # GPT-2-style Transformer (117M)
│   ├── training/            # Custom training loop
│   ├── data_processing/     # Memmap preprocessing + LangGame/Math datasets
│   └── experiments/         # train.py, eval.py for small-scale
│
└── nemo/                    # Our large-scale implementation
    ├── setup/               # Container setup (Enroot)
    ├── data/                # Megatron preprocessing
    └── run_cpt.py           # NeMo training entrypoint
```

### Shared Components

```
src/
├── tokenization/            # SHARED by both training systems
│   ├── patok_processor.py         # Patok (affix-aware)
│   └── stochastok_processor.py    # StochasTok (expansion-only)
│
├── evaluation/              # SHARED evaluation framework
│   ├── affixation.py              # PACUTE affixation tasks
│   ├── composition.py             # PACUTE composition tasks
│   ├── manipulation.py            # PACUTE manipulation tasks
│   ├── syllabification.py         # PACUTE syllabification tasks
│   ├── hierarchical_tasks.py      # Hierarchical benchmark generator
│   └── benchmarks/                # MCQ evaluation infrastructure
│
└── analysis/                # SHARED analysis tools
    ├── morphological_metrics.py   # MorphScore, Boundary F1, etc.
    └── information_theory.py      # Entropy, compression analysis
```

### Utilities

```
scripts/
├── download_seapile.py                    # Dataset download (SHARED)
├── generate_hierarchical_benchmark.py     # Generate diagnostic tasks
├── analyze_*.py                           # Various analysis scripts
└── compare_tokenizers.py                  # Baseline comparisons
```

### Evaluation Data

```
data/
├── benchmarks/                  # All evaluation tasks (2,236 items)
│   ├── mcq_affixation.jsonl           # PACUTE MCQ format
│   ├── gen_affixation.jsonl           # PACUTE generative format
│   ├── hierarchical_mcq.jsonl         # Diagnostic MCQ
│   └── hierarchical_gen.jsonl         # Diagnostic generative
│
├── corpora/                     # Training and linguistic data
│   ├── seapile-v2.jsonl               # CPT corpus (7.4GB)
│   ├── affix_annotations.jsonl        # 472 morpheme-annotated words
│   ├── top_1k_words                   # For LangGame generation
│   └── pacute_data/                   # PACUTE source data
│
└── affixes/
    └── filipino_affixes.txt         # 93 Filipino affixes (Patok guidance)
```

---

## Workflow: From Data to Results

### Phase 1: Baseline (Current)
```bash
# 1. Preprocess data (vanilla tokenization)
python training/nemo/data/preprocess_data.py

# 2. Train baseline model
qsub jobs/run_cpt.pbs

# 3. Evaluate on all benchmarks
python scripts/evaluate_model.py --checkpoint baseline_model --benchmarks all

# 4. Analyze results
python scripts/analyze_results.py --model baseline
```

### Phase 2: StochasTok (Next)
```bash
# 1. Preprocess with StochasTok expansion
python training/nemo/data/preprocess_data_stochastok.py --expand_prop 0.1

# 2. Train StochasTok model
qsub jobs/run_cpt_stochastok.pbs

# 3. Evaluate
python scripts/evaluate_model.py --checkpoint stochastok_model --benchmarks all

# 4. Compare to baseline
python scripts/compare_results.py --models baseline stochastok
```

### Phase 3: Patok (Final)
```bash
# 1. Preprocess with Patok
python training/nemo/data/preprocess_data_patok.py \
    --expand_prop 0.3 --contract_prop 0.3 --affix_preference 0.7

# 2. Train Patok model
qsub jobs/run_cpt_patok.pbs

# 3. Evaluate
python scripts/evaluate_model.py --checkpoint patok_model --benchmarks all

# 4. Final comparison
python scripts/compare_results.py --models baseline stochastok patok
python scripts/generate_paper_tables.py  # LaTeX tables for paper
```

---

## Key Insights

### Why This Matters

1. **Linguistic**: Filipino is agglutinative (builds words by stacking affixes)
   - Example: `pinagsamantalahan` = pinag- + sama + -han + -an (4 morphemes)
   - Standard BPE destroys these boundaries
   
2. **Practical**: Better morphological understanding → better downstream performance
   - Machine translation
   - Information extraction
   - Question answering
   - Text generation

3. **Scientific**: Tests whether explicit linguistic bias helps LLMs
   - Inductive bias vs. learned bias
   - Small data regime (CPT on 7GB, not 100TB)
   - Agglutinative languages understudied in LLM research

### What We'll Learn

1. **Does affix-aware tokenization help?** (Patok vs. Baseline)
2. **Is linguistic knowledge better than generic subword awareness?** (Patok vs. StochasTok)
3. **Does model scale change the answer?** (117M vs. 1B parameters)
4. **At what linguistic complexity does the benefit appear?** (Hierarchical levels 0-5)
5. **Can we quantify morphological alignment improvements?** (MorphScore, Boundary F1)

---

## Next Steps

1. **Complete preprocessing**: Finish converting seapile to Megatron format
2. **Train baseline**: Establish Gemma 3 1B performance with vanilla tokenization
3. **Integrate StochasTok**: Modify NeMo pipeline for StochasTok preprocessing
4. **Integrate Patok**: Modify NeMo pipeline for Patok preprocessing
5. **Build evaluation pipeline**: Automate benchmark evaluation
6. **Run experiments**: Train all 3 variants
7. **Analyze results**: Statistical testing and ablations
8. **Write paper**: Report findings with LaTeX tables and plots

---

## Questions This Document Answers

- **What are we testing?** Affix-aware CPT vs. baseline vs. stochastic expansion
- **Why?** Hypothesis that morphological awareness improves understanding
- **How?** Three tokenization approaches on two model scales
- **Where?** Training in `training/nemo/`, evaluation in `src/evaluation/`
- **What benchmarks?** LangGame, Math, PACUTE, Hierarchical (2,236 tasks total)
- **What's the innovation?** Patok = StochasTok + Filipino affix awareness
- **What's the status?** Baseline ready, StochasTok/Patok integration pending
- **What's the expected outcome?** Patok > StochasTok > Baseline on morphology

---

## Related Documentation

- **[README.md](../README.md)** - Project overview and quick start
- **[docs/SETUP.md](SETUP.md)** - Environment and preprocessing setup
- **[docs/USAGE.md](USAGE.md)** - Training workflow and PBS jobs
- **[docs/AFFIX_PROCESSING.md](AFFIX_PROCESSING.md)** - Patok implementation details
- **[docs/HIERARCHICAL_TASKS.md](HIERARCHICAL_TASKS.md)** - Hierarchical benchmark design
