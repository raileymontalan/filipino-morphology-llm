# Filipino Morphology LLM

Research project testing whether **morpheme-aware tokenization** improves language model performance on Filipino morphological tasks.

## Research Question

**Can tokenization that preserves morpheme boundaries improve LLM understanding of agglutinative morphology?**

We compare three tokenization approaches:
1. **Baseline**: Standard BPE (GPT-2, Gemma)
2. **Stochastok**: Stochastic token expansion (~10%)
3. **Patok**: Affix-aware expand-contract (30%+30%)

**Models:** Gemma 3 1B (1B parameters)  
**Data:** SEA-PILE v2 Filipino corpus (7.4GB)  
**Evaluation:** 15,023 morphological tasks

---

## Quick Start

### 1. Setup Environment

```bash
# Configure
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY
source .env

# Option A: Docker (recommended for cloud/local)
docker pull nvcr.io/nvidia/nemo:24.07

# Option B: Enroot (for HPC clusters)
bash training/nemo/setup/setup_enroot.sh
enroot list | grep nemo_framework
```

See **[SETUP.md](SETUP.md)** for detailed setup instructions.

### 2. Preprocess Data

```bash
# Download SEA-PILE v2 Filipino corpus
python scripts/download_seapile.py

# Preprocess for each tokenization mode (parallel, ~1hr each)
bash scripts/preprocess_all_vanilla.sh     # Baseline BPE
bash scripts/preprocess_all_stochastok.sh  # Stochastok expansion
bash scripts/preprocess_all_patok.sh       # Patok morphology-aware
```

See **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)** for complete preprocessing guide.

### 3. Train Model

```bash
# Quick test (10 steps)
qsub jobs/run_cpt_test.pbs

# Full training
qsub jobs/run_cpt.pbs
```

See **[docs/TRAINING.md](docs/TRAINING.md)** for complete training guide.

### 4. Evaluate

```bash
# Generate benchmarks
python scripts/generate_benchmarks.py

# Evaluate model
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute
```

See **[docs/EVALUATION.md](docs/EVALUATION.md)** for complete evaluation guide.

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP.md](SETUP.md)** | Environment setup and installation |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | Research overview, methods, and experimental design |
| **[docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md)** | Complete training plan with timeline and commands |
| **[docs/TRAINING.md](docs/TRAINING.md)** | Training workflows and configurations |
| **[docs/EVALUATION.md](docs/EVALUATION.md)** | Benchmark generation and model evaluation |
| **[docs/BENCHMARK_FORMATS.md](docs/BENCHMARK_FORMATS.md)** | Benchmark format specifications (MCQ vs GEN) |
| **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)** | Data preprocessing guide |
| **[job_templates/README.md](job_templates/README.md)** | PBS job templates and setup wizard |
| **[docs/SECURITY.md](docs/SECURITY.md)** | Security best practices for cluster deployment |
| **[docs/GEMMA3_MONKEY_PATCH.md](docs/GEMMA3_MONKEY_PATCH.md)** | Known Gemma3 bugs and workarounds |

---

## Evaluation Benchmarks

### PACUTE (2,240 MCQ / 1,840 GEN)
**P**ilipino **A**ffix and **C**haracter-Level **U**nderstanding of **T**okens **E**valuation

- **Affixation** (140 MCQ + 140 GEN): Filipino affix identification and application
- **Composition** (900 MCQ + 500 GEN): Character counting, diacritics, word formation
- **Manipulation** (800 MCQ + 800 GEN): Character operations (insert, delete, swap)
- **Syllabification** (400 MCQ + 400 GEN): Syllable counting, stress, reduplication

Both Multiple Choice (log probability) and Generative (text generation) formats available.

### Hierarchical Benchmark (600 MCQ + 598 GEN)
Diagnostic tasks across 6 compositional levels to identify where models fail:

- **Level 0: Character Recognition** (~200 tasks) - Character identification and counting
- **Level 1: Character Manipulation** (~200 tasks) - Insert, delete, swap, substitute operations
- **Level 2: Morpheme Decomposition** (~200 tasks) - ⚠️ **Critical bottleneck** - Affix identification, root extraction
- **Level 3: Morpheme Manipulation** (~200 tasks) - Affix removal and replacement
- **Level 4: Morpheme Composition** (~200 tasks) - Affix application and combination
- **Level 5: Complex Reasoning** (~200 tasks) - Multi-step morphological transformations

**Key Insight:** Level 2 (Morpheme Decomposition) is the critical bottleneck. Failures here cascade through Levels 3-5.

### Additional Benchmarks
- **CUTE** (1,400 GEN): Character Understanding Tasks Evaluation across 14 task types
- **LangGame** (1,000 MCQ + 1,000 GEN): Subword understanding
- **Multi-digit Addition** (1,000 MCQ + 1,000 GEN): Numerical reasoning

**Total: 10,678 evaluation tasks** across all benchmarks. All benchmarks support filtering by evaluation mode (`--eval-mode mcq|gen|both`).

---

## Repository Structure

```
filipino-morphology-llm/
├── README.md                       # This file
├── CLAUDE.md                       # AI assistant guidance
├── SETUP.md                        # Environment setup guide
├── setup.py                        # Package installation
├── requirements.txt                # Python dependencies
│
├── docs/                           # 📚 Documentation
│   ├── RESEARCH.md                 # Research overview & methods
│   ├── TRAINING.md                 # Training workflows
│   ├── EVALUATION.md               # Evaluation guide
│   ├── BENCHMARK_FORMATS.md        # MCQ vs GEN format specs
│   ├── SECURITY.md                 # Security best practices
│   └── GEMMA3_MONKEY_PATCH.md      # Gemma3 bug workarounds
│
├── src/                            # 📦 Source Code (importable package)
│   ├── tokenization/               # Tokenization processors
│   │   ├── base_processor.py       # Base class with common utilities
│   │   ├── stochastok_processor.py # Stochastic token expansion
│   │   ├── patok_morphology.py     # Morphology-aware Patok (RECOMMENDED)
│   │   ├── patok_processor.py      # DEPRECATED - use patok_morphology.py
│   │   └── affix_decomposition.py  # Affix decomposition utilities
│   │
│   ├── evaluation/                 # Evaluation framework
│   │   ├── loaders/                # Benchmark loaders (registry pattern)
│   │   │   ├── registry.py         # @register_loader decorator
│   │   │   ├── pacute.py           # PACUTE benchmark loader
│   │   │   ├── hierarchical.py     # Hierarchical benchmark loader
│   │   │   ├── cute.py             # CUTE benchmark loader
│   │   │   ├── langgame.py         # LangGame benchmark loader
│   │   │   └── multi_digit_addition.py
│   │   ├── datasets/               # Dataset generation
│   │   │   ├── generators/         # Task generators by category
│   │   │   │   ├── affixation.py
│   │   │   │   ├── composition.py
│   │   │   │   ├── manipulation.py
│   │   │   │   ├── syllabification.py
│   │   │   │   ├── stress.py
│   │   │   │   └── hierarchical.py
│   │   │   └── scripts/            # Benchmark generation scripts
│   │   ├── evaluators/             # Evaluation logic
│   │   │   ├── mcq_evaluator.py    # Log-probability MCQ scoring
│   │   │   ├── hierarchical.py     # Hierarchical evaluation
│   │   │   └── math_evaluator.py   # Math task evaluation
│   │   ├── metrics/                # Evaluation metrics
│   │   └── utils/                  # Utilities (constants, strings, syllabification)
│   │
│   └── analysis/                   # Analysis tools
│       ├── tokenization/           # Tokenizer comparison & analysis
│       ├── affixes/                # Affix coverage analysis
│       ├── datasets/               # Dataset comparison
│       ├── morphological_metrics.py
│       └── information_theory.py
│
├── training/                       # 🏋️ Training Pipelines
│   ├── nemo/                       # NeMo Framework CPT (ACTIVE)
│   │   ├── run_cpt.py              # Main training script
│   │   ├── data/                   # Data preprocessing
│   │   │   ├── preprocess_data.py  # JSONL → Megatron binary
│   │   │   ├── split_jsonl.py      # Split corpus into chunks
│   │   │   └── DATA_PREPROCESSING.md
│   │   ├── setup/                  # Container setup scripts
│   │   │   ├── setup_enroot.sh     # Enroot container setup
│   │   │   └── setup_singularity.sh
│   │   └── examples/               # Example configurations
│   │
│   └── stochastok/                 # ⚠️ DEPRECATED - Legacy GPT-2 training
│       └── DEPRECATED.md           # See this file for details
│
├── scripts/                        # 🔧 Utility Scripts
│   ├── generate_benchmarks.py      # Generate all evaluation benchmarks
│   ├── run_evaluation.py           # Run model evaluation
│   ├── run_full_evaluation.sh      # Comprehensive evaluation script
│   ├── analyze_inference_results.py# Analyze evaluation outputs
│   ├── download_seapile.py         # Download SEA-PILE corpus
│   └── verify_setup.py             # Verify environment setup
│
├── data/                           # 📊 Data Files
│   ├── benchmarks/                 # Generated benchmarks (JSONL)
│   │   ├── affixation_mcq.jsonl
│   │   ├── affixation_gen.jsonl
│   │   ├── composition_mcq.jsonl
│   │   ├── hierarchical_mcq.jsonl
│   │   └── ...
│   ├── affixes/                    # Filipino affix lists
│   │   └── filipino_affixes.txt
│   ├── corpora/                    # Training corpora (gitignored)
│   │   └── pacute_data/            # PACUTE source data
│   ├── tokenizer_expansions/       # Cached tokenizer expansions
│   ├── vocabularies/               # Tokenizer vocabularies
│   └── word_frequencies.csv        # Filipino word frequencies
│
├── configs/                        # ⚙️ Training Configurations
│   ├── pretraining.yaml            # Pretraining config
│   └── instruction_tuning.yaml     # Instruction tuning config
│
├── job_templates/                  # 📝 PBS Job Templates
│   ├── README.md                   # Template usage guide
│   ├── setup_jobs.sh               # Interactive setup wizard
│   ├── run_cpt.template.pbs
│   ├── preprocess_data.template.pbs
│   └── ...
│
├── jobs/                           # Generated PBS jobs (gitignored)
│
├── notebooks/                      # 📓 Jupyter Notebooks
│   ├── create_affixation.ipynb     # Affixation task development
│   ├── create_composition_*.ipynb  # Composition/manipulation tasks
│   └── diksiyonaryo.ipynb          # Dictionary exploration
│
├── tests/                          # 🧪 Test Files
│   ├── test_affixation.py
│   ├── test_composition.py
│   ├── test_manipulation.py
│   ├── test_syllabification.py
│   └── test_patok_morphology.py
│
└── results/                        # 📈 Evaluation Results (gitignored)
    └── <model_name>/
        ├── evaluation_results_*.json
        └── inference/*.jsonl
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `src/tokenization/` | Core tokenization processors (Stochastok, Patok) |
| `src/evaluation/` | Benchmark loading, generation, and evaluation |
| `src/analysis/` | Analysis tools for tokenization and morphology |
| `training/nemo/` | NeMo Framework training (Gemma 3 1B CPT) |
| `scripts/` | Command-line utilities for benchmarks and evaluation |
| `data/benchmarks/` | Generated evaluation tasks in JSONL format |
| `data/affixes/` | Filipino affix lists for Patok processor |
| `job_templates/` | PBS job templates (version-controlled) |
| `jobs/` | Generated PBS jobs with real paths (gitignored) |
| `results/` | Model evaluation outputs (gitignored) |

---

## Key Features

### Tokenization Processors

**Stochastok** (`src/tokenization/stochastok_processor.py`):
```python
processor = StochastokProcessor(tokenizer, expand_prop=0.1)
expanded_ids = processor.expand(token_ids)
```

**Patok** (`src/tokenization/patok_morphology.py`):
```python
processor = MorphologyAwarePatokProcessor(
    tokenizer,
    prefix_file='src/tokenization/affixes/prefix.txt',
    infix_file='src/tokenization/affixes/infix.txt',
    suffix_file='src/tokenization/affixes/suffix.txt',
)
processed_ids = processor.contract_expand(token_ids)
```

### Evaluation Framework

```python
# Generate all benchmarks
from scripts.generate_benchmarks import main
main()

# Evaluate model
from scripts.run_evaluation import evaluate_model
results = evaluate_model(model_name="gpt2", benchmarks=["pacute"])
```

---

## Expected Results

| Method | PACUTE Affixation | Hierarchical Level 2 | Improvement |
|--------|-------------------|---------------------|-------------|
| **Baseline** | 40-50% | 30-40% | - |
| **Stochastok** | 50-65% | 45-55% | +10-15% |
| **Patok** | 60-70% | 55-70% | +20-30% |

Affix-aware tokenization (Patok) is expected to show the largest improvements on morphological understanding tasks.

---

## Citation

If you use this code or benchmarks, please cite:

```bibtex
@misc{filipino-morphology-llm,
  title={Affix-Aware Tokenization for Filipino Morphological Understanding},
  author={Africa, David Demitri and Montalan, Railey and Gamboa, Lance Calvin},
  year={2025},
  url={https://github.com/DavidDemitriAfrica/filipino-morphology-llm}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaboration:
- Create an issue on [GitHub](https://github.com/DavidDemitriAfrica/filipino-morphology-llm/issues)
- Email: raileymontalan@outlook.com

---

**Ready to get started?** See **[SETUP.md](SETUP.md)** for installation instructions.
