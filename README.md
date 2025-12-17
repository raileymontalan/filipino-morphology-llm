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
├── scripts/                        # 🔧 Workflow Scripts
│   ├── generate_benchmarks.py      # Generate all evaluation benchmarks
│   ├── run_evaluation.py           # Run model evaluation (CLI)
│   ├── run_analysis.py             # Analysis tools (CLI)
│   ├── run_full_evaluation.sh      # Comprehensive evaluation script
│   ├── build_tokenizer_expansions.py # Build tokenizer expansion files
│   ├── download_seapile.py         # Download SEA-PILE corpus
│   └── preprocess_all_*.sh         # Preprocessing workflows
│
├── data/                           # 📊 Data Files
│   ├── benchmarks/                 # Generated benchmarks (JSONL)
│   │   ├── affixation_mcq.jsonl
│   │   ├── affixation_gen.jsonl
│   │   ├── composition_mcq.jsonl
│   │   ├── hierarchical_mcq.jsonl
│   │   └── ...
│   ├── affixes_filipino/           # Filipino affix lists (NEW location)
│   │   ├── prefix.txt              # Prefix affixes
│   │   ├── infix.txt               # Infix affixes
│   │   └── suffix.txt              # Suffix affixes
│   ├── expansions/                 # Tokenizer expansions (gitignored, large)
│   ├── corpora/                    # Training corpora (gitignored)
│   │   └── pacute_data/            # PACUTE source data
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
├── tests/                          # 🧪 Test Files (test_*.py convention)
│   ├── test_affixation.py          # Affixation task tests
│   ├── test_composition.py         # Composition task tests
│   ├── test_manipulation.py        # Manipulation task tests
│   ├── test_syllabification.py     # Syllabification tests
│   ├── test_patok_morphology.py    # Patok processor tests
│   ├── test_stochastok_processor.py# Stochastok processor tests
│   └── test_verify_setup.py        # Environment verification
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

## Baseline Results (Pretrained Models)

Results on 32 pretrained models before Filipino CPT, sorted by PACUTE accuracy:

| Model | Type | PACUTE | Hier. | LangGame | CUTE | Math |
|-------|------|--------|-------|----------|------|------|
| **SEA-LION-Gemma-v3-9B-IT** | IT | 43.7% | 27.9% | 80.4% | 72.1% | 97.6% |
| **Qwen2.5-7B-Instruct** | IT | 42.3% | 27.4% | 68.3% | 43.2% | 83.5% |
| **SEA-LION-v3-8B-IT** | IT | 40.8% | 27.9% | 65.6% | 71.0% | 93.4% |
| **Qwen2.5-7B** | PT | 39.6% | 28.0% | 63.5% | 65.1% | 53.6% |
| **Qwen2.5-14B-Instruct** | IT | 39.2% | 27.4% | 76.7% | 69.2% | 92.0% |
| **Qwen2.5-14B** | PT | 37.9% | 27.0% | 65.0% | 74.1% | 69.7% |
| **Gemma-2-9B** | PT | 37.9% | 27.4% | 57.6% | 68.1% | 98.1% |
| **LLaMA-3.1-8B-Instruct** | IT | 37.2% | 27.5% | 62.2% | 69.4% | 97.5% |
| **SEA-LION-Gemma-v3-9B** | PT | 36.3% | 27.9% | 45.6% | 60.4% | 97.6% |
| **LLaMA-3.1-8B** | PT | 35.5% | 27.4% | 38.9% | 68.2% | 94.9% |
| **LLaMA-3.2-1B-Instruct** | IT | 35.5% | 27.2% | 33.3% | 46.6% | 59.3% |
| **Qwen2.5-3B-Instruct** | IT | 35.4% | 28.4% | 57.0% | 28.8% | 64.3% |
| **LLaMA-3.2-3B-Instruct** | IT | 35.3% | 25.7% | 58.4% | 59.2% | 93.7% |
| **SEA-LION-v3-8B** | PT | 35.3% | 27.9% | 49.3% | 58.0% | 97.2% |
| **Qwen2.5-3B** | PT | 34.5% | 27.0% | 46.7% | 41.8% | 64.4% |
| **Gemma-7B-Instruct** | IT | 34.0% | 26.5% | 54.6% | 49.0% | 37.4% |
| **LLaMA-3.2-1B** | PT | 33.2% | 27.5% | 32.5% | 51.0% | 17.0% |
| **Qwen3-4B-Instruct** | IT | 30.3% | 28.9% | 61.0% | 22.8% | 16.6% |
| **Qwen2.5-1.5B** | PT | 30.3% | 28.0% | 40.7% | 55.3% | 74.3% |
| **Gemma-7B** | PT | 30.2% | 25.7% | 59.4% | 66.5% | 97.1% |
| **LLaMA-3.2-3B** | PT | 30.2% | 26.7% | 37.7% | 58.9% | 80.9% |
| **Gemma-2-2B-Instruct** | IT | 30.1% | 28.5% | 51.5% | 27.5% | 38.1% |
| **Qwen2.5-0.5B-Instruct** | IT | 29.8% | 26.9% | 37.8% | 34.1% | 26.1% |
| **Gemma-2B-Instruct** | IT | 29.2% | 26.7% | 40.9% | 36.2% | 7.3% |
| **Qwen3-4B-Thinking** | IT | 28.7% | 27.9% | 52.8% | 43.3% | 88.4% |
| **Qwen2.5-1.5B-Instruct** | IT | 28.2% | 27.7% | 42.3% | 39.0% | 62.4% |
| **Gemma-2B** | PT | 27.8% | 26.2% | 32.1% | 50.6% | 91.6% |
| **Qwen2.5-0.5B** | PT | 27.6% | 27.9% | 36.2% | 30.2% | 19.0% |
| **GPT-2-xl** | PT | 25.9% | 25.7% | 26.4% | 28.4% | - |
| **GPT-2-large** | PT | 23.5% | 28.0% | 25.0% | 30.3% | - |
| **GPT-2-medium** | PT | 23.1% | 28.0% | 24.9% | 20.4% | - |
| **GPT-2** | PT | 22.6% | 28.9% | 24.5% | 23.2% | - |

**Key Findings:**
- **SEA-LION models excel on Filipino morphology** - trained on Southeast Asian languages including Filipino
- **PACUTE scales with model size** - 22.6% (GPT-2 124M) → 43.7% (SEA-LION 9B)
- **Hierarchical remains flat** - ~26-29% across all models (near 25% random baseline)
- **LangGame shows strong scaling** - 24.5% → 80.4%
- **Math: LLaMA/SEA-LION/Gemma dominate** - 90%+ vs GPT-2's 0%
- **Gemma-2-9B added** - Strong baseline at 37.9% PACUTE, 98.1% Math

*MCQ benchmarks (PACUTE, Hierarchical, LangGame): 25% = random baseline (4 options). CUTE and Math: contains-match accuracy.*

### Performance Visualizations

#### Top Models on Filipino Morphology (PACUTE)
<p align="center">
  <img src="docs/images/top_models_pacute.png" alt="Top 10 Models on PACUTE" width="700">
</p>

Instruct-tuned models generally outperform base models on Filipino morphological understanding, with SEA-LION models showing particularly strong performance due to their Southeast Asian language training data.

#### Model Size vs Performance
<p align="center">
  <img src="docs/images/size_vs_performance.png" alt="Model Size vs Performance" width="700">
</p>

Performance on PACUTE scales with model size, though regional models (SEA-LION) punch above their weight class compared to general-purpose models like LLaMA and GPT-2.

#### Model Family Comparison
<p align="center">
  <img src="docs/images/family_comparison.png" alt="Model Family Comparison" width="700">
</p>

Average performance by model family across all benchmarks. SEA-LION and Qwen2.5 families show the strongest overall performance.

#### Performance Heatmap
<p align="center">
  <img src="docs/images/heatmap.png" alt="Full Results Heatmap" width="600">
</p>

Complete results across all 32 models and 4 benchmarks. Darker colors indicate higher accuracy.

#### Benchmark Comparison (Top 15 Models)
<p align="center">
  <img src="docs/images/benchmark_comparison.png" alt="Benchmark Comparison" width="700">
</p>

Head-to-head comparison across benchmarks for the top 15 performing models.

---

## Continued Pretraining Results

Results from continued pretraining of Gemma-2-2B on SEA-PILE v2 Filipino corpus (~7.4GB) with different tokenization strategies.

### MCQ Benchmark Accuracy

| Model | Training Steps | PACUTE | Hierarchical | LangGame | Math (EM) |
|-------|---------------|--------|--------------|----------|-----------|
| **gemma-2-2b (base)** | 0 | 30.4% | 27.5% | 40.0% | 88.5%* |
| | | | | | |
| **Vanilla (Baseline BPE)** | | | | | |
| vanilla-step999 | 1K | 22.9% | 24.7% | 25.8% | 0.0% |
| vanilla-step1999 | 2K | 27.1% | **26.7%** | 27.5% | 0.0% |
| vanilla-step2999 | 3K | 23.7% | 24.4% | 26.7% | 0.0% |
| vanilla-step3999 | 4K | 23.3% | 25.4% | 26.6% | 0.0% |
| vanilla-step4999 | 5K | 23.6% | 25.5% | 26.2% | 0.0% |
| | | | | | |
| **Stochastok (~10% expansion)** | | | | | |
| stochastok-step1999 | 2K | 26.5% | 23.4% | 27.7% | 0.0% |
| stochastok-step2999 | 3K | 24.9% | 24.0% | 28.6% | 0.0% |
| stochastok-step3999 | 4K | **28.1%** | 25.2% | **28.7%** | 0.0% |
| stochastok-step4999 | 5K | 27.1% | 26.0% | 27.7% | 0.0% |
| | | | | | |
| **Patok (Morphology-aware)** | | | | | |
| patok-step1999 | 2K | 26.6% | 26.2% | 26.5% | 0.0% |
| patok-step2999 | 3K | 26.6% | 25.7% | 26.7% | 0.0% |
| patok-step3999 | 4K | 27.4% | 25.2% | 26.1% | 0.0% |
| patok-step4999 | 5K | 27.0% | 25.5% | 26.2% | 0.0% |

*MCQ benchmarks: 25% = random baseline (4 options). Math: exact match accuracy (base model uses contains-match*). Bold indicates best across all methods. CUTE benchmark skipped (generative-only).*

### Key Findings

1. **Stochastok achieves best PACUTE performance**: 28.1% at step 4K vs vanilla 27.1% at step 2K vs patok 27.4% at step 4K
2. **Stochastok dominates LangGame**: 28.7% (stochastok-step3999) significantly outperforms vanilla (~26-27%) and patok (~26%)
3. **Patok shows consistent Hierarchical scores**: Patok-step1999 achieves 26.2% on hierarchical reasoning, competitive with vanilla-step1999 (26.7%)
4. **Vanilla peaks early, then degrades**: Best performance at step 2K, declining thereafter
5. **Catastrophic forgetting of math**: All trained models lose math capability entirely (88.5% → 0.0%)
6. **All methods below base model**: None of the CPT approaches surpasses the base Gemma-2-2B on morphological benchmarks - suggests either insufficient training or need for different approach

### Trained Model Checkpoints

All checkpoints available on HuggingFace:

| Method | HuggingFace Repository | Available Steps |
|--------|----------------------|-----------------|
| Vanilla | [davidafrica/gemma2-2b-filipino-vanilla](https://huggingface.co/davidafrica/gemma2-2b-filipino-vanilla) | 999, 1999, 2999, 3999, 4999 |
| Stochastok | [davidafrica/gemma2-2b-filipino-stochastok](https://huggingface.co/davidafrica/gemma2-2b-filipino-stochastok) | 1999, 2999, 3999, 4999 |
| Patok | [davidafrica/gemma2-2b-filipino-patok](https://huggingface.co/davidafrica/gemma2-2b-filipino-patok) | 1999, 2999, 3999, 4999 |

*Note: step999 checkpoints for stochastok/patok were incomplete due to distributed checkpoint format issues.*

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

## Development

### Code Organization

**Key File Locations (Updated Dec 2024):**
- **Tokenization**: `src/tokenization/` - Base processor, Patok, Stochastok
- **Evaluation**: `src/evaluation/` - Benchmarks, metrics, evaluators  
- **Analysis**: `src/analysis/` - Inference analysis, visualizations
- **Scripts**: `scripts/` - Workflow orchestration (generate, preprocess, evaluate)
- **Tests**: `tests/test_*.py` - All test files follow `test_` prefix convention
- **Affix Data**: `data/affixes_filipino/` - prefix.txt, infix.txt, suffix.txt
- **Expansions**: Large JSON files in `data/expansions/` (gitignored, regenerate with `scripts/build_tokenizer_expansions.py`)

### Import Conventions

All Python files use consistent path setup:
```python
from pathlib import Path
import sys
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

For scripts that work in containers (`/workspace/`) and locally:
```python
workspace_src = Path("/workspace/src")
local_src = Path(__file__).parent.parent / "src"
src_path = workspace_src if workspace_src.exists() else local_src
sys.path.insert(0, str(src_path))
```

### Pre-commit Hooks

Install development tools:
```bash
pip install pre-commit black isort flake8
pre-commit install
```

Hooks automatically:
- Format code with Black (100 char line length)
- Sort imports with isort
- Check for large files (>5MB)
- Prevent committing large expansion JSONs to src/
- Validate test naming conventions

Run manually: `pre-commit run --all-files`

### Testing

Run all tests:
```bash
python -m pytest tests/
```

Verify setup:
```bash
python tests/test_verify_setup.py
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
