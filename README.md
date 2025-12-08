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
nano .env  # Add: HF_TOKEN, WANDB_API_KEY, ENROOT_PATH, etc.
source .env

# Install container (~15 minutes)
bash training/nemo/setup/setup_enroot.sh

# Verify
enroot list | grep nemo_framework
```

See **[SETUP.md](SETUP.md)** for detailed setup instructions.

### 2. Preprocess Data

```bash
# Test first
qsub jobs/preprocess_test_chunk1.pbs

# Full preprocessing (parallel, faster)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs
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
python scripts/generate_evaluation_datasets.py

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
| **[docs/TRAINING.md](docs/TRAINING.md)** | Training workflows and configurations |
| **[docs/EVALUATION.md](docs/EVALUATION.md)** | Benchmark generation and model evaluation |
| **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)** | Data preprocessing guide |

---

## Evaluation Benchmarks

### PACUTE (11,225 tasks)
**P**ilipino **A**ffix and **C**haracter-Level **U**nderstanding of **T**okens **E**valuation

- **Affixation** (280 tasks): Filipino affix identification and application
- **Composition** (3,905 tasks): Character counting, diacritics, word formation
- **Manipulation** (5,120 tasks): Character operations (insert, delete, swap)
- **Syllabification** (1,280 tasks): Syllable counting, stress, reduplication

### Hierarchical (1,798 tasks)
Diagnostic tasks across 6 compositional levels to identify where models fail:
- Level 0: Character Recognition
- Level 1: Character Manipulation
- Level 2: Morpheme Decomposition ⚠️ **Critical bottleneck**
- Level 3: Morpheme Manipulation
- Level 4: Morpheme Composition
- Level 5: Complex Reasoning

### Additional Benchmarks
- **LangGame** (3,000 tasks): Subword understanding
- **Math** (3,000 tasks): Multi-digit addition

---

## Repository Structure

```
filipino-morphology-llm/
├── README.md                    # This file
├── SETUP.md                     # Setup guide
├── docs/                        # Documentation
│   ├── RESEARCH.md             # Research overview
│   ├── TRAINING.md             # Training guide
│   └── EVALUATION.md           # Evaluation guide
├── data/                        # Data files
│   ├── benchmarks/             # Evaluation benchmarks
│   ├── chunks/                 # Preprocessed chunks
│   └── processed/              # Binary format data
├── src/                         # Source code
│   ├── evaluation/             # Benchmark generators & evaluators
│   └── tokenization/           # Tokenization processors
│       ├── stochastok_processor.py
│       └── patok_processor.py
├── training/                    # Training code
│   ├── nemo/                   # NeMo CPT (current focus)
│   │   ├── run_cpt.py          # Main training script
│   │   ├── setup/              # Setup scripts
│   │   └── data/               # Data preprocessing
│   │       └── DATA_PREPROCESSING.md
│   └── stochastok/             # Small-scale training (GPT-2)
├── jobs/                        # PBS job scripts
│   ├── run_cpt.pbs             # Training job
│   ├── preprocess_data_parallel.pbs
│   └── preprocess_test_chunk1.pbs
└── scripts/                     # Utility scripts
    ├── generate_evaluation_datasets.py  # Generate benchmarks
    ├── run_evaluation.py           # Evaluate models
    ├── run_evaluation_batch.sh     # Batch evaluation
    └── evaluate_downstream.py      # Downstream tasks
```

---

## Key Features

### Tokenization Processors

**Stochastok** (`src/tokenization/stochastok_processor.py`):
```python
processor = StochastokProcessor(tokenizer, expand_prop=0.1)
expanded_ids = processor.expand(token_ids)
```

**Patok** (`src/tokenization/patok_processor.py`):
```python
processor = PatokProcessor(tokenizer, 
                          expand_prop=0.3, 
                          contract_prop=0.3,
                          affix_preference=0.7)
processed_ids = processor.affix_aware_expand_contract(token_ids)
```

### Evaluation Framework

```python
# Generate all benchmarks
from scripts.generate_evaluation_datasets import main
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
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/filipino-morphology-llm}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaboration:
- Create an issue on GitHub
- Email: your.email@example.com

---

**Ready to get started?** See **[SETUP.md](SETUP.md)** for installation instructions.
