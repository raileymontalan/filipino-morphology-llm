# Documentation Index

Quick guide to all documentation in this repository.

## Start Here

**New to this project?** Follow this order:

1. **[README.md](README.md)** - Project overview and quick start
2. **[SETUP.md](SETUP.md)** - Environment setup (required first step)
3. **[docs/RESEARCH.md](docs/RESEARCH.md)** - Understand the research
4. **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)** - Preprocess data
5. **[docs/TRAINING.md](docs/TRAINING.md)** - Train models
6. **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluate and analyze

---

## All Documentation

### Root Level
- **[README.md](README.md)** - Project overview, quick start, repository structure
- **[SETUP.md](SETUP.md)** - Complete environment setup guide
- **[INDEX.md](INDEX.md)** - This file

### Research & Methodology
- **[docs/RESEARCH.md](docs/RESEARCH.md)** - Research question, experimental design, tokenization methods, expected results

### Workflows
- **[docs/TRAINING.md](docs/TRAINING.md)** - Training workflows, PBS jobs, monitoring, troubleshooting
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Benchmark generation, model evaluation, analysis
- **[docs/BENCHMARK_FORMATS.md](docs/BENCHMARK_FORMATS.md)** - Benchmark format specifications (MCQ vs GEN)
- **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)** - Data preprocessing for both vanilla and stochastok modes

---

## Quick Reference by Task

### "I want to set up the environment"
→ **[SETUP.md](SETUP.md)**

### "I want to understand the research"
→ **[docs/RESEARCH.md](docs/RESEARCH.md)**

### "I want to preprocess data"
→ **[training/nemo/data/DATA_PREPROCESSING.md](training/nemo/data/DATA_PREPROCESSING.md)**

### "I want to train a model"
→ **[docs/TRAINING.md](docs/TRAINING.md)**

### "I want to evaluate a model"
→ **[docs/EVALUATION.md](docs/EVALUATION.md)**

### "I want to understand tokenization modes"
→ **[docs/RESEARCH.md](docs/RESEARCH.md)** (sections 1-3: Baseline, Stochastok, Patok)

### "I want to run experiments comparing tokenization methods"
→ **[docs/TRAINING.md](docs/TRAINING.md)** (section: Experimental Pipeline)

### "I want to understand the benchmarks"
→ **[docs/EVALUATION.md](docs/EVALUATION.md)** (sections 1-4: PACUTE, Hierarchical, LangGame, Math)

---

## Document Summary

| File | Lines | Purpose |
|------|-------|---------|
| README.md | ~200 | Project overview, quick start |
| SETUP.md | ~770 | Environment setup, installation |
| docs/RESEARCH.md | ~400 | Research design, methods, implementation |
| docs/TRAINING.md | ~350 | Training workflows, configurations |
| docs/EVALUATION.md | ~450 | Benchmarks, evaluation, analysis |
| training/nemo/data/DATA_PREPROCESSING.md | ~300 | Data preprocessing guide |

**Total: ~2,470 lines** of focused documentation (down from 40+ scattered files)

---

## Helper Scripts

In addition to markdown docs, these executable scripts provide quick reference:

- `training/nemo/data/preprocessing_reference.sh` - Preprocessing command examples
- `jobs/QUICK_REFERENCE_PBS.sh` - PBS job command examples

---

## Documentation Philosophy

Each document has a single, clear purpose:
- **No duplication** - Each topic covered once
- **Clear hierarchy** - Know where to start
- **Cross-references** - Easy navigation between related topics
- **Action-oriented** - Step-by-step workflows
- **Complete** - All information in one place per topic

---

**Questions?** All documentation follows the same structure:
1. Quick Start (TL;DR)
2. Overview
3. Detailed Steps
4. Configuration Options
5. Troubleshooting
6. Tips & Best Practices
