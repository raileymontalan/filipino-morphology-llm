# Filipino Morphology LLM

Research project testing whether **affix-aware continued pretraining** improves language model performance on Filipino morphological tasks.

> **📖 New to this project?** Read **[docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)** for the complete research design, experimental setup, and expected outcomes.

## Research Question

**Can tokenization that preserves morpheme boundaries improve LLM understanding of agglutinative morphology?**

We compare three approaches:
1. **Baseline**: Standard BPE (GPT-2, Gemma)
2. **StochasTok**: Stochastic token expansion [(Sims et al. 2025)](https://github.com/anyasims/stochastok)
3. **Patok** (new): Affix-aware expansion with Filipino linguistic guidance

**Models**: GPT-2 (117M) and Gemma 3 1B (1B parameters)  
**Evaluation**: 2,236 morphological tasks across 5 benchmarks

---

## What This Repository Contains

### Evaluation Framework
- **PACUTE benchmark**: 1,040 tasks testing morphological understanding
  - Affixation (280 items): Identify and apply Filipino affixes
  - Composition (280 items): Character counting and word formation
  - Manipulation (320 items): Character operations (insert, delete, swap)
  - Syllabification (160 items): Syllable counting and extraction

- **Hierarchical tasks**: 1,196 tasks across 6 diagnostic levels
  - Level 0: Character recognition
  - Level 1: Character manipulation
  - Level 2: Morpheme decomposition
  - Level 3: Morpheme manipulation
  - Level 4: Morpheme composition
  - Level 5: Complex morphological reasoning

### Data
- **Morpheme annotations**: 472 Filipino words with boundary annotations
- **Affix inventory**: 92 Filipino affixes (prefixes, infixes, suffixes, circumfixes)
- **Syllabified words**: 16,828 words with syllable boundaries
- **Word frequencies**: 118,801 entries for frequency-aware sampling

### Tokenization
- **Patok**: Morphology-aware tokenization with Aho-Corasick affix detection
  - Contract-expand with 95% affix preservation
  - Re-expands by splitting off known affixes ("maganda" → "ma" + "ganda")
  - Handles reduplication ("gaganda" → "ga" + "ganda")
  - Uses 92 Filipino affixes (data/affixes/filipino_affixes.txt)
  - `src/tokenization/patok_morphology.py`
- **StochasTok**: Stochastic token expansion baseline
- **Affix decomposition**: Handles out-of-vocabulary affixes

### Analysis Tools
- **MorphScore**: Measures alignment between token and morpheme boundaries
- **Boundary F1**: Precision/recall of morpheme boundary detection
- **Fragmentation**: Tokens per morpheme
- **Affix coverage**: Vocabulary analysis across tokenizers

## Baseline Results

We analyzed standard BPE tokenization (GPT-2) on 100 morpheme-annotated Filipino words:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MorphScore | 0.235 | Only 23.5% of morpheme boundaries preserved |
| Boundary F1 | 0.165 | Poor precision and recall |
| Fragmentation | 1.574 | ~1.6 tokens per morpheme |

**Finding**: Standard BPE systematically destroys morpheme boundaries.

Example:
```
Word: matulog (ma- + tulog = "will sleep")
Morphemes: ma | tulog
GPT-2:     mat | ul | og
→ Prefix boundary destroyed
```

### Affix Vocabulary Coverage

| Tokenizer | Affixes in Vocab | Coverage |
|-----------|------------------|----------|
| GPT-2 | 41/92 | 44.6% |
| cl100k_base | 42/92 | 45.7% |

**Finding**: ~55% of Filipino affixes require decomposition into sub-tokens.

### Tokenization Comparison

We compared three approaches on 100 morpheme-annotated Filipino words:

| Metric | GPT-2 Baseline | Real Patok | Oracle | Patok Improvement |
|--------|----------------|------------|--------|-------------------|
| MorphScore | 0.235 | 0.657 | 0.990 | +0.422 (+179%) |
| Boundary F1 | 0.165 | 0.365 | 0.643 | +0.199 (+121%) |
| Fragmentation | 1.574 | 1.906 | 1.658 | +0.332 |

**Findings**:
- Real Patok substantially improves morpheme boundary alignment (+179% MorphScore)
- Captures 56% of oracle's theoretical maximum improvement
- Higher fragmentation expected (splitting at morpheme boundaries)
- Oracle represents upper bound with known boundaries

**Note**: Oracle uses ground-truth morpheme boundaries. Real Patok uses 92 Filipino affixes + reduplication detection.

## Getting Started

### Prerequisites
- NVIDIA GPU cluster with **PBS job scheduler** and **Enroot** container runtime
- 20GB free disk space for container and caches
- Weights & Biases account ([free signup](https://wandb.ai))
- HuggingFace account for model downloads

### Quick Setup (PBS + Enroot)

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add your WANDB_API_KEY, HF_TOKEN, ENROOT paths, BIND_MOUNTS
source .env

# 2. Setup NeMo Framework container (~15 minutes, one-time setup)
bash training/nemo/setup/setup_enroot.sh

# 3. Verify installation
enroot list  # Should show: nemo_framework
```

### Data Preprocessing (Required Before Training!)

NeMo requires data in **Megatron binary format** (`.bin` + `.idx` files), not raw JSONL.

```bash
# Option 1: Test with one chunk first (recommended)
qsub jobs/preprocess_test_chunk1.pbs

# Option 2: Preprocess full dataset
qsub jobs/preprocess_data.pbs

# Option 3: Parallel processing (20x faster!)
# Edit jobs/preprocess_data_parallel.pbs: change #PBS -J 1-1 to #PBS -J 1-20
qsub jobs/preprocess_data_parallel.pbs

# Verify outputs
ls -lh data/processed/*.bin
```

See **[docs/SETUP.md](docs/SETUP.md)** for detailed preprocessing guide.

### Training

```bash
# 1. Test training (1 GPU, 10 steps, ~5 minutes)
qsub jobs/run_cpt_test.pbs

# 2. Check test results
qstat
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>.OU

# 3. If test succeeds, run full training (multi-GPU)
qsub jobs/run_cpt.pbs
```

### Documentation
- **[docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)** - 📖 **START HERE**: Research design & experimental setup
- **[docs/SETUP.md](docs/SETUP.md)** - Environment setup & data preprocessing
- **[docs/USAGE.md](docs/USAGE.md)** - Training workflow & PBS job reference
- **[docs/AFFIX_PROCESSING.md](docs/AFFIX_PROCESSING.md)** - Patok implementation details
- **[docs/HIERARCHICAL_TASKS.md](docs/HIERARCHICAL_TASKS.md)** - Hierarchical benchmark design
- **[jobs/QUICK_REFERENCE_PBS.sh](jobs/QUICK_REFERENCE_PBS.sh)** - PBS commands cheatsheet
- **`.env.example`** - Environment variable template

### Need Help?
- **Research context**: [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)
- **Setup issues**: [docs/SETUP.md](docs/SETUP.md)
- **Training issues**: [docs/USAGE.md](docs/USAGE.md)

## Repository Structure

```
filipino-morphology-llm/
├── src/                        # Core library code
│   ├── tokenization/           # Patok, StochasTok, affix decomposition
│   ├── evaluation/             # PACUTE + hierarchical tasks
│   ├── analysis/               # Morphological metrics
│   ├── models/                 # Transformer architecture
│   ├── training/               # Training infrastructure
│   └── data_processing/        # Dataset preprocessing
├── scripts/                    # Utilities & tools
│   ├── setup/                  # Environment setup scripts
│   ├── data/                   # Data preprocessing
│   ├── training/               # Training entrypoints
│   └── analysis/               # Analysis & evaluation
├── data/
│   ├── affixes/                # Filipino affix list
│   ├── benchmarks/             # 2,236 evaluation items
│   ├── corpora/                # Annotations, syllables, frequencies
│   ├── chunks/                 # Split JSONL files for preprocessing
│   └── processed/              # Megatron binary format (.bin + .idx)
├── docs/                       # Documentation
│   ├── SETUP.md                # Environment & preprocessing
│   └── USAGE.md                # Training & job submission
├── jobs/                       # PBS job scripts
├── experiments/                # Training and evaluation scripts
└── configs/                    # Training configurations
```

## Installation

```bash
git clone https://github.com/DavidDemitriAfrica/filipino-morphology-llm.git
cd filipino-morphology-llm
pip install -r requirements.txt
```

## Usage

### Morphology-Aware Tokenization

```python
import tiktoken
from src.tokenization.patok_morphology import MorphologyAwarePatokProcessor

tokenizer = tiktoken.get_encoding("gpt2")
patok = MorphologyAwarePatokProcessor(tokenizer)

text = "Nagkukumahog na pinadalhan ng magagandang parlorista"
token_ids = patok.process(text)
```

### Analyze Tokenizer

Check affix coverage:
```bash
python scripts/analyze_affix_coverage.py \
    --compare gpt2 cl100k_base \
    --affixes-file data/affixes/filipino_affixes.txt
```

### Generate Hierarchical Tasks

```bash
python scripts/generate_hierarchical_benchmark.py
```

### Run Baseline Analysis

```bash
python scripts/analyze_tokenization_simple.py
```

### Compare Tokenizers

```bash
python scripts/compare_tokenizers.py
```

## What Still Needs Doing

1. **Train with Patok**: Apply morphology-aware tokenization during pre-training
2. **Evaluate on tasks**: Test on 2,236 evaluation items
3. **Measure downstream**: Compare Patok vs baseline on hierarchical tasks
4. **Get Filipino corpus**: Need training data for pre-training

## Key Files

### Data
- `data/benchmarks/hierarchical_mcq.jsonl` - 598 hierarchical MCQ tasks
- `data/benchmarks/hierarchical_gen.jsonl` - 598 hierarchical generative tasks
- `data/benchmarks/mcq_*.jsonl` - 1,040 PACUTE tasks
- `data/corpora/affix_annotations.jsonl` - 472 morpheme-annotated words

### Scripts
- `scripts/create_affix_annotations.py` - Generate morpheme annotations
- `scripts/generate_hierarchical_benchmark.py` - Create hierarchical tasks
- `scripts/analyze_tokenization_simple.py` - Baseline BPE analysis
- `scripts/compare_tokenizers.py` - Oracle comparison
- `scripts/analyze_affix_coverage.py` - Vocabulary coverage

### Analysis
- `results/tokenization_baseline.json` - GPT-2 baseline metrics
- `results/tokenization_comparison.json` - GPT-2 vs Oracle vs Real Patok
- `data/vocabularies/affix_analysis_*.json` - Coverage analyses

## Attribution

This repository builds upon two existing repositories with proper attribution:

### StochasTok (MIT License)
- **Source**: https://github.com/anyasims/stochastok
- **Fork**: https://github.com/raileymontalan/stochastok
- **Components**: 
  - Tokenization (`src/tokenization/patok_processor.py`, `stochastok_processor.py`)
  - Model architecture (`src/models/`)
  - Training infrastructure (`src/training/`)
  - Data processing (`src/data_processing/`)
  - Evaluation framework (`src/evaluation/benchmarks/`)
- **Paper**: Sims et al. (2025). "Stochastic Tokenization Improves Subword Understanding"

### PACUTE (CC0 1.0 Universal)
- **Source**: Philippine Annotated Corpus for Understanding Tagalog Entities
- **Components**:
  - Task generation (`src/evaluation/affixation.py`, `composition.py`, `manipulation.py`, `syllabification.py`)
  - Morphological operations (`src/evaluation/string_operations.py`, `syllabification_operations.py`)
  - Evaluation data (`data/benchmarks/*.jsonl` - 1,040 items)
  - Linguistic data (`data/corpora/pacute_data/` - syllables, frequencies)

### Key Files from Source Repositories

**From StochasTok:**
- Tokenization: `patok_processor.py`, `stochastok_processor.py`
- Models: Complete transformer implementation in `src/models/`
- Training: Full training framework in `src/training/`
- Experiments: `experiments/train.py`, `experiments/eval.py`
- Configs: `configs/pretraining.yaml`, `configs/instruction_tuning.yaml`
- Data: `data/affixes/filipino_affixes.txt` (93 affixes)

**From PACUTE:**
- Tasks: Affixation, composition, manipulation, syllabification
- Data: `data/benchmarks/*.jsonl` (1,040 evaluation items)
- Operations: String and syllabification primitives
- Corpora: 16,828 syllabified words, 2M+ word frequencies

All original functionality has been preserved:
- ✅ All Python files from both repositories
- ✅ All configuration files
- ✅ All data files
- ✅ All tests and notebooks
- ✅ All documentation

For detailed component attribution, see the full file listing in the repository structure above.

## New Contributions

- Morphology-aware Patok with Aho-Corasick affix detection
- 472 morpheme-annotated Filipino words
- 1,196 hierarchical diagnostic tasks (6 levels)
- Baseline BPE analysis (MorphScore = 0.235)
- Affix coverage analysis (44.6% vocabulary coverage)
- Morphology-aware tokenization comparison (Patok: +179% MorphScore)

## Citation

If you use this repository, please cite:

```bibtex
@misc{africa2025filipino,
  title={Filipino Morphology-Aware Language Model},
  author={Africa, David Demitri and Montalan, Jann Railey and Gamboa, Lance and
          Flores, Richell Isaiah and Layacan, Jimson Paulo and
          Susanto, Yosephine and Ngui, Jian Gang},
  year={2025},
  note={Author order to be determined}
}
```

**Original Sources**:

**StochasTok**:
```bibtex
@misc{sims2025stochastokimprovingfinegrainedsubword,
  title={StochasTok: Improving Fine-Grained Subword Understanding in LLMs},
  author={Anya Sims and Thom Foster and Klara Kaleb and Tuan-Duy H. Nguyen and
          Joseph Lee and Jakob N. Foerster and Yee Whye Teh and Cong Lu},
  year={2025},
  eprint={2506.01687},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.01687}
}
```

**PACUTE** (same as main citation):
```bibtex
@misc{africa2025filipino,
  title={Filipino Morphology-Aware Language Model},
  author={Africa, David Demitri and Montalan, Jann Railey and Gamboa, Lance and
          Flores, Richell Isaiah and Layacan, Jimson Paulo and
          Susanto, Yosephine and Ngui, Jian Gang},
  year={2025},
  note={Author order to be determined}
}
```

## License

Components have different licenses - see [LICENSE](LICENSE) for details:
- StochasTok components: MIT License
- PACUTE components: CC0 1.0 Universal
- New contributions: MIT License
