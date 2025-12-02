# Filipino Morphology LLM

Unified framework for morphological tokenization and evaluation in Filipino. Merges [StochasTok](https://github.com/anyasims/stochastok) (morphologically-aware tokenization) with [PACUTE](https://github.com/DavidDemitriAfrica/pacute) (Filipino linguistic evaluation).

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

### Oracle Analysis

We tested an oracle tokenizer that splits at known morpheme boundaries before applying BPE:

| Metric | GPT-2 | Oracle | Δ |
|--------|-------|--------|---|
| MorphScore | 0.235 | 0.990 | +0.755 |
| Boundary F1 | 0.165 | 0.643 | +0.478 |

**Finding**: Respecting morpheme boundaries could improve alignment by 320%, establishing an upper bound for what morphologically-aware tokenization might achieve.

**Note**: This is an oracle experiment (we provided the boundaries). Real Patok would need to learn these during training.

## Repository Structure

```
filipino-morphology-llm/
├── src/
│   ├── tokenization/          # Patok, StochasTok, affix decomposition
│   ├── evaluation/            # PACUTE + hierarchical tasks
│   ├── analysis/              # Morphological metrics
│   ├── models/                # Transformer architecture
│   ├── training/              # Training infrastructure
│   └── data_processing/       # Dataset preprocessing
├── data/
│   ├── affixes/               # Filipino affix list
│   ├── benchmarks/            # 2,236 evaluation items
│   ├── corpora/               # Annotations, syllables, frequencies
│   └── vocabularies/          # Tokenizer analyses
├── experiments/               # Training and evaluation scripts
├── scripts/                   # Analysis utilities
└── configs/                   # Training configurations
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
- `results/tokenization_comparison.json` - Oracle vs baseline
- `data/vocabularies/affix_analysis_*.json` - Coverage analyses

## Attribution

This repository merges two existing repositories:

**StochasTok** (MIT License)
- Source: https://github.com/anyasims/stochastok
- Components: Tokenization, models, training, data processing
- Paper: Sims et al. (2025). "Stochastic Tokenization Improves Subword Understanding"

**PACUTE** (CC0 1.0 Universal)
- Source: Philippine Annotated Corpus for Understanding Tagalog Entities
- Components: Evaluation tasks, morphological operations, benchmarks
- Data: 1,040 evaluation items, 16,828 syllabified words

See [ORIGINAL_SOURCES.md](ORIGINAL_SOURCES.md) for detailed attribution.

## New Contributions

- Morphology-aware Patok with Aho-Corasick affix detection
- 472 morpheme-annotated Filipino words
- 1,196 hierarchical diagnostic tasks (6 levels)
- Baseline BPE analysis (MorphScore = 0.235)
- Affix coverage analysis (44.6% vocabulary coverage)
- Oracle upper bound analysis (MorphScore = 0.990)

## Citation

If you use this repository, please cite:

**StochasTok**:
```bibtex
@article{sims2025stochastok,
  title={Stochastic Tokenization Improves Subword Understanding},
  author={Sims, Anya and others},
  year={2025}
}
```

**PACUTE**:
```bibtex
@misc{pacute2024,
  title={Philippine Annotated Corpus for Understanding Tagalog Entities},
  year={2024}
}
```

## License

Components have different licenses - see [LICENSE](LICENSE) for details:
- StochasTok components: MIT License
- PACUTE components: CC0 1.0 Universal
- New contributions: MIT License
