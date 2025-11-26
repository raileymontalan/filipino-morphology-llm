# Filipino Morphology LLM

A framework for morphological tokenization and evaluation in Filipino and other morphologically rich languages.

## Motivation

Standard subword tokenizers (BPE, WordPiece) are trained on high-resource languages and often misalign with morphological boundaries in morphologically rich languages. For Filipino, a language with extensive affixation (prefixes, infixes, suffixes, circumfixes), this misalignment limits model performance on tasks requiring morphological understanding.

Recent work on English (CUTE, StochasTok) shows that subword-level understanding can be improved through tokenization strategies. However, these methods do not address the specific challenges of morphologically rich languages where affix boundaries are critical for semantic interpretation.

This repository addresses three research questions:

1. Can stochastic tokenization be extended with morphological constraints to better capture affix structure?
2. Do improvements in morphological tokenization transfer to performance on linguistic understanding tasks?
3. Can morphological alignment be quantified using information-theoretic metrics?

## Components

**Patok**: Extends StochasTok with affix-specific processing. During tokenization, applies expand-contract cycles that preferentially form Filipino affixes based on a linguistic affix inventory.

**PACUTE** (Philippine Annotated Corpus for Understanding Tagalog Entities): Evaluation benchmark with 1,040 items testing character manipulation, morpheme decomposition, morpheme manipulation, and morpheme composition. Adapts CUTE methodology to Filipino.

**Hierarchical Tasks**: Six-level diagnostic framework that isolates capability failures (e.g., distinguishing between inability to recognize morphemes vs inability to manipulate them).

**Morphological Metrics**: Quantitative measures including MorphScore (boundary alignment), affix preservation rates, and mutual information between morphemes and tokens.

## Repository Structure

```
filipino-morphology-llm/
├── src/
│   ├── tokenization/       # Patok, StochasTok, affix decomposition
│   ├── models/             # Transformer architecture
│   ├── training/           # Training infrastructure
│   ├── evaluation/         # PACUTE benchmark, hierarchical tasks
│   ├── data_processing/    # Dataset preprocessing
│   └── analysis/           # Morphological and information-theoretic metrics
├── data/
│   ├── affixes/            # 93 Filipino affixes
│   ├── benchmarks/         # 1,040 PACUTE evaluation items
│   ├── corpora/            # Word frequencies, syllabifications
│   └── vocabularies/       # Tokenizer coverage analyses
├── configs/                # Training configurations
├── experiments/            # Training and evaluation scripts
├── notebooks/              # Analysis notebooks
└── scripts/                # Utility scripts
```

## Installation

```bash
git clone https://github.com/DavidDemitriAfrica/filipino-morphology-llm.git
cd filipino-morphology-llm
pip install -r requirements.txt
pip install -e .
```

## Usage

### Tokenization

Apply Patok to a corpus:

```bash
python src/data_processing/patok_expand_contract_dataset.py \
    --input data/corpora/source \
    --output data/processed/patok_tokenized \
    --expand_prop 0.3 \
    --contract_prop 0.3 \
    --affix_preference 0.7
```

Analyze affix coverage across tokenizers:

```bash
python scripts/analyze_affix_coverage.py \
    --compare gpt2 cl100k_base \
    --affixes-file data/affixes/filipino_affixes.txt
```

### Training

```bash
python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=patok_tokenized
```

### Evaluation

Run hierarchical evaluation:

```bash
python experiments/eval.py \
    --checkpoint checkpoints/model.pt \
    --benchmark pacute_hierarchical \
    --output results/
```

Compute morphological metrics:

```python
from src.analysis import MorphologicalMetrics, MorphologicalAnnotation

annotations = [
    MorphologicalAnnotation(
        word="nagluto",
        morphemes=["nag", "luto"],
        morpheme_boundaries=[3],
        affix_types=["prefix", "root"]
    )
]

metrics = MorphologicalMetrics(tokenizer)
morph_score = metrics.compute_morph_score(annotations)
preservation = metrics.compute_affix_preservation_score(annotations)
```

## Patok Algorithm

Patok processes token sequences through iterative expand-contract cycles:

1. **Expand**: Randomly split tokens into sub-tokens (respecting vocabulary constraints)
2. **Contract**: Merge adjacent tokens with preference for forming linguistic affixes
3. **Repeat**: Apply multiple cycles (default: 3)

Affix preference is controlled by a weight parameter (default: 0.7) that biases merging toward combinations that form entries in the Filipino affix inventory.

For out-of-vocabulary affixes, uses a decomposition algorithm that finds optimal sub-token sequences based on morphological validity (CV structure, boundary alignment, morpheme frequency).

## PACUTE Benchmark

### Task Categories

1. **Composition** (280 items): Spelling, character counting, length comparisons
2. **Manipulation** (320 items): Character insertion, deletion, substitution, permutation
3. **Affixation** (280 items): Prefix, suffix, infix, circumfix operations
4. **Syllabification** (160 items): Syllable counting, stress classification

### Hierarchical Levels

- Level 0: Character recognition
- Level 1: Character manipulation
- Level 2: Morpheme decomposition
- Level 3: Morpheme manipulation
- Level 4: Morpheme composition
- Level 5: Complex morphological reasoning

Tasks are structured compositionally such that failure at level N indicates expected failure at level N+1, enabling precise diagnosis of capability gaps.

## Metrics

### Morphological Alignment

**MorphScore**: Proportion of morpheme boundaries that align with token boundaries.

**Affix Preservation**: Frequency with which affixes appear as complete tokens rather than being split.

**Boundary F1**: Precision and recall treating token boundaries as predictions of morpheme boundaries.

**Fragmentation**: Average number of tokens per morpheme.

### Information Theoretic

**Mutual Information I(M;T)**: Quantifies information about morphemes provided by tokenization. Higher values indicate stronger morphological alignment.

**Conditional Entropy H(M|T)**: Uncertainty about morphemes given tokens. Lower values indicate more predictable morphological structure.

**Consistency Entropy**: Variability in how the same affix is tokenized across words. Lower values indicate more consistent treatment.

## Data

**Filipino Affixes**: 93 affixes from linguistic references (Wiktionary, Tagalog grammar sources).

**Syllabifications**: 16,828 words with syllable boundaries and stress annotations.

**Word Frequencies**: 2M+ entries from Filipino corpus data.

**UP Diksiyonaryo**: 67,691 dictionary entries used for validation.

## Citation

```bibtex
@misc{filipino-morphology-llm-2025,
  author = {[Authors]},
  title = {Morphological Tokenization for Low-Resource Languages},
  year = {2025},
  url = {https://github.com/DavidDemitriAfrica/filipino-morphology-llm}
}
```

## References

- CUTE: Edman et al. (2024). "A Character-level Understanding Task for English"
- StochasTok: Sims et al. (2025). "Stochastic Tokenization Improves Subword Understanding"

## License

[To be determined]

## Acknowledgments

Built on StochasTok (Sims et al.) and inspired by CUTE (Edman et al.). Filipino linguistic resources from UP Diksiyonaryo.
