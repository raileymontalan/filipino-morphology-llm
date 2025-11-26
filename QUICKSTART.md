# Quick Start

## Setup

```bash
cd /home/ubuntu/filipino-morphology-llm
pip install -r requirements.txt
pip install -e .
python scripts/verify_setup.py
```

## Basic Usage

### Tokenize with Patok

```python
from src.tokenization import PatokProcessor

processor = PatokProcessor(
    base_tokenizer="gpt2",
    affixes_file="data/affixes/filipino_affixes.txt",
    expand_prop=0.3,
    contract_prop=0.3,
    affix_preference=0.7
)

text = "Nagluto siya ng masarap na pagkain."
tokens = processor.process(text)
```

### Evaluate on PACUTE

```python
from src.evaluation import (
    create_affixation_dataset,
    create_composition_dataset
)

affixation_tasks = create_affixation_dataset(
    n_inflection=50,
    n_identification=50,
    output_dir="data/benchmarks/custom/",
    format="mcq"
)
```

### Train Model

```bash
python src/data_processing/patok_expand_contract_dataset.py \
    --dataset_name openwebtext \
    --output_dir data/processed/ \
    --expand_prop 0.3 \
    --contract_prop 0.3 \
    --affix_preference 0.7

python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=processed/openwebtext-patok
```

### Compare Tokenizations

```python
import tiktoken
from src.tokenization import PatokProcessor, StochastokProcessor

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

stochastok = StochastokProcessor(gpt2_tokenizer, expand_prop=0.1)
patok = PatokProcessor(
    base_tokenizer="gpt2",
    affixes_file="data/affixes/filipino_affixes.txt",
    expand_prop=0.3,
    contract_prop=0.3,
    affix_preference=0.7
)

text = "Pinagmamalaki niya ang kanyang mga anak."

baseline_tokens = gpt2_tokenizer.encode(text)
stochastok_tokens = stochastok.process_tokens(baseline_tokens)
patok_tokens = patok.process(text)
```

### Morphological Analysis

```python
from src.analysis import (
    morpheme_token_mutual_information,
    affix_consistency_entropy
)
import pandas as pd

words_df = pd.read_json("data/corpora/pacute_data/syllables.jsonl", lines=True)

mi_score = morpheme_token_mutual_information(patok, words_df)
consistency = affix_consistency_entropy(patok, words_df["word"].tolist())
```

## Training Pipeline

```bash
# Preprocess
python src/data_processing/tokenize_dataset.py \
    --dataset openwebtext \
    --output data/processed/openwebtext-base

# Apply Patok
python src/data_processing/patok_expand_contract_dataset.py \
    --input data/processed/openwebtext-base \
    --output data/processed/openwebtext-patok

# Train
python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=openwebtext-patok \
    trainer.save_dir=checkpoints/patok_model

# Evaluate
python experiments/eval.py \
    --checkpoint checkpoints/patok_model/final.pt \
    --benchmarks pacute,winogrande,hellaswag
```

## Generate PACUTE Benchmark

```python
from src.evaluation import (
    create_affixation_dataset,
    create_composition_dataset,
    create_manipulation_dataset,
    create_syllabification_dataset,
    load_frequency_data,
    sample_by_frequency
)

word_freq_df = load_frequency_data("data/corpora/pacute_data/word_frequencies.csv")
syllables_df = pd.read_json("data/corpora/pacute_data/syllables.jsonl", lines=True)

sampled_words = sample_by_frequency(
    word_freq_df,
    n_samples=100,
    freq_weight=0.5
)

custom_tasks = create_affixation_dataset(
    words_df=sampled_words,
    n_inflection=50,
    n_identification=50,
    output_dir="data/benchmarks/custom/",
    format="both"
)
```

## Configuration

Override parameters via Hydra:

```bash
python experiments/train.py \
    model.n_layers=12 \
    model.hidden_dim=768 \
    trainer.batch_size=64 \
    trainer.learning_rate=3e-4 \
    tokenization.affix_preference=0.8
```

Or create custom config:

```yaml
# configs/custom.yaml
defaults:
  - pretraining

model:
  n_layers: 12
  hidden_dim: 768

trainer:
  batch_size: 64
  learning_rate: 3e-4
```

```bash
python experiments/train.py --config-name custom
```

## Testing

```bash
pytest tests/
pytest tests/test_affix_processor.py
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Import Errors

```python
# Ensure package is installed
pip install -e /home/ubuntu/filipino-morphology-llm

# Or use explicit imports
from src.tokenization import PatokProcessor
```

### Path Issues

```python
from pathlib import Path
repo_root = Path(__file__).parent.parent
affix_file = repo_root / "data/affixes/filipino_affixes.txt"
```

### CUDA Errors

```bash
python -c "import torch; print(torch.cuda.is_available())"
python experiments/train.py trainer.device=cpu
```

## Resources

- StochasTok: https://arxiv.org/abs/2506.01687
- CUTE: https://aclanthology.org/2024.emnlp-main.177/
- UP Diksiyonaryo: https://updiksiyonaryo.ph/
