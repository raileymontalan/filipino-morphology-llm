# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project testing whether **morpheme-aware tokenization** improves LLM performance on Filipino morphological tasks. Compares three tokenization approaches (Baseline BPE, Stochastok expansion, Patok affix-aware) on Gemma 3 1B using SEA-PILE v2 Filipino corpus (7.4GB) and 15,023 morphological evaluation tasks.

## Common Development Commands

### Environment Setup (One-Time)
```bash
# Configure environment variables
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY, ENROOT_PATH, BIND_MOUNTS
source .env

# Install NeMo container (~15 minutes)
bash training/nemo/setup/setup_enroot.sh

# Verify installation
enroot list | grep nemo_framework
./run_in_enroot.sh python -c "import nemo; print(nemo.__version__)"
```

### Data Preprocessing
```bash
# Test preprocessing single chunk first
qsub jobs/preprocess_test_chunk1.pbs

# Full parallel preprocessing (20x faster)
qsub -J 1-20 jobs/preprocess_data_parallel.pbs

# Monitor progress
qstat -t

# Verify outputs
ls -lh data/processed/*.bin | wc -l  # Should show 20
```

### Training
```bash
# Quick test (10 steps)
qsub jobs/run_cpt_test.pbs

# Full training
qsub jobs/run_cpt.pbs

# Custom hyperparameters
qsub -v MAX_STEPS=5000,GBS=512,LR=5e-5 jobs/run_cpt.pbs

# Monitor training
qstat -u $USER
tail -f nemo_experiments/<experiment_name>/nemo_log_globalrank-0_localrank-0.txt
```

### Evaluation
```bash
# Generate all benchmarks
python scripts/generate_benchmarks.py

# Evaluate model
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute

# Evaluate specific format (MCQ or GEN)
python scripts/run_evaluation.py --models gpt2 --eval-mode mcq

# View results
cat results/benchmark_evaluation/*.json
```

### Testing
```bash
# Run single test file
pytest tests/test_patok_morphology.py -v

# Run specific test
pytest tests/test_affixation.py::test_affix_identification -v

# Run all tests
pytest tests/ -v
```

## Architecture Overview

### Tokenization Pipeline

The core innovation is comparing three tokenization approaches:

1. **Baseline (Vanilla BPE)**: Standard tokenization using HuggingFace tokenizers
2. **Stochastok**: Stochastic token expansion (~10%) - randomly splits tokens into subword units during training
3. **Patok**: Affix-aware expand-contract (30%+30%) - preferentially splits/merges at morpheme boundaries

**Key Files:**
- `src/tokenization/base_processor.py` - Base class for tokenization processors
- `src/tokenization/stochastok_processor.py` - Implements stochastic expansion
- `src/tokenization/patok_processor.py` - Implements affix-aware processing with Filipino morphology
- `src/tokenization/patok_morphology.py` - Filipino affix detection and morpheme boundary identification

**Critical Implementation Detail**: Tokenization processors apply DURING preprocessing, not during model training. The `preprocess_data.py` script uses these processors to create different versions of the training data:
- Vanilla: `data/processed/google-gemma-3-1b-pt/`
- Stochastok: `data/processed/google-gemma-3-1b-pt_stochastok_0.1/`
- Patok: `data/processed/google-gemma-3-1b-pt_patok/` (future work)

### Training Architecture

Training uses **NeMo Framework 2.1+** which requires:
- **Megatron binary format** (.bin + .idx files) - NOT raw JSONL or HuggingFace Arrow
- Preprocessing MUST run inside the NeMo container where Megatron tools are available
- Data is memory-mapped for efficient multi-GPU distributed training

**Key Files:**
- `training/nemo/run_cpt.py` - Main CPT training script using NeMo 2.0 API
- `training/nemo/data/preprocess_data.py` - Converts JSONL → Megatron binary format
- `training/nemo/data/split_jsonl.py` - Splits large corpus into chunks for parallel preprocessing

**Critical Architecture Decision**: NeMo 2.0 uses a different API than NeMo 1.x:
- Uses `nemo.collections.llm` package (not `nemo_nlp`)
- Model definition via `llm.GemmaConfig3()` and `llm.GemmaModel3()`
- Training via `nl.Trainer()` with strategy="ddp"
- Data loading via `PreTrainingDataModule` expecting Megatron format

**Gemma3 Monkey Patch**: The codebase includes a critical monkey patch for `Gemma3SelfAttention.forward()` to accept `rotary_pos_cos_sin` parameter that gets passed by transformer layer but isn't in the original signature. This is applied in `training/nemo/run_cpt.py` lines 62-106. See `docs/GEMMA3_MONKEY_PATCH.md` for details.

### Evaluation Architecture

Evaluation uses a **hierarchical diagnostic framework** to identify where models fail:

**Benchmark Hierarchy:**
- **PACUTE** (4,080 tasks): Filipino-specific morphology tests
  - Affixation: Identify/apply Filipino affixes (prefixes, infixes, suffixes)
  - Composition: Character counting, diacritics, word formation
  - Manipulation: Insert/delete/swap/replace characters
  - Syllabification: Syllable counting, stress, reduplication

- **Hierarchical** (1,198 tasks): 6-level diagnostic cascade
  - Level 0: Character Recognition → Level 1: Character Manipulation
  - Level 2: **Morpheme Decomposition** (critical bottleneck) → Level 3: Morpheme Manipulation
  - Level 4: Morpheme Composition → Level 5: Complex Reasoning
  - **Key Insight**: Level 2 failures cascade through Levels 3-5, making morpheme boundary understanding critical

- **CUTE** (1,400 tasks): Character understanding across 14 task types
- **LangGame** (2,000 tasks): Subword understanding via word games
- **Multi-digit Addition** (2,000 tasks): Numerical reasoning baseline

**Format Types:**
- **MCQ**: Log probability-based selection from 4 options
- **GEN**: Free-form generation with exact match scoring

**Key Files:**
- `src/evaluation/loaders/` - Benchmark loading (pacute.py, hierarchical.py, etc.)
- `src/evaluation/datasets/generators/` - Generate benchmark tasks
- `src/evaluation/evaluators/` - Evaluation logic (MCQ vs GEN)
- `scripts/generate_benchmarks.py` - Generate all benchmark JSONL files
- `scripts/run_evaluation.py` - Run evaluation on models

**Critical Evaluation Pattern**: Each benchmark supports filtering by evaluation mode via `--eval-mode mcq|gen|both`. Generators create both MCQ and GEN versions with task IDs linking equivalent questions.

### PBS Job System

The codebase uses **template-based PBS job generation** to avoid committing cluster-specific paths:

- **Templates** (version-controlled): `job_templates/*.template.pbs` with placeholder variables
- **Generated Jobs** (gitignored): `jobs/*.pbs` with actual paths substituted
- **Setup Wizard**: `job_templates/setup_jobs.sh` - Interactive script to generate jobs from templates

**Environment Variables Pattern**: All cluster-specific configuration goes in `.env`:
```bash
HF_HOME=/scratch/$USER/cache/huggingface
WANDB_DIR=/scratch/$USER/logs/wandb
ENROOT_PATH=/scratch/$USER/enroot/
BIND_MOUNTS=/scratch/$USER/cache:/cache,/scratch/$USER/logs:/logs
```

This `.env` file is sourced by PBS jobs and provides configuration to the container runtime.

### Container Architecture

Two container runtimes supported (Enroot preferred):

**Enroot** (Recommended):
- Uses `.sqsh` (squashfs) images stored at `$ENROOT_PATH`
- Container managed via `enroot` commands
- Executed via `./run_in_enroot.sh` wrapper script
- GPU access automatic (no special flags needed)

**Singularity/Apptainer** (Alternative):
- Uses `.sif` (Singularity Image Format) files
- Executed via `./run_in_singularity.sh` with `--nv` flag for GPU
- Container stored at `$CONTAINER_CACHEDIR`

**Critical Container Workflow**:
1. Container setup is ONE-TIME per user (creates reusable image)
2. Preprocessing MUST run inside container (Megatron tools only available there)
3. Training MUST run inside container (NeMo + optimizations)
4. Evaluation CAN run outside container (only needs transformers)

**Mount Points**: Project directory auto-mounted as `/workspace`, shared directories mounted via `$BIND_MOUNTS`:
- `/workspace` → current project directory
- `/cache` → HuggingFace model cache
- `/logs` → training outputs

## Code Organization Patterns

### Tokenization Processor Pattern
All tokenization processors inherit from `TokenizerProcessor` base class:
```python
class PatokProcessor(TokenizerProcessor):
    def __init__(self, tokenizer, expand_prop=0.3, contract_prop=0.3, affix_preference=0.7):
        super().__init__(tokenizer)
        self.set_expansions()
        self.set_contractions()
        self.load_affixes(affixes_file)

    def expand(self, token_ids, expand_prop, max_num_to_expand, disable_tqdm):
        # Split tokens at morpheme boundaries

    def contract(self, token_ids, contract_prop, max_num_to_contract, disable_tqdm):
        # Merge tokens respecting morpheme boundaries
```

Processors are applied DURING preprocessing, not during training.

### Benchmark Generator Pattern
Benchmark generators follow consistent structure:
```python
def generate_<benchmark_name>_benchmark(num_samples=1000, seed=42):
    """Generate benchmark tasks in standardized format."""
    tasks = []
    for i in range(num_samples):
        task = {
            "id": f"{benchmark_name}_{i}",
            "question": "...",
            "answer": "...",
            "options": ["A", "B", "C", "D"],  # MCQ only
            "category": "...",
            "difficulty": "easy|medium|hard"
        }
        tasks.append(task)
    return tasks
```

All benchmarks saved as JSONL in `data/benchmarks/`.

### Evaluation Loader Pattern
Benchmark loaders registered via decorator:
```python
from evaluation.loaders.registry import register_loader

@register_loader("pacute")
def load_pacute(benchmark_name, eval_mode="both", max_samples=None):
    """Load PACUTE benchmark variants with format filtering."""
    # Load MCQ and/or GEN versions based on eval_mode
    return tasks
```

This allows `run_evaluation.py` to dynamically load benchmarks via `load_benchmark(name)`.

## Important Development Constraints

### Data Format Requirements
- **Training input**: JSONL with `{"text": "..."}` format (one document per line)
- **Preprocessing output**: Megatron binary (`.bin` + `.idx` pairs)
- **Training data paths**: Must be prefixes WITHOUT `_text_document` suffix
- **Benchmark format**: JSONL with standardized task schema

### Container Workflow Requirements
- **NEVER** run preprocessing on login node (needs container)
- **ALWAYS** source `.env` before submitting PBS jobs
- **NEVER** commit generated helper scripts (`run_in_enroot.sh`, `run_in_singularity.sh`)
- **ALWAYS** use PBS jobs for preprocessing and training (not interactive)

### Security Requirements
- **NEVER** commit `.env` file (use `.env.example` for templates)
- **NEVER** hardcode cluster paths in version-controlled files
- **NEVER** commit API keys, tokens, or secrets
- **ALWAYS** use templates for PBS jobs with placeholder variables

### Testing Requirements
- Test preprocessing with single chunk before full 20-chunk processing
- Test training with `run_cpt_test.pbs` (10 steps) before full runs
- Test evaluation with `--max-samples 100` before full benchmarks

## File Structure Patterns

```
filipino-morphology-llm/
├── src/                          # Source code (importable package)
│   ├── tokenization/            # Tokenization processors
│   │   ├── base_processor.py
│   │   ├── stochastok_processor.py
│   │   └── patok_processor.py
│   └── evaluation/              # Evaluation framework
│       ├── loaders/             # Benchmark loaders (registry pattern)
│       ├── datasets/generators/ # Task generators
│       └── evaluators/          # Evaluation logic
├── training/                     # Training implementations
│   └── nemo/                    # NeMo Framework CPT
│       ├── run_cpt.py           # Main training script
│       ├── data/                # Data preprocessing
│       │   ├── preprocess_data.py
│       │   └── split_jsonl.py
│       └── setup/               # Container setup
├── scripts/                      # Executable utilities
│   ├── generate_benchmarks.py   # Generate evaluation datasets
│   └── run_evaluation.py        # Run model evaluation
├── job_templates/               # PBS job templates (version-controlled)
├── jobs/                        # Generated PBS jobs (gitignored)
├── data/                        # Data files
│   ├── benchmarks/             # Evaluation tasks (JSONL)
│   ├── chunks/                 # Split training data
│   ├── processed/              # Megatron binary format
│   └── affixes/                # Filipino affix lists
└── docs/                        # Documentation
```

## Common Development Workflows

### Adding a New Tokenization Method
1. Create processor in `src/tokenization/<name>_processor.py` inheriting from `TokenizerProcessor`
2. Implement `expand()` and optionally `contract()` methods
3. Add tokenization mode to `preprocess_data.py` argument parser
4. Add conditional logic in `preprocess_data.py` to instantiate processor
5. Test with single chunk preprocessing
6. Generate full preprocessed dataset
7. Train model with new tokenization
8. Evaluate and compare results

### Adding a New Benchmark
1. Create generator in `src/evaluation/datasets/generators/<name>.py`
2. Implement `generate_<name>_benchmark()` returning list of standardized tasks
3. Add loader in `src/evaluation/loaders/<name>.py` with `@register_loader` decorator
4. Add to `scripts/generate_benchmarks.py` generation workflow
5. Test generation: `python scripts/generate_benchmarks.py --benchmarks <name>`
6. Add to evaluation: modify `scripts/run_evaluation.py` benchmark list
7. Run evaluation: `python scripts/run_evaluation.py --benchmarks <name>`

### Debugging Training Issues
1. Check container exists: `enroot list | grep nemo_framework`
2. Verify preprocessed data exists: `ls -lh data/processed/*.bin`
3. Test interactively: `./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 10`
4. Check training logs: `tail -f nemo_experiments/<name>/nemo_log_*.txt`
5. Monitor W&B dashboard for metrics
6. Check for Gemma3 bugs: see `docs/GEMMA3_MONKEY_PATCH.md`

### Comparing Tokenization Methods
1. Preprocess same data with different modes (vanilla vs stochastok vs patok)
2. Train models with same hyperparameters on each tokenized dataset
3. Evaluate all models on same benchmarks
4. Focus comparison on:
   - PACUTE Affixation (morpheme understanding)
   - Hierarchical Level 2 (morpheme decomposition bottleneck)
   - PACUTE Manipulation (character operations)
5. Expected improvements: Patok > Stochastok > Baseline

## Key Documentation References

- `README.md` - Project overview, quick start, expected results
- `SETUP.md` - Complete environment setup and container installation
- `docs/RESEARCH.md` - Research question, tokenization methods, experimental design
- `docs/TRAINING.md` - Training workflows, PBS jobs, monitoring
- `docs/EVALUATION.md` - Benchmark generation, evaluation procedures, analysis
- `docs/GEMMA3_MONKEY_PATCH.md` - Known Gemma3 bugs and workarounds
- `training/nemo/data/DATA_PREPROCESSING.md` - Data preprocessing guide
- `job_templates/README.md` - PBS job template system

## Model-Specific Notes

### Gemma3 Issues
- `Gemma3SelfAttention.forward()` signature mismatch requires monkey patch (applied automatically in `run_cpt.py`)
- Gemma3 uses custom local/global RoPE, not standard rotary embeddings
- See `docs/GEMMA3_MONKEY_PATCH.md` for full details and workarounds

### NeMo 2.0 API Changes
- Use `nemo.collections.llm` not `nemo.collections.nlp`
- Model config via `llm.GemmaConfig3()` not YAML files
- Trainer via `nl.Trainer()` with Lightning 2.0 API
- Data module must be `PreTrainingDataModule` with Megatron format

### Preprocessing Requirements
- Must specify `--tokenizer-type HuggingFaceTokenizer` with `--tokenizer-model`
- Text key must match JSONL structure (default: "text")
- Output prefix must NOT include `_text_document` suffix
- Workers should be 64 for optimal parallel tokenization
