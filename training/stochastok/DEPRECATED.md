# DEPRECATED - Stochastok Training Pipeline

**Status: DEPRECATED** - This training pipeline is no longer functional.

## Why Deprecated

This directory contains a legacy training pipeline that was designed for small-scale GPT-2 experiments. It has the following issues:

1. **Broken Imports**: References non-existent modules:
   - `models.components.base_tokenizer`
   - `trainers.build_trainers`
   - `dataset_preprocessing.utils`

2. **Superseded**: The NeMo-based training pipeline (`training/nemo/`) is now the primary training method.

3. **Incompatible Structure**: This code was written for a different project structure and was never fully integrated.

## Current Training Approach

Use the NeMo Framework training pipeline instead:

```bash
# See training/nemo/ for the current training approach
python training/nemo/run_cpt.py --max-steps 100
```

## If You Need This Code

If you need to revive this training pipeline:
1. The core tokenization processors are still available in `src/tokenization/`
2. You'll need to rewrite the training loop using current project structure
3. Consider using the NeMo approach instead, which supports larger scale training

## Files in This Directory

- `data_processing/` - Data preprocessing (broken imports)
- `experiments/` - Training entry points (broken imports)
- `training/` - Training utilities (broken imports)
- `models/` - Model building (broken imports)
