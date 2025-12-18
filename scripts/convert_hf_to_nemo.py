#!/usr/bin/env python3
"""
Convert HuggingFace checkpoint to NeMo format (single GPU).

This script must be run BEFORE distributed training to avoid
race conditions when multiple ranks try to import simultaneously.

Usage:
    ./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b
"""

import argparse
import sys
from pathlib import Path


def main():
    """Convert HuggingFace model checkpoint to NeMo format for training."""
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to NeMo format")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace model ID to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/checkpoints/nemo",
        help="Output directory for NeMo checkpoint",
    )
    args = parser.parse_args()

    # Verify we're in the container
    try:
        import nemo
        import nemo.collections.llm as llm

        print(f"Running in NeMo Framework {nemo.__version__}")
    except ImportError as e:
        print(f"Error: {e}")
        print("This script must run inside the NeMo Framework container.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model config based on model name
    model_name = args.model.lower()

    if "gemma-2-2b" in model_name:
        print("Using Gemma2Config2B configuration")
        config = llm.Gemma2Config2B(
            seq_length=2048,
            vocab_size=256128,
        )
        model = llm.Gemma2Model(config=config)
    elif "gemma-2-9b" in model_name:
        print("Using Gemma2Config9B configuration")
        config = llm.Gemma2Config9B(
            seq_length=2048,
            vocab_size=256128,
        )
        model = llm.Gemma2Model(config=config)
    elif "llama-3.2-1b" in model_name or "llama-3.2-1B" in model_name:
        print("Using Llama32Config1B configuration")
        config = llm.Llama32Config1B(
            seq_length=2048,
            vocab_size=128256,
        )
        model = llm.LlamaModel(config=config)
    elif "llama-3.2-3b" in model_name or "llama-3.2-3B" in model_name:
        print("Using Llama32Config3B configuration")
        config = llm.Llama32Config3B(
            seq_length=2048,
            vocab_size=128256,
        )
        model = llm.LlamaModel(config=config)
    else:
        print(f"Error: Unknown model {args.model}")
        print(
            "Supported models: google/gemma-2-2b, google/gemma-2-9b, meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B"
        )
        sys.exit(1)

    # Import checkpoint from HuggingFace
    print(f"\nImporting checkpoint from HuggingFace: {args.model}")
    print("This may take a few minutes...")

    try:
        imported_path = llm.import_ckpt(
            model=model,
            source=f"hf://{args.model}",
            output_path=str(output_dir / args.model.replace("/", "_")),
        )
        print("\n✓ Successfully converted checkpoint!")
        print(f"  NeMo checkpoint saved to: {imported_path}")
        print("\nTo use in training, run:")
        print("  ./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \\")
        print(f"    --resume-from {imported_path} \\")
        print("    --data-path /workspace/data/processed/vanilla/chunk_001_text_document ...")
    except Exception as e:
        print(f"\n✗ Error converting checkpoint: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
