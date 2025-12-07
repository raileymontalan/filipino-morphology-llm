#!/usr/bin/env python3
"""
Preprocess JSONL data to Megatron binary format (.bin + .idx) for NeMo training.

NeMo's PreTrainingDataModule expects Megatron binary format, not raw JSONL.

IMPORTANT: This script must be run INSIDE the NeMo container, not on the host!
The preprocessing tools are only available in the container environment.

Usage (inside container):
    python /workspace/scripts/preprocess_data.py \\
        --input /workspace/data/chunks/chunk_0001.jsonl \\
        --output-prefix /workspace/data/processed/chunk_0001 \\
        --tokenizer-model google/gemma-3-1b-pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL to Megatron binary format"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix (will create <prefix>_text_document.bin and .idx)",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="google/gemma-3-1b-pt",
        help="HuggingFace tokenizer model name",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="JSON key containing the text",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    
    # Verify input exists
    if not input_path.exists():
        print(f"✗ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Megatron Binary Format Preprocessing")
    print("=" * 80)
    print(f"Input:           {args.input}")
    print(f"Output prefix:   {args.output_prefix}")
    print(f"Tokenizer:       {args.tokenizer_model}")
    print(f"Text key:        {args.text_key}")
    print(f"Workers:         {args.workers}")
    print("=" * 80)
    print()
    
    # Use Megatron-LM preprocessing tool (included in NeMo container)
    print("Locating Megatron preprocessing tools...")
    
    # The official Megatron-LM preprocessing script
    preprocess_script = "/opt/megatron-lm/tools/preprocess_data.py"
    
    if not Path(preprocess_script).exists():
        print(f"✗ Error: Megatron preprocessing script not found: {preprocess_script}")
        print()
        print("This script must run inside the NeMo container!")
        print("The PBS job should handle this automatically.")
        sys.exit(1)
    
    print(f"✓ Found preprocessing script: {preprocess_script}")
    print()
    
    # Build the preprocessing command for Megatron-LM
    # Use HuggingFaceTokenizer with --tokenizer-model parameter
    cmd = [
        "python",
        preprocess_script,
        "--input", str(input_path),
        "--output-prefix", args.output_prefix,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", args.tokenizer_model,
        "--json-keys", args.text_key,
        "--workers", str(args.workers),
        "--append-eod",  # Add end-of-document token
    ]
    
    print("Running preprocessing command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        
        # Verify output files were created
        bin_file = Path(f"{args.output_prefix}_text_document.bin")
        idx_file = Path(f"{args.output_prefix}_text_document.idx")
        
        if bin_file.exists() and idx_file.exists():
            print()
            print("=" * 80)
            print("✓ Preprocessing Complete!")
            print("=" * 80)
            print(f"Binary file: {bin_file} ({bin_file.stat().st_size / 1e9:.2f} GB)")
            print(f"Index file:  {idx_file} ({idx_file.stat().st_size / 1e6:.2f} MB)")
            print()
            print("To use in training, specify:")
            print(f"  --data-path {args.output_prefix}")
            print("=" * 80)
            return 0
        else:
            print("✗ Error: Output files not created")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
