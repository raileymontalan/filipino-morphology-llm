#!/usr/bin/env python3
"""
Split a large JSONL file into smaller chunks for parallel preprocessing.

This script splits your data so you can preprocess chunks in parallel,
then they can be blended during training.

Usage:
    python scripts/split_jsonl.py --input data.jsonl --output-dir data/chunks --num-chunks 10
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split JSONL into chunks")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for chunks (will create subdirectory based on tokenizer)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to create",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-3-1b-pt",
        help="Tokenizer name (used for organizing output directory)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    # Create tokenizer-specific subdirectory
    tokenizer_name = args.tokenizer.replace("/", "-")
    output_dir = Path(args.output_dir) / tokenizer_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Counting lines in {input_path}...")
    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)

    lines_per_chunk = (total_lines + args.num_chunks - 1) // args.num_chunks

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Total lines: {total_lines:,}")
    print(f"Chunks: {args.num_chunks}")
    print(f"Lines per chunk: ~{lines_per_chunk:,}")
    print()

    print("Splitting file...")
    chunk_idx = 0
    line_count = 0
    output_file = None

    with open(input_path, "r") as f:
        for line in f:
            if line_count % lines_per_chunk == 0:
                if output_file:
                    output_file.close()
                chunk_idx += 1
                chunk_path = output_dir / f"chunk_{chunk_idx:04d}.jsonl"
                output_file = open(chunk_path, "w")
                print(f"  Writing {chunk_path.name}...")

            output_file.write(line)
            line_count += 1

    if output_file:
        output_file.close()

    print()
    print(f"✓ Created {chunk_idx} chunks in {output_dir}")
    print()
    print("Next steps:")
    print("  1. Submit parallel preprocessing (recommended):")
    print(f"     qsub -J 1-{chunk_idx} -v TOKENIZER={args.tokenizer} jobs/preprocess_data_parallel.pbs")
    print()
    print("  2. Then use chunks in training:")
    print(f"     export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh {chunk_idx} {args.tokenizer})")
    print("     qsub jobs/run_cpt.pbs")


if __name__ == "__main__":
    main()
