#!/usr/bin/env python3
"""
Split a large JSONL file into smaller chunks for parallel preprocessing.

This script splits your data so you can preprocess chunks in parallel,
then they can be blended during training.

Usage:
    python scripts/split_jsonl.py --input data.jsonl --output-dir data/chunks --num-chunks 10
"""

import argparse
import json
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
        help="Output directory for chunks",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to create",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Counting lines in {input_path}...")
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    lines_per_chunk = (total_lines + args.num_chunks - 1) // args.num_chunks
    
    print(f"Total lines: {total_lines:,}")
    print(f"Chunks: {args.num_chunks}")
    print(f"Lines per chunk: ~{lines_per_chunk:,}")
    print()
    
    print("Splitting file...")
    chunk_idx = 0
    line_count = 0
    output_file = None
    
    with open(input_path, 'r') as f:
        for line in f:
            if line_count % lines_per_chunk == 0:
                if output_file:
                    output_file.close()
                chunk_idx += 1
                chunk_path = output_dir / f"chunk_{chunk_idx:04d}.jsonl"
                output_file = open(chunk_path, 'w')
                print(f"  Writing {chunk_path.name}...")
            
            output_file.write(line)
            line_count += 1
    
    if output_file:
        output_file.close()
    
    print()
    print(f"✓ Created {chunk_idx} chunks in {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Submit parallel preprocessing:")
    print(f"     qsub jobs/preprocess_data_parallel.pbs")
    print(f"  2. Or process each chunk individually:")
    for i in range(1, chunk_idx + 1):
        chunk_name = f"chunk_{i:04d}"
        print(f"     qsub -v INPUT=/workspace/{output_dir}/{chunk_name}.jsonl,OUTPUT=/workspace/{output_dir}/{chunk_name}_text_document jobs/preprocess_data.pbs")


if __name__ == "__main__":
    main()
