#!/usr/bin/env python3
"""
Split a large JSONL file into smaller chunks for parallel preprocessing.

This script splits your data into chunks of 2^16 (65,536) lines each,
so you can preprocess chunks in parallel, then they can be blended during training.

Usage:
    python scripts/split_jsonl.py --input data.jsonl --output-dir data/chunks
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    input_filename = input_path.stem
    output_dir = Path(args.output_dir) / input_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Counting lines in {input_path}...")
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Fixed chunk size: 2^16 lines (65,536)
    lines_per_chunk = 2**16
    num_chunks = (total_lines + lines_per_chunk - 1) // lines_per_chunk
    
    print(f"Total lines: {total_lines:,}")
    print(f"Lines per chunk: {lines_per_chunk:,}")
    print(f"Chunks: {num_chunks}")
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
    print(f"  1. Submit parallel preprocessing (recommended):")
    print(f"     qsub -J 1-{chunk_idx} -v TOKENIZER=tokenizer jobs/preprocess_data_parallel.pbs")
    print()
    print(f"  2. Then use chunks in training:")
    print(f"     export DATA_PATH=$(training/nemo/data/generate_chunk_paths.sh {chunk_idx} tokenizer)")
    print(f"     qsub jobs/run_cpt.pbs")


if __name__ == "__main__":
    main()
