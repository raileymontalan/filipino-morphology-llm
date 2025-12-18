"""
Convert Benchmark JSONL Files to Memmap Format.

This script converts all JSONL benchmark files to memory-mapped binary format
for efficient training. Each JSONL file is tokenized and saved as a .bin file
along with metadata.

Usage:
    python src/evaluation/benchmark_generation/convert_benchmarks_to_memmaps.py

Output:
    data/memmaps/<benchmark_name>/
        - <benchmark_name>.bin (memory-mapped tokens)
        - <benchmark_name>_metadata.json (shape, dtype, etc.)
"""

import sys
from pathlib import Path

from evaluation.datasets.converters.converters import (
    convert_benchmark_directory_to_memmaps,
)

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class SimpleTokenizer:
    """
    Simple character-level tokenizer for benchmarks.
    This can be replaced with any tokenizer that has an encode() method.
    """

    def __init__(self):
        # Build vocabulary from printable ASCII
        self.char_to_id = {chr(i): i for i in range(128)}
        self.id_to_char = {i: chr(i) for i in range(128)}
        self.pad_token_id = 0

    def encode(self, text):
        """Encode text to list of token IDs."""
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        """Decode list of token IDs to text."""
        return "".join([self.id_to_char.get(i, "?") for i in ids])


def main():
    """Convert all benchmark JSONL files to memmap format."""
    print("=" * 70)
    print("CONVERTING BENCHMARKS TO MEMMAP FORMAT")
    print("=" * 70)

    # Paths
    benchmark_dir = project_root / "data" / "benchmarks"
    memmap_dir = project_root / "data" / "memmaps"

    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Memmap output directory: {memmap_dir}")
    print()

    # Check if benchmark directory exists
    if not benchmark_dir.exists():
        print(f"✗ Benchmark directory not found: {benchmark_dir}")
        print("  Run benchmark generation scripts first.")
        return 1

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    print(f"✓ Using SimpleTokenizer (character-level, vocab size: {len(tokenizer.char_to_id)})")
    print()

    # Option to use a real tokenizer
    use_real_tokenizer = False
    if use_real_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print(f"✓ Using GPT-2 tokenizer (vocab size: {len(tokenizer)})")
        except Exception as e:
            print(f"⚠ Could not load transformers tokenizer: {e}")
            print("  Falling back to SimpleTokenizer")
            tokenizer = SimpleTokenizer()

    # Convert all benchmarks
    print("Converting JSONL files to memmaps...")
    print()

    try:
        results = convert_benchmark_directory_to_memmaps(
            benchmark_dir,
            memmap_dir,
            tokenizer,
            pattern="*.jsonl",
            text_field="question",  # Default field to tokenize
            pad_token_id=(tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0),
            save_metadata=True,
        )

        # Summary
        print()
        print("=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)

        if results:
            print(f"✓ Successfully converted {len(results)} benchmark files")
            print(f"✓ Memmaps saved to: {memmap_dir}")
            print()
            print("Converted files:")
            for filename, paths in results.items():
                print(f"  - {filename}")
                for key, path in paths.items():
                    print(f"    → {key}: {path}")
        else:
            print("✗ No files were converted")
            return 1

        print()
        print("=" * 70)
        print("NOTE: These memmaps use a simple character-level tokenizer.")
        print("For training, you may want to regenerate with your model's tokenizer.")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
