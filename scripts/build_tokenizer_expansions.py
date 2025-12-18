#!/usr/bin/env python3
"""
Build tokenizer expansions for both Stochastok and Patok processors.

This script builds and caches expansions for a given tokenizer, which can then
be reused across training runs without rebuilding.

Usage:
    python scripts/build_tokenizer_expansions.py <tokenizer_name>

Examples:
    python scripts/build_tokenizer_expansions.py gpt2
    python scripts/build_tokenizer_expansions.py google/gemma-3-1b-pt
    python scripts/build_tokenizer_expansions.py meta-llama/Llama-3.2-1B
"""

import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from setup_paths import setup_project_paths
from tokenization import MorphologyAwarePatokProcessor, StochastokProcessor
from tokenization.base_processor import TokenizerProcessor

# Setup project paths
setup_project_paths()


def build_expansions(tokenizer_name: str):
    """
    Build tokenizer expansions for both Stochastok and Patok.

    Args:
        tokenizer_name: HuggingFace model identifier (e.g., "gpt2", "google/gemma-3-1b-pt")
    """
    print("=" * 80)
    print("Building Tokenizer Expansions (Stochastok + Patok)")
    print("=" * 80)
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Start time: {os.popen('date').read().strip()}")
    print("=" * 80)
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        sys.exit(1)

    print(f"✓ Loaded tokenizer: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # Get sanitized tokenizer name for checking existing files
    base_processor = TokenizerProcessor(tokenizer)
    sanitized_name = base_processor.tokenizer_name
    print(f"  Sanitized name: {sanitized_name}")
    print()

    # Check if expansions already exist
    expansions_path = base_processor.get_cache_path("expansions", f"expansions_{sanitized_name}.json")
    expansions_exist = os.path.exists(expansions_path)

    if expansions_exist:
        print("=" * 80)
        print("✓ Expansions already exist for this tokenizer!")
        print("=" * 80)
        print(f"Location: {expansions_path}")
        print()
        print("Skipping build. To rebuild, delete the existing file.")
        print("=" * 80)
        return

    # Build expansions (used by both Stochastok and Patok)
    print("=" * 80)
    print("Building tokenizer expansions...")
    print("=" * 80)
    print("(This may take several minutes to hours for large vocabularies)")
    print()

    try:
        # This will build and cache the expansions
        base_processor.set_expansions()
        print()
        print("✓ Expansions built successfully")
        print(f"  Number of expansions: {len(base_processor.expansions)}")
        print(f"  Saved to: {expansions_path}")
        print()
    except Exception as e:
        print(f"✗ Error building expansions: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Verify by initializing both processors
    print("=" * 80)
    print("Verifying processors can load expansions...")
    print("=" * 80)
    print()

    try:
        print("Testing StochastokProcessor...")
        stochastok = StochastokProcessor(tokenizer, expand_prop=0.1)
        print(f"  ✓ StochastokProcessor initialized with {len(stochastok.expansions)} expansions")
        print()

        print("Testing MorphologyAwarePatokProcessor...")
        # Get project root for affix files
        project_root = Path(__file__).parent.parent
        affix_dir = project_root / "data" / "affixes"

        # Check if affix files exist
        prefix_file = affix_dir / "prefixes.txt"
        infix_file = affix_dir / "infixes.txt"
        suffix_file = affix_dir / "suffixes.txt"

        if all(f.exists() for f in [prefix_file, infix_file, suffix_file]):
            patok = MorphologyAwarePatokProcessor(
                tokenizer,
                prefix_file=str(prefix_file),
                infix_file=str(infix_file),
                suffix_file=str(suffix_file),
            )
            print(f"  ✓ MorphologyAwarePatokProcessor initialized with {len(patok.expansions)} expansions")
            print(f"    and {len(patok.affix_ids)} affix IDs")
        else:
            print("  ⚠️  Affix files not found, testing without morphology awareness")
            patok = MorphologyAwarePatokProcessor(tokenizer)
            print(f"  ✓ MorphologyAwarePatokProcessor initialized with {len(patok.expansions)} expansions")
        print()
    except Exception as e:
        print(f"  ✗ Error verifying processors: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✓ All expansions built and verified!")
    print("=" * 80)
    print(f"End time: {os.popen('date').read().strip()}")
    print("=" * 80)


def main():
    """Execute the main tokenizer expansion building process."""
    if len(sys.argv) < 2:
        print("Error: TOKENIZER_NAME not provided")
        print()
        print("Usage:")
        print("  python scripts/build_tokenizer_expansions.py <tokenizer_name>")
        print()
        print("Examples:")
        print("  python scripts/build_tokenizer_expansions.py gpt2")
        print("  python scripts/build_tokenizer_expansions.py google/gemma-3-1b-pt")
        print("  python scripts/build_tokenizer_expansions.py meta-llama/Llama-3.2-1B")
        sys.exit(1)

    tokenizer_name = sys.argv[1]
    build_expansions(tokenizer_name)


if __name__ == "__main__":
    main()
