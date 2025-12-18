#!/usr/bin/env python3
"""
Test script to verify StochastokProcessor still works correctly.
"""

from transformers import AutoTokenizer

from tokenization.stochastok_processor import StochastokProcessor
from setup_paths import setup_project_paths

setup_project_paths()


def test_stochastok():
    """Test that StochastokProcessor works correctly."""

    print("=" * 80)
    print("Testing StochastokProcessor")
    print("=" * 80)
    print()

    # Load a small tokenizer for testing
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"✓ Loaded tokenizer: {tokenizer.__class__.__name__}")
    print()

    # Initialize processor
    print("Initializing StochastokProcessor...")
    processor = StochastokProcessor(tokenizer=tokenizer, expand_prop=0.1)
    print()

    # Test inherited attributes
    print("Testing inherited attributes from TokenizerProcessor:")
    print(f"  - tokenizer: {type(processor.tokenizer).__name__}")
    print(f"  - tokenizer_name: {processor.tokenizer_name}")
    print(f"  - expansions (type): {type(processor.expansions).__name__}")
    print(f"  - expansions (size): {len(processor.expansions)}")
    print()

    # Test Stochastok-specific attributes
    print("Testing Stochastok-specific attributes:")
    print(f"  - expand_prop: {processor.expand_prop}")
    print()

    # Test expansion format
    print("Testing expansion format (should be tuples):")
    sample_token_id = list(processor.expansions.keys())[0]
    sample_expansions = processor.expansions[sample_token_id]
    print(f"  Token ID {sample_token_id} has {len(sample_expansions)} expansion(s)")
    print(f"  First expansion type: {type(sample_expansions[0])}")
    print(f"  First expansion value: {sample_expansions[0]}")
    if isinstance(sample_expansions[0], tuple):
        print("  ✓ Expansions are tuples (correct format)")
    else:
        print("  ✗ Expansions are not tuples (needs fix)")
    print()

    # Test expansion functionality
    print("Testing expansion with a sample text:")
    test_text = "Hello, this is a test of the stochastok tokenizer."
    print(f"  Input: '{test_text}'")

    try:
        # Encode without expansion
        token_ids_orig = tokenizer.encode(test_text)
        print(f"  Original tokens: {len(token_ids_orig)}")

        # Expand with 10% expansion rate
        token_ids_expanded = processor.expand(token_ids_orig.copy(), expand_prop=0.1, disable_tqdm=True)
        print(f"  Expanded tokens: {len(token_ids_expanded)}")
        print(f"  Expansion: {len(token_ids_expanded) - len(token_ids_orig)} tokens added")

        # Decode both
        decoded_orig = tokenizer.decode(token_ids_orig)
        decoded_expanded = tokenizer.decode(token_ids_expanded)
        print(f"  Original decoded: '{decoded_orig}'")
        print(f"  Expanded decoded: '{decoded_expanded}'")

        if decoded_orig == decoded_expanded:
            print("  ✓ Decoding matches (expansion is reversible)")
        else:
            print("  ✗ Decoding mismatch!")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
    print()

    # Test multiple expansion rates
    print("Testing different expansion rates:")
    base_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    for rate in [0.0, 0.1, 0.3, 0.5]:
        try:
            expanded = processor.expand(base_tokens.copy(), expand_prop=rate, disable_tqdm=True)
            growth = len(expanded) - len(base_tokens)
            print(f"  Rate {rate:.1f}: {len(base_tokens)} → {len(expanded)} tokens (+{growth})")
        except Exception as e:
            print(f"  Rate {rate:.1f}: ✗ Error: {e}")
    print()

    print("=" * 80)
    print("✓ Stochastok tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_stochastok()
