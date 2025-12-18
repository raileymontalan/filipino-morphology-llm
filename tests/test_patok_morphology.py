#!/usr/bin/env python3
"""
Test script for morphology-aware Patok tokenizer.

Tests the implementation from colleague using Filipino example sentences.
"""

from transformers import AutoTokenizer

from setup_paths import setup_project_paths
from tokenization.patok_morphology import MorphologyAwarePatokProcessor

setup_project_paths()


def print_tokens(label, token_ids, tokenizer):
    """Pretty print token IDs and their string representations."""

    try:
        tokens = [tokenizer.decode_single_token_bytes(tid).decode("utf-8", "replace") for tid in token_ids]
    except Exception:
        tokens = [tokenizer.decode(tid) for tid in token_ids]
    print(f"\n{label}:")
    print(f"  Token IDs: {token_ids}")
    print(f"  Tokens: {tokens}")
    print(f"  Length: {len(token_ids)} tokens")


def main():
    print("=" * 80)
    print("Testing Morphology-Aware Patok Tokenizer")
    print("=" * 80)

    # Initialize tokenizer
    model = "openai/gpt-oss-20b"
    print(f"\n1. Initializing {model} tokenizer...")
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Test sentences from colleague's notebook
    test_sentences = [
        "Nagkukumahog na pinadalhan ng magagandang parlorista ang poging tagalungsod ng pagkarami-raming pagkain.",
        "Lupang hinirang, duyan ka ng magiting, sa manlulupig, di ka pasisiil",
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: {sentence}")
        print("=" * 80)

        # Baseline tokenization
        baseline_ids = tokenizer.encode(sentence)
        print_tokens(f"Baseline ({model})", baseline_ids, tokenizer)

        # Initialize Patok processor
        print("\n2. Initializing Morphology-Aware Patok Processor...")
        patok = MorphologyAwarePatokProcessor(
            tokenizer,
            prefix_file="data/affixes_filipino/prefix.txt",
            infix_file="data/affixes_filipino/infix.txt",
            suffix_file="data/affixes_filipino/suffix.txt",
        )

        # Process with Patok
        print("\n3. Applying Patok processing...")
        patok_ids = patok.contract_expand(baseline_ids.copy(), disable_tqdm=False)
        print_tokens("Patok (Morphology-Aware)", patok_ids, tokenizer)

        # Verify roundtrip
        baseline_text = tokenizer.decode(baseline_ids)
        patok_text = tokenizer.decode(patok_ids)

        print("\n4. Verification:")
        print(f"  Original: {sentence}")
        print(f"  Baseline decoded: {baseline_text}")
        print(f"  Patok decoded: {patok_text}")
        print(f"  Roundtrip matches: {patok_text == baseline_text}")

        # Analyze changes
        print("\n5. Analysis:")
        print(f"  Baseline tokens: {len(baseline_ids)}")
        print(f"  Patok tokens: {len(patok_ids)}")
        print(f"  Change: {len(patok_ids) - len(baseline_ids):+d} tokens")

        # Check for affixes in output
        affix_tokens = [tid for tid in patok_ids if tid in patok.affix_ids]
        if affix_tokens:
            try:
                affix_strs = [
                    tokenizer.decode_single_token_bytes(tid).decode("utf-8", "replace") for tid in affix_tokens
                ]
            except Exception:
                affix_strs = [tokenizer.decode(tid) for tid in affix_tokens]
            print(f"  Preserved affixes: {affix_strs}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
