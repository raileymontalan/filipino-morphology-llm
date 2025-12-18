#!/usr/bin/env python3
r"""
Preprocess JSONL data to Megatron binary format (.bin + .idx) for NeMo training.

NeMo's PreTrainingDataModule expects Megatron binary format, not raw JSONL.

IMPORTANT: This script must be run INSIDE the NeMo container, not on the host!
The preprocessing tools are only available in the container environment.

Supports three tokenization modes:
1. vanilla: Standard tokenization (default)
2. stochastok: Stochastic token expansion (~10% expansion)
3. patok: Morphology-aware expand-contract with Filipino affix awareness

Usage (inside container):
    # Vanilla tokenization (baseline)
    python /workspace/scripts/preprocess_data.py \\
        --input /workspace/data/chunks/chunk_0001.jsonl \\
        --output-prefix /workspace/data/processed/chunk_0001 \\
        --tokenizer-model google/gemma-3-1b-pt

    # Stochastok tokenization
    python /workspace/scripts/preprocess_data.py \\
        --input /workspace/data/chunks/chunk_0001.jsonl \\
        --output-prefix /workspace/data/processed/chunk_0001_stochastok \\
        --tokenizer-model google/gemma-3-1b-pt \\
        --tokenization-mode stochastok \\
        --expand-prop 0.1

    # Patok tokenization (morphology-aware)
    python /workspace/scripts/preprocess_data.py \\
        --input /workspace/data/chunks/chunk_0001.jsonl \\
        --output-prefix /workspace/data/processed/chunk_0001_patok \\
        --tokenizer-model google/gemma-3-1b-pt \\
        --tokenization-mode patok \\
        --contract-prop 0.9 \\
        --expand-prop 0.1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL to Megatron binary format with optional stochastok expansion"
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
        help="Output prefix (will create <prefix>.bin and .idx)",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="/workspace/data/tokenizer/gemma2_tokenizer.model",
        help="Path to SentencePiece tokenizer.model file",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace tokenizer name for stochastok/patok modes",
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
    parser.add_argument(
        "--tokenization-mode",
        type=str,
        choices=["vanilla", "stochastok", "patok"],
        default="vanilla",
        help="Tokenization mode: 'vanilla' (default), 'stochastok', or 'patok' (morphology-aware)",
    )
    parser.add_argument(
        "--expand-prop",
        type=float,
        default=0.1,
        help="Proportion of tokens to expand (stochastok: 0.1, patok: 0.1)",
    )
    parser.add_argument(
        "--contract-prop",
        type=float,
        default=0.9,
        help="Proportion of tokens to contract (patok mode only, default: 0.9)",
    )
    parser.add_argument(
        "--affix-awareness",
        type=float,
        default=0.95,
        help="Probability of affix-aware processing (patok mode only, default: 0.95)",
    )
    parser.add_argument(
        "--prefix-file",
        type=str,
        default="/workspace/data/affixes_filipino/prefix.txt",
        help="Path to prefix file (patok mode only)",
    )
    parser.add_argument(
        "--infix-file",
        type=str,
        default="/workspace/data/affixes_filipino/infix.txt",
        help="Path to infix file (patok mode only)",
    )
    parser.add_argument(
        "--suffix-file",
        type=str,
        default="/workspace/data/affixes_filipino/suffix.txt",
        help="Path to suffix file (patok mode only)",
    )
    parser.add_argument(
        "--expansions-file",
        type=str,
        default=None,
        help="Path to pre-built expansions JSON file (patok mode only, speeds up initialization)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible tokenization (default: 42)",
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
    print(f"Input:              {args.input}")
    print(f"Output prefix:      {args.output_prefix}")
    print(f"Tokenizer:          {args.tokenizer_model}")
    print(f"Text key:           {args.text_key}")
    print(f"Workers:            {args.workers}")
    print(f"Tokenization mode:  {args.tokenization_mode}")
    if args.tokenization_mode == "stochastok":
        print(f"Expand proportion:  {args.expand_prop}")
        print(f"Random seed:        {args.seed}")
    elif args.tokenization_mode == "patok":
        print(f"Contract proportion: {args.contract_prop}")
        print(f"Expand proportion:   {args.expand_prop}")
        print(f"Affix awareness:     {args.affix_awareness}")
        print(f"Prefix file:         {args.prefix_file}")
        print(f"Infix file:          {args.infix_file}")
        print(f"Suffix file:         {args.suffix_file}")
        print(f"Expansions file:     {args.expansions_file or '(will build from scratch)'}")
        print(f"Random seed:         {args.seed}")
    print("=" * 80)
    print()

    # Choose preprocessing mode
    if args.tokenization_mode == "vanilla":
        return preprocess_vanilla(args, input_path)
    elif args.tokenization_mode == "stochastok":
        return preprocess_stochastok(args, input_path)
    elif args.tokenization_mode == "patok":
        return preprocess_patok(args, input_path)
    else:
        print(f"✗ Error: Unknown tokenization mode: {args.tokenization_mode}")
        return 1


def preprocess_vanilla(args, input_path):
    """Standard Megatron preprocessing without token expansion."""
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
    # Use SentencePieceTokenizer with local tokenizer.model file
    cmd = [
        "python",
        preprocess_script,
        "--input",
        str(input_path),
        "--output-prefix",
        args.output_prefix,
        "--tokenizer-type",
        "SentencePieceTokenizer",
        "--tokenizer-model",
        args.tokenizer_model,
        "--json-keys",
        args.text_key,
        "--workers",
        str(args.workers),
        "--append-eod",  # Add end-of-document token
    ]

    print("Running preprocessing command:")
    print(" ".join(cmd))
    print()

    # Run the command
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        # Verify output files were created (Megatron adds _text_document suffix)
        bin_file_megatron = Path(f"{args.output_prefix}_text_document.bin")
        idx_file_megatron = Path(f"{args.output_prefix}_text_document.idx")

        if bin_file_megatron.exists() and idx_file_megatron.exists():
            # Rename files to remove _text_document suffix for simpler usage
            bin_file = Path(f"{args.output_prefix}.bin")
            idx_file = Path(f"{args.output_prefix}.idx")

            print()
            print("=" * 80)
            print("✓ Preprocessing Complete!")
            print("=" * 80)
            print("Renaming output files (removing _text_document suffix)...")
            bin_file_megatron.rename(bin_file)
            idx_file_megatron.rename(idx_file)

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


def preprocess_stochastok(args, input_path):
    """
    Preprocess with stochastok token expansion.

    This involves:
    1. Tokenizing text with standard tokenizer
    2. Applying stochastok expansion to token sequences
    3. Writing expanded sequences to Megatron binary format
    """
    import json
    import random

    import numpy as np
    from tqdm import tqdm

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        print(f"✓ Loaded tokenizer: {args.hf_tokenizer}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return 1

    print()
    print("Initializing StochastokProcessor...")

    # Import StochastokProcessor
    try:
        import subprocess
        import sys

        # Install dependencies if needed
        missing_deps = []
        try:
            pass
        except ImportError:
            missing_deps.append("pyahocorasick")
        try:
            pass
        except ImportError:
            missing_deps.append("tiktoken")
        if missing_deps:
            print(f"Installing missing dependencies: {', '.join(missing_deps)}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_deps + ["-q"])

        # Add src to path for proper imports
        # Try workspace path first (container), fall back to relative path
        workspace_src = Path("/workspace/src")
        local_src = Path(__file__).parent.parent.parent.parent / "src"
        src_path = workspace_src if workspace_src.exists() else local_src
        sys.path.insert(0, str(src_path))
        from tokenization.stochastok_processor import StochastokProcessor

        processor = StochastokProcessor(tokenizer, expand_prop=args.expand_prop)
        print(f"✓ StochastokProcessor initialized (using {src_path})")
        print(f"  Number of expandable tokens: {len(processor.expansions)}")
    except Exception as e:
        print(f"✗ Error initializing StochastokProcessor: {e}")
        print()
        print("Make sure StochastokProcessor is available in src/tokenization/")
        return 1

    print()
    print("Creating intermediate JSONL with expanded tokens...")

    # Create temporary file for expanded tokens
    temp_dir = Path(args.output_prefix).parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_jsonl = temp_dir / f"{Path(args.output_prefix).name}_stochastok_temp.jsonl"

    try:
        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(temp_jsonl, "w", encoding="utf-8") as outfile,
        ):
            total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

            total_original_tokens = 0
            total_expanded_tokens = 0

            for line in tqdm(infile, total=total_lines, desc="Processing documents"):
                try:
                    doc = json.loads(line)
                    text = doc.get(args.text_key, "")

                    if not text:
                        continue

                    # Tokenize
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    original_length = len(token_ids)

                    # Apply stochastok expansion
                    expanded_ids = processor.expand(token_ids, expand_prop=args.expand_prop, disable_tqdm=True)

                    total_original_tokens += original_length
                    total_expanded_tokens += len(expanded_ids)

                    # Decode back to text for Megatron preprocessing
                    # Note: This may introduce some artifacts, but Megatron will re-tokenize
                    expanded_text = tokenizer.decode(expanded_ids, skip_special_tokens=True)

                    # Write to temp file
                    output_doc = {args.text_key: expanded_text}
                    outfile.write(json.dumps(output_doc, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue

        print()
        print(f"✓ Created temporary expanded JSONL: {temp_jsonl}")
        print(f"  Original tokens:  {total_original_tokens:,}")
        print(f"  Expanded tokens:  {total_expanded_tokens:,}")
        print(f"  Expansion ratio:  {total_expanded_tokens / total_original_tokens:.2%}")

    except Exception as e:
        print(f"✗ Error during token expansion: {e}")
        return 1

    print()
    print("Running Megatron preprocessing on expanded data...")

    # Now run standard Megatron preprocessing on the expanded data
    preprocess_script = "/opt/megatron-lm/tools/preprocess_data.py"

    if not Path(preprocess_script).exists():
        print(f"✗ Error: Megatron preprocessing script not found: {preprocess_script}")
        return 1

    cmd = [
        "python",
        preprocess_script,
        "--input",
        str(temp_jsonl),
        "--output-prefix",
        args.output_prefix,
        "--tokenizer-type",
        "SentencePieceTokenizer",
        "--tokenizer-model",
        args.tokenizer_model,
        "--json-keys",
        args.text_key,
        "--workers",
        str(args.workers),
        "--append-eod",
    ]

    print("Running preprocessing command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )

        # Verify output files (Megatron adds _text_document suffix)
        bin_file_megatron = Path(f"{args.output_prefix}_text_document.bin")
        idx_file_megatron = Path(f"{args.output_prefix}_text_document.idx")

        if bin_file_megatron.exists() and idx_file_megatron.exists():
            # Rename files to remove _text_document suffix for simpler usage
            bin_file = Path(f"{args.output_prefix}.bin")
            idx_file = Path(f"{args.output_prefix}.idx")

            print()
            print("=" * 80)
            print("✓ Stochastok Preprocessing Complete!")
            print("=" * 80)
            print("Renaming output files (removing _text_document suffix)...")
            bin_file_megatron.rename(bin_file)
            idx_file_megatron.rename(idx_file)

            # Clean up temporary file
            temp_jsonl.unlink()
            print(f"✓ Cleaned up temporary file: {temp_jsonl}")

            print(f"Binary file: {bin_file} ({bin_file.stat().st_size / 1e9:.2f} GB)")
            print(f"Index file:  {idx_file} ({idx_file.stat().st_size / 1e6:.2f} MB)")
            print()
            print("Expansion statistics:")
            print(f"  Original tokens:  {total_original_tokens:,}")
            print(f"  Expanded tokens:  {total_expanded_tokens:,}")
            print(f"  Expansion ratio:  {total_expanded_tokens / total_original_tokens:.2%}")
            print()
            print("To use in training, specify:")
            print(f"  --data-path {args.output_prefix}")
            print("=" * 80)
            return 0
        else:
            print("✗ Error: Output files not created")
            # Clean up temporary file even on failure
            if temp_jsonl.exists():
                temp_jsonl.unlink()
            return 1

    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed with exit code {e.returncode}")
        # Clean up temporary file
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        return e.returncode


def preprocess_patok(args, input_path):
    """
    Preprocess with Patok morphology-aware tokenization.

    This involves:
    1. Tokenizing text with standard tokenizer
    2. Applying Patok contract-expand with Filipino affix awareness
    3. Writing processed sequences to Megatron binary format
    """
    import json
    import random

    import numpy as np
    from tqdm import tqdm

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        print(f"✓ Loaded tokenizer: {args.hf_tokenizer}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return 1

    print()
    print("Initializing MorphologyAwarePatokProcessor...")

    # Import MorphologyAwarePatokProcessor
    try:
        import sys

        # Add src to path for proper imports
        # Try workspace path first (container), fall back to relative path
        workspace_src = Path("/workspace/src")
        local_src = Path(__file__).parent.parent.parent.parent / "src"
        src_path = workspace_src if workspace_src.exists() else local_src
        sys.path.insert(0, str(src_path))
        from tokenization.patok_morphology import MorphologyAwarePatokProcessor

        # Verify affix files exist
        for affix_file in [args.prefix_file, args.infix_file, args.suffix_file]:
            if not Path(affix_file).exists():
                print(f"✗ Error: Affix file not found: {affix_file}")
                return 1

        processor = MorphologyAwarePatokProcessor(
            tokenizer,
            prefix_file=args.prefix_file,
            infix_file=args.infix_file,
            suffix_file=args.suffix_file,
            contract_prop=args.contract_prop,
            expand_prop=args.expand_prop,
            affix_awareness=args.affix_awareness,
            expansions_file=args.expansions_file,
        )
        print("✓ MorphologyAwarePatokProcessor initialized")
        print(f"  Number of affix versions: {len(processor.affixes)}")
        print(f"  Number of affix token IDs: {len(processor.affix_ids)}")
        print(f"  Number of expandable tokens: {len(processor.expansions)}")
    except Exception as e:
        print(f"✗ Error initializing MorphologyAwarePatokProcessor: {e}")
        print()
        print("Make sure MorphologyAwarePatokProcessor is available in /workspace/src/tokenization/")
        print("And that affix files exist at the specified paths.")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("Creating intermediate JSONL with Patok-processed tokens...")

    # Create temporary file for processed tokens
    temp_dir = Path(args.output_prefix).parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_jsonl = temp_dir / f"{Path(args.output_prefix).name}_patok_temp.jsonl"

    try:
        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(temp_jsonl, "w", encoding="utf-8") as outfile,
        ):
            total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

            total_original_tokens = 0
            total_processed_tokens = 0

            for line in tqdm(infile, total=total_lines, desc="Processing documents"):
                try:
                    doc = json.loads(line)
                    text = doc.get(args.text_key, "")

                    if not text:
                        continue

                    # Tokenize
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                    original_length = len(token_ids)

                    # Apply Patok contract-expand
                    processed_ids = processor.contract_expand(
                        token_ids,
                        contract_prop=args.contract_prop,
                        expand_prop=args.expand_prop,
                        disable_tqdm=True,
                    )

                    total_original_tokens += original_length
                    total_processed_tokens += len(processed_ids)

                    # Decode back to text for Megatron preprocessing
                    processed_text = tokenizer.decode(processed_ids, skip_special_tokens=True)

                    # Write to temp file
                    output_doc = {args.text_key: processed_text}
                    outfile.write(json.dumps(output_doc, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue

        print()
        print(f"✓ Created temporary Patok-processed JSONL: {temp_jsonl}")
        print(f"  Original tokens:   {total_original_tokens:,}")
        print(f"  Processed tokens:  {total_processed_tokens:,}")
        if total_original_tokens > 0:
            ratio = total_processed_tokens / total_original_tokens
            print(f"  Token ratio:       {ratio:.2%}")

    except Exception as e:
        print(f"✗ Error during Patok processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("Running Megatron preprocessing on Patok-processed data...")

    # Now run standard Megatron preprocessing on the processed data
    preprocess_script = "/opt/megatron-lm/tools/preprocess_data.py"

    if not Path(preprocess_script).exists():
        print(f"✗ Error: Megatron preprocessing script not found: {preprocess_script}")
        return 1

    cmd = [
        "python",
        preprocess_script,
        "--input",
        str(temp_jsonl),
        "--output-prefix",
        args.output_prefix,
        "--tokenizer-type",
        "SentencePieceTokenizer",
        "--tokenizer-model",
        args.tokenizer_model,
        "--json-keys",
        args.text_key,
        "--workers",
        str(args.workers),
        "--append-eod",
    ]

    print("Running preprocessing command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )

        # Verify output files (Megatron adds _text_document suffix)
        bin_file_megatron = Path(f"{args.output_prefix}_text_document.bin")
        idx_file_megatron = Path(f"{args.output_prefix}_text_document.idx")

        if bin_file_megatron.exists() and idx_file_megatron.exists():
            # Rename files to remove _text_document suffix
            bin_file = Path(f"{args.output_prefix}.bin")
            idx_file = Path(f"{args.output_prefix}.idx")

            print()
            print("=" * 80)
            print("✓ Patok Preprocessing Complete!")
            print("=" * 80)
            print("Renaming output files (removing _text_document suffix)...")
            bin_file_megatron.rename(bin_file)
            idx_file_megatron.rename(idx_file)

            # Clean up temporary file
            temp_jsonl.unlink()
            print(f"✓ Cleaned up temporary file: {temp_jsonl}")

            print(f"Binary file: {bin_file} ({bin_file.stat().st_size / 1e9:.2f} GB)")
            print(f"Index file:  {idx_file} ({idx_file.stat().st_size / 1e6:.2f} MB)")
            print()
            print("Patok processing statistics:")
            print(f"  Original tokens:   {total_original_tokens:,}")
            print(f"  Processed tokens:  {total_processed_tokens:,}")
            if total_original_tokens > 0:
                print(f"  Token ratio:       {total_processed_tokens / total_original_tokens:.2%}")
            print()
            print("To use in training, specify:")
            print(f"  --data-path {args.output_prefix}")
            print("=" * 80)
            return 0
        else:
            print("✗ Error: Output files not created")
            if temp_jsonl.exists():
                temp_jsonl.unlink()
            return 1

    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed with exit code {e.returncode}")
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
