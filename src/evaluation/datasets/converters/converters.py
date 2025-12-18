"""
Benchmark Data Conversion Utilities.

Provides functions to convert between different benchmark data formats:
- JSONL: Human-readable format for evaluation
- Memmap: Memory-mapped binary format for efficient training

Supports bidirectional conversion and batch processing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm


def save_as_memmap(data: np.ndarray, output_path: Union[str, Path], dtype=np.uint16) -> None:
    """
    Save numpy array as memory-mapped binary file.

    Args:
        data: Numpy array to save
        output_path: Path to output .bin file
        dtype: Data type for memmap (default: uint16 for token IDs)

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> save_as_memmap(data, "data/benchmarks/test.bin")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.memmap(str(output_path), dtype=dtype, mode="w+", shape=data.shape)
    arr[:] = data
    arr.flush()


def load_memmap(input_path: Union[str, Path], shape: Optional[tuple] = None, dtype=np.uint16) -> np.ndarray:
    """
    Load memory-mapped binary file as numpy array.

    Args:
        input_path: Path to input .bin file
        shape: Shape to reshape the array (if None, returns 1D)
        dtype: Data type for memmap (default: uint16)

    Returns:
        Memory-mapped numpy array

    Example:
        >>> data = load_memmap("data/benchmarks/test.bin")
    """
    mode = "r"  # Read-only mode
    if shape is None:
        return np.memmap(str(input_path), dtype=dtype, mode=mode)
    else:
        return np.memmap(str(input_path), dtype=dtype, mode=mode).reshape(shape)


def jsonl_to_memmap(
    jsonl_path: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer,
    text_field: str = "text",
    max_length: Optional[int] = None,
    pad_token_id: int = 0,
    save_metadata: bool = True,
) -> Dict[str, Path]:
    """
    Convert JSONL benchmark file to memmap format.

    Args:
        jsonl_path: Path to input JSONL file
        output_dir: Directory to save memmap files
        tokenizer: Tokenizer instance with encode() method
        text_field: Name of field containing text to tokenize
        max_length: Maximum sequence length (if None, uses longest sequence)
        pad_token_id: Token ID for padding
        save_metadata: Whether to save metadata JSON file

    Returns:
        Dictionary mapping output type to file paths

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> paths = jsonl_to_memmap(
        ...     "data/benchmarks/mcq_affixation.jsonl",
        ...     "data/memmaps/affixation",
        ...     tokenizer
        ... )
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {jsonl_path.name} to memmap format...")

    # Load and tokenize all data
    tokenized_sequences = []
    original_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            data = json.loads(line)
            original_data.append(data)

            # Extract text to tokenize
            if text_field in data:
                text = data[text_field]
            else:
                # Try to construct text from common fields
                text = data.get("question", "") or data.get("prompt", "") or str(data)

            # Tokenize
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text)
            elif hasattr(tokenizer, "__call__"):
                tokens = tokenizer(text, return_tensors=None)["input_ids"]
            else:
                raise ValueError("Tokenizer must have encode() or __call__() method")

            tokenized_sequences.append(tokens)

    # Determine padding length
    if max_length is None:
        max_length = max(len(seq) for seq in tokenized_sequences)

    # Pad sequences
    padded_sequences = []
    for seq in tokenized_sequences:
        if len(seq) > max_length:
            padded = seq[:max_length]
        else:
            padded = seq + [pad_token_id] * (max_length - len(seq))
        padded_sequences.append(padded)

    # Convert to numpy array
    data_array = np.array(padded_sequences, dtype=np.uint16)

    # Save as memmap
    memmap_path = output_dir / f"{jsonl_path.stem}.bin"
    save_as_memmap(data_array, memmap_path)

    output_paths = {"data": memmap_path}

    # Save metadata
    if save_metadata:
        metadata = {
            "source_file": str(jsonl_path),
            "num_sequences": len(padded_sequences),
            "max_length": max_length,
            "shape": list(data_array.shape),
            "dtype": str(data_array.dtype),
            "pad_token_id": pad_token_id,
        }
        metadata_path = output_dir / f"{jsonl_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        output_paths["metadata"] = metadata_path

    print(f"✓ Saved memmap: {memmap_path}")
    print(f"  Shape: {data_array.shape}")
    print(f"  Total sequences: {len(padded_sequences)}")

    return output_paths


def convert_benchmark_directory_to_memmaps(
    benchmark_dir: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer,
    pattern: str = "*.jsonl",
    **kwargs,
) -> Dict[str, Dict[str, Path]]:
    """
    Convert all JSONL files in a directory to memmap format.

    Args:
        benchmark_dir: Directory containing JSONL files
        output_dir: Directory to save memmap files
        tokenizer: Tokenizer instance
        pattern: Glob pattern for JSONL files (default: "*.jsonl")
        **kwargs: Additional arguments for jsonl_to_memmap

    Returns:
        Dictionary mapping source filename to output paths

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> results = convert_benchmark_directory_to_memmaps(
        ...     "data/benchmarks",
        ...     "data/memmaps",
        ...     tokenizer
        ... )
    """
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)

    jsonl_files = list(benchmark_dir.glob(pattern))

    if not jsonl_files:
        print(f"No files matching '{pattern}' found in {benchmark_dir}")
        return {}

    print(f"Found {len(jsonl_files)} JSONL files to convert")

    results = {}
    for jsonl_file in jsonl_files:
        benchmark_output_dir = output_dir / jsonl_file.stem
        try:
            paths = jsonl_to_memmap(jsonl_file, benchmark_output_dir, tokenizer, **kwargs)
            results[jsonl_file.name] = paths
        except Exception as e:
            print(f"✗ Failed to convert {jsonl_file.name}: {e}")

    print(f"\n✓ Successfully converted {len(results)}/{len(jsonl_files)} files")
    return results


def save_dataset_as_jsonl(data: List[Dict], output_path: Union[str, Path], ensure_ascii: bool = False) -> None:
    """
    Save list of dictionaries as JSONL file.

    Args:
        data: List of dictionaries to save
        output_path: Path to output JSONL file
        ensure_ascii: Whether to escape non-ASCII characters

    Example:
        >>> data = [
        ...     {"question": "What is 2+2?", "answer": "4"},
        ...     {"question": "What is 3+3?", "answer": "6"}
        ... ]
        >>> save_dataset_as_jsonl(data, "data/benchmarks/math.jsonl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=ensure_ascii)
            f.write(json_line + "\n")


def load_jsonl(input_path: Union[str, Path]) -> List[Dict]:
    """
    Load JSONL file as list of dictionaries.

    Args:
        input_path: Path to JSONL file

    Returns:
        List of dictionaries

    Example:
        >>> data = load_jsonl("data/benchmarks/math.jsonl")
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Convenience function for common use case
def create_memmap_from_token_ids(
    token_ids: List[List[int]],
    output_path: Union[str, Path],
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Create memmap from list of token ID sequences.

    Args:
        token_ids: List of token ID sequences
        output_path: Path to output .bin file
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length (if None, uses longest)

    Returns:
        Padded numpy array that was saved

    Example:
        >>> token_ids = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        >>> arr = create_memmap_from_token_ids(
        ...     token_ids,
        ...     "data/memmaps/tokens.bin",
        ...     pad_token_id=0
        ... )
    """
    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in token_ids)

    # Pad sequences
    padded = []
    for seq in token_ids:
        if len(seq) > max_length:
            padded_seq = seq[:max_length]
        else:
            padded_seq = seq + [pad_token_id] * (max_length - len(seq))
        padded.append(padded_seq)

    # Convert to array and save
    arr = np.array(padded, dtype=np.uint16)
    save_as_memmap(arr, output_path)

    return arr
