import json
import os
import random


def load_langgame(format="mcq", **kwargs):
    """
    Load and process the LangGame benchmark from JSONL file.

    Args:
        format: "mcq" or "gen" (default: "mcq")

    Yields:
        - prefix: The question
        - ground_truth: The correct answer
        - false_options: List of incorrect options (empty for gen format)
    """
    current_file_path = os.path.abspath(__file__)
    dir_of_file = os.path.dirname(current_file_path)
    jsonl_path = os.path.join(dir_of_file, f"../../../data/benchmarks/langgame_{format}.jsonl")

    # Load all samples from JSONL
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"LangGame ({format.upper()}): Loaded {len(samples)} examples.")

    # Shuffle indices
    index = list(range(len(samples)))
    random.shuffle(index)

    for i in index:
        sample = samples[i]
        prefix = sample["question"]
        sample_id = sample.get("id", f"langgame_{format}_{i:05d}")

        if format == "mcq":
            options = sample["options"]
            ground_truth = options[0]
            false_options = options[1:]
        else:  # gen
            ground_truth = sample["answer"]
            false_options = []

        yield prefix, ground_truth, false_options, sample_id
