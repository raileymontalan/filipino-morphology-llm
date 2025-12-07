from datasets import load_dataset
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load the SEA-PILE-v2 dataset
dataset = load_dataset("aisingapore/SEA-PILE-v2", data_dir="tl", split="train")

# Create output directory if it doesn't exist
output_dir = "data/corpora"
os.makedirs(output_dir, exist_ok=True)

# Convert to JSONL format (required by NeMo)
with open(os.path.join(output_dir, "seapile-v2.jsonl"), "w") as f:
    for item in dataset:
        json.dump({"text": item["text"]}, f)
        f.write("\n")