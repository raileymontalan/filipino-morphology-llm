from datasets import load_dataset
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load the SEA-PILE-v2 dataset
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

# Create output directory if it doesn't exist
output_dir = "data/corpora"
os.makedirs(output_dir, exist_ok=True)

# Convert to JSONL format (required by NeMo)
with open(os.path.join(output_dir, "fineweb.jsonl"), "w") as f:
    for item in dataset:
        json.dump({"text": item["text"]}, f)
        f.write("\n")