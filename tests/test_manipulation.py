import os
import random

import pandas as pd

from pacute.manipulation import create_manipulation_dataset
from setup_paths import setup_project_paths

setup_project_paths()

random.seed(100)
num_samples = 20
os.makedirs("tasks", exist_ok=True)
syllables = pd.read_json("data/syllables.jsonl", lines=True)

print("\nGenerating MCQ manipulation dataset...")
mcq_manipulation = create_manipulation_dataset(
    syllables, num_samples=num_samples, mode="mcq", random_seed=1859, freq_weight=0.75
)
mcq_manipulation.to_json("tasks/mcq_manipulation.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_manipulation)} samples")
print(f"Subcategories: {mcq_manipulation['subcategory'].value_counts().to_dict()}")

print("\nGenerating GEN manipulation dataset...")
gen_manipulation = create_manipulation_dataset(
    syllables, num_samples=num_samples, mode="gen", random_seed=1859, freq_weight=0.75
)
gen_manipulation.to_json("tasks/gen_manipulation.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_manipulation)} samples")
print(f"Subcategories: {gen_manipulation['subcategory'].value_counts().to_dict()}")

print("\nAll manipulation datasets generated successfully!")
