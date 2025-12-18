import os
import random

import pandas as pd
from pacute.composition import create_composition_dataset

from setup_paths import setup_project_paths

setup_project_paths()

random.seed(100)
num_samples = 20
os.makedirs("tasks", exist_ok=True)
syllables = pd.read_json("data/syllables.jsonl", lines=True)

print("Generating MCQ composition dataset...")
mcq_composition = create_composition_dataset(
    syllables, num_samples=num_samples, mode="mcq", random_seed=1859, freq_weight=0.75
)
mcq_composition.to_json("tasks/mcq_composition.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_composition)} samples")
print(f"Subcategories: {mcq_composition['subcategory'].value_counts().to_dict()}")

print("\nGenerating GEN composition dataset...")
gen_composition = create_composition_dataset(
    syllables, num_samples=num_samples, mode="gen", random_seed=1859, freq_weight=0.75
)
gen_composition.to_json("tasks/gen_composition.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_composition)} samples")
print(f"Subcategories: {gen_composition['subcategory'].value_counts().to_dict()}")

print("\nAll composition datasets generated successfully!")
