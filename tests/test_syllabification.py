import os
import random

import pandas as pd
from pacute.syllabification import create_syllabification_dataset

from setup_paths import setup_project_paths

setup_project_paths()

random.seed(100)
num_samples = 20
os.makedirs("tasks", exist_ok=True)
syllables = pd.read_json("data/syllables.jsonl", lines=True)

print("\nGenerating MCQ syllabification dataset...")
mcq_syllabification = create_syllabification_dataset(
    syllables, num_samples=num_samples, mode="mcq", random_seed=1859, freq_weight=0.75
)
mcq_syllabification.to_json("tasks/mcq_syllabification.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_syllabification)} samples")
print(f"Subcategories: {mcq_syllabification['subcategory'].value_counts().to_dict()}")

print("\nGenerating GEN syllabification dataset...")
gen_syllabification = create_syllabification_dataset(
    syllables, num_samples=num_samples, mode="gen", random_seed=1859, freq_weight=0.75
)
gen_syllabification.to_json("tasks/gen_syllabification.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_syllabification)} samples")
print(f"Subcategories: {gen_syllabification['subcategory'].value_counts().to_dict()}")

print("\nAll syllabification datasets generated successfully!")
