"""
Generate PACUTE Benchmarks.

Creates evaluation datasets for Filipino morphological understanding:
- Affixation: Understanding prefixes, suffixes, infixes
- Composition: Combining morphemes to form words
- Manipulation: String operations on words
- Syllabification: Breaking words into syllables

Each category generates both MCQ and generative format tasks.
Output: data/benchmarks/*.jsonl files
"""

import sys
from pathlib import Path

import pandas as pd

from evaluation.datasets.generators.affixation import create_affixation_dataset
from evaluation.datasets.generators.composition import create_composition_dataset
from evaluation.datasets.generators.manipulation import create_manipulation_dataset
from evaluation.datasets.generators.syllabification import (
    create_syllabification_dataset,
)

# Add src to path
# Go up 5 levels: scripts -> datasets -> evaluation -> src -> filipino-morphology-llm
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def generate_affixation_benchmarks(output_dir: str, random_seed: int = 1859):
    """Generate affixation benchmarks (MCQ and generative)."""
    print("\n" + "=" * 60)
    print("GENERATING AFFIXATION BENCHMARKS")
    print("=" * 60)

    # Load inflection data
    inflections_path = project_root / "data" / "corpora" / "pacute_data" / "inflections.xlsx"
    if not inflections_path.exists():
        print(f"ERROR: Inflections file not found at {inflections_path}")
        return False

    inflections = pd.read_excel(inflections_path, sheet_name="data")
    print(f"Loaded {len(inflections)} inflection pairs")

    # Generate MCQ
    print("\nGenerating MCQ affixation dataset...")
    mcq_dataset = create_affixation_dataset(inflections, mode="mcq", random_seed=random_seed)
    mcq_path = Path(output_dir) / "affixation_mcq.jsonl"
    mcq_dataset.to_json(mcq_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(mcq_dataset)} MCQ samples → {mcq_path}")

    # Generate GEN
    print("\nGenerating GEN affixation dataset...")
    gen_dataset = create_affixation_dataset(inflections, mode="gen", random_seed=random_seed)
    gen_path = Path(output_dir) / "affixation_gen.jsonl"
    gen_dataset.to_json(gen_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(gen_dataset)} GEN samples → {gen_path}")

    return True


def generate_composition_benchmarks(output_dir: str, num_samples: int = 100, random_seed: int = 1859):
    """Generate composition benchmarks (MCQ and generative).

    Args:
        num_samples: Number of samples PER SUBCATEGORY (not total)
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPOSITION BENCHMARKS")
    print("=" * 60)

    # Load syllables data
    syllables_path = project_root / "data" / "corpora" / "pacute_data" / "syllables.jsonl"
    if not syllables_path.exists():
        print(f"ERROR: Syllables file not found at {syllables_path}")
        return False

    syllables = pd.read_json(syllables_path, lines=True)
    print(f"Loaded {len(syllables)} syllabified words")

    # Generate MCQ
    print(f"\nGenerating MCQ composition dataset ({num_samples} samples per subcategory)...")
    mcq_dataset = create_composition_dataset(
        syllables,
        num_samples=num_samples,
        mode="mcq",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    mcq_path = Path(output_dir) / "composition_mcq.jsonl"
    mcq_dataset.to_json(mcq_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(mcq_dataset)} MCQ samples → {mcq_path}")
    print(f"  Subcategories: {mcq_dataset['subcategory'].value_counts().to_dict()}")

    # Generate GEN
    print(f"\nGenerating GEN composition dataset ({num_samples} samples per subcategory)...")
    gen_dataset = create_composition_dataset(
        syllables,
        num_samples=num_samples,
        mode="gen",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    gen_path = Path(output_dir) / "composition_gen.jsonl"
    gen_dataset.to_json(gen_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(gen_dataset)} GEN samples → {gen_path}")
    print(f"  Subcategories: {gen_dataset['subcategory'].value_counts().to_dict()}")

    return True


def generate_manipulation_benchmarks(output_dir: str, num_samples: int = 100, random_seed: int = 1859):
    """Generate manipulation benchmarks (MCQ and generative).

    Args:
        num_samples: Number of samples PER SUBCATEGORY (not total)
    """
    print("\n" + "=" * 60)
    print("GENERATING MANIPULATION BENCHMARKS")
    print("=" * 60)

    # Load syllables data
    syllables_path = project_root / "data" / "corpora" / "pacute_data" / "syllables.jsonl"
    if not syllables_path.exists():
        print(f"ERROR: Syllables file not found at {syllables_path}")
        return False

    syllables = pd.read_json(syllables_path, lines=True)
    print(f"Loaded {len(syllables)} syllabified words")

    # Generate MCQ
    print(f"\nGenerating MCQ manipulation dataset ({num_samples} samples)...")
    mcq_dataset = create_manipulation_dataset(
        syllables,
        num_samples=num_samples,
        mode="mcq",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    mcq_path = Path(output_dir) / "manipulation_mcq.jsonl"
    mcq_dataset.to_json(mcq_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(mcq_dataset)} MCQ samples → {mcq_path}")
    print(f"  Subcategories: {mcq_dataset['subcategory'].value_counts().to_dict()}")

    # Generate GEN
    print(f"\nGenerating GEN manipulation dataset ({num_samples} samples)...")
    gen_dataset = create_manipulation_dataset(
        syllables,
        num_samples=num_samples,
        mode="gen",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    gen_path = Path(output_dir) / "manipulation_gen.jsonl"
    gen_dataset.to_json(gen_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(gen_dataset)} GEN samples → {gen_path}")
    print(f"  Subcategories: {gen_dataset['subcategory'].value_counts().to_dict()}")

    return True


def generate_syllabification_benchmarks(output_dir: str, num_samples: int = 100, random_seed: int = 1859):
    """Generate syllabification benchmarks (MCQ and generative).

    Args:
        num_samples: Number of samples PER SUBCATEGORY (not total)
    """
    print("\n" + "=" * 60)
    print("GENERATING SYLLABIFICATION BENCHMARKS")
    print("=" * 60)

    # Load syllables data
    syllables_path = project_root / "data" / "corpora" / "pacute_data" / "syllables.jsonl"
    if not syllables_path.exists():
        print(f"ERROR: Syllables file not found at {syllables_path}")
        return False

    syllables = pd.read_json(syllables_path, lines=True)
    print(f"Loaded {len(syllables)} syllabified words")

    # Generate MCQ
    print(f"\nGenerating MCQ syllabification dataset ({num_samples} samples)...")
    mcq_dataset = create_syllabification_dataset(
        syllables,
        num_samples=num_samples,
        mode="mcq",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    mcq_path = Path(output_dir) / "syllabification_mcq.jsonl"
    mcq_dataset.to_json(mcq_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(mcq_dataset)} MCQ samples → {mcq_path}")
    print(f"  Subcategories: {mcq_dataset['subcategory'].value_counts().to_dict()}")

    # Generate GEN
    print(f"\nGenerating GEN syllabification dataset ({num_samples} samples)...")
    gen_dataset = create_syllabification_dataset(
        syllables,
        num_samples=num_samples,
        mode="gen",
        random_seed=random_seed,
        freq_weight=0.75,
    )
    gen_path = Path(output_dir) / "syllabification_gen.jsonl"
    gen_dataset.to_json(gen_path, lines=True, orient="records", force_ascii=False)
    print(f"✓ Generated {len(gen_dataset)} GEN samples → {gen_path}")
    print(f"  Subcategories: {gen_dataset['subcategory'].value_counts().to_dict()}")

    return True


def main():
    """Generate all PACUTE benchmarks."""
    # Setup output directory
    output_dir = project_root / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PACUTE BENCHMARK GENERATION")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Generate all benchmarks
    results = {
        "Affixation": generate_affixation_benchmarks(str(output_dir)),
        "Composition": generate_composition_benchmarks(str(output_dir), num_samples=100),
        "Manipulation": generate_manipulation_benchmarks(str(output_dir), num_samples=100),
        "Syllabification": generate_syllabification_benchmarks(str(output_dir), num_samples=100),
    }

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for benchmark, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{benchmark:20s}: {status}")

    all_success = all(results.values())
    if all_success:
        print("\n✓ All PACUTE benchmarks generated successfully!")
        print(f"✓ Output location: {output_dir}")
    else:
        print("\n✗ Some benchmarks failed to generate. Check errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
