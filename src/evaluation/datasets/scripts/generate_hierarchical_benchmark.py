"""
Generate Complete Hierarchical PACUTE Benchmark.

Creates evaluation tasks across 6 hierarchical levels:
- Level 0: Character Recognition
- Level 1: Character Manipulation
- Level 2: Morpheme Decomposition
- Level 3: Morpheme Manipulation
- Level 4: Morpheme Composition
- Level 5: Complex Morphological Reasoning

Output: MCQ and generative format files for each level
"""

import json
import sys
from pathlib import Path

import pandas as pd

from evaluation.datasets.generators.hierarchical import HierarchicalTaskGenerator

# Add src to path
# Go up 5 levels: scripts -> datasets -> evaluation -> src -> filipino-morphology-llm
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_words_for_tasks():
    """Load syllabified words for task generation."""
    words = []
    syllables_path = project_root / "data" / "corpora" / "pacute_data" / "syllables.jsonl"

    with open(syllables_path) as f:
        for line in f:
            data = json.loads(line)
            word = data.get("normalized_word", "").lower()
            syllables = data.get("normalized_syllable_list", [])

            if word and len(word) >= 3:
                words.append(
                    {
                        "word": word,
                        "syllables": syllables,
                        "num_syllables": len(syllables),
                    }
                )

    return pd.DataFrame(words)


def load_affix_annotations():
    """Load morpheme annotations and reshape to expected format."""
    annotations = []
    annotations_path = project_root / "data" / "corpora" / "affix_annotations.jsonl"

    with open(annotations_path) as f:
        for line in f:
            data = json.loads(line)

            # Reshape to expected format with separate columns for each affix type
            ann = {
                "word": data["word"],
                "root": data.get("root", ""),
                "prefix": None,
                "infix": None,
                "suffix": None,
            }

            # Populate the appropriate affix column
            affix_type = data.get("affix_type", "")
            affix = data.get("affix", "")

            if affix_type == "prefix":
                ann["prefix"] = affix
            elif affix_type == "infix":
                ann["infix"] = affix
            elif affix_type == "suffix":
                ann["suffix"] = affix

            annotations.append(ann)

    return pd.DataFrame(annotations)


def generate_all_levels(words_df, affixes_df, tasks_per_level=100):
    """Generate tasks for all 6 levels."""
    generator = HierarchicalTaskGenerator(words_df, affixes_df)

    all_tasks_mcq = []
    all_tasks_gen = []

    print("Generating hierarchical tasks...")
    print()

    # Level 0: Character Recognition
    print("Level 0: Character Recognition")
    try:
        level0_char_id = generator.generate_level0_character_identification(n=tasks_per_level // 2, format="mcq")
        print(f"  ✓ Character identification (MCQ): {len(level0_char_id)} tasks")
        all_tasks_mcq.extend(level0_char_id)

        level0_char_count = generator.generate_level0_character_counting(n=tasks_per_level // 2, format="mcq")
        print(f"  ✓ Character counting (MCQ): {len(level0_char_count)} tasks")
        all_tasks_mcq.extend(level0_char_count)

        # Generative versions
        level0_char_id_gen = generator.generate_level0_character_identification(n=tasks_per_level // 2, format="gen")
        level0_char_count_gen = generator.generate_level0_character_counting(n=tasks_per_level // 2, format="gen")
        all_tasks_gen.extend(level0_char_id_gen + level0_char_count_gen)
        print(f"  ✓ Generative: {len(level0_char_id_gen) + len(level0_char_count_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 0 generation failed: {e}")

    print()

    # Level 1: Character Manipulation
    print("Level 1: Character Manipulation")
    try:
        level1_deletion = generator.generate_level1_character_deletion(n=tasks_per_level // 4, format="mcq")
        print(f"  ✓ Character deletion (MCQ): {len(level1_deletion)} tasks")
        all_tasks_mcq.extend(level1_deletion)

        level1_insertion = generator.generate_level1_character_insertion(n=tasks_per_level // 4, format="mcq")
        print(f"  ✓ Character insertion (MCQ): {len(level1_insertion)} tasks")
        all_tasks_mcq.extend(level1_insertion)

        level1_substitution = generator.generate_level1_character_substitution(n=tasks_per_level // 4, format="mcq")
        print(f"  ✓ Character substitution (MCQ): {len(level1_substitution)} tasks")
        all_tasks_mcq.extend(level1_substitution)

        level1_permutation = generator.generate_level1_character_permutation(n=tasks_per_level // 4, format="mcq")
        print(f"  ✓ Character permutation (MCQ): {len(level1_permutation)} tasks")
        all_tasks_mcq.extend(level1_permutation)

        # Generative versions
        level1_gen = []
        level1_gen.extend(generator.generate_level1_character_deletion(n=tasks_per_level // 4, format="gen"))
        level1_gen.extend(generator.generate_level1_character_insertion(n=tasks_per_level // 4, format="gen"))
        level1_gen.extend(generator.generate_level1_character_substitution(n=tasks_per_level // 4, format="gen"))
        level1_gen.extend(generator.generate_level1_character_permutation(n=tasks_per_level // 4, format="gen"))
        all_tasks_gen.extend(level1_gen)
        print(f"  ✓ Generative: {len(level1_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 1 generation failed: {e}")

    print()

    # Level 2: Morpheme Decomposition
    print("Level 2: Morpheme Decomposition")
    try:
        level2_affix_id = generator.generate_level2_affix_identification(n=tasks_per_level // 2, format="mcq")
        print(f"  ✓ Affix identification (MCQ): {len(level2_affix_id)} tasks")
        all_tasks_mcq.extend(level2_affix_id)

        level2_root_extract = generator.generate_level2_root_extraction(n=tasks_per_level // 2, format="mcq")
        print(f"  ✓ Root extraction (MCQ): {len(level2_root_extract)} tasks")
        all_tasks_mcq.extend(level2_root_extract)

        # Generative versions
        level2_gen = []
        level2_gen.extend(generator.generate_level2_affix_identification(n=tasks_per_level // 2, format="gen"))
        level2_gen.extend(generator.generate_level2_root_extraction(n=tasks_per_level // 2, format="gen"))
        all_tasks_gen.extend(level2_gen)
        print(f"  ✓ Generative: {len(level2_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 2 generation failed: {e}")

    print()

    # Level 3: Morpheme Manipulation
    print("Level 3: Morpheme Manipulation")
    try:
        level3_affix_remove = generator.generate_level3_affix_removal(n=tasks_per_level, format="mcq")
        print(f"  ✓ Affix removal (MCQ): {len(level3_affix_remove)} tasks")
        all_tasks_mcq.extend(level3_affix_remove)

        # Generative versions
        level3_gen = generator.generate_level3_affix_removal(n=tasks_per_level, format="gen")
        all_tasks_gen.extend(level3_gen)
        print(f"  ✓ Generative: {len(level3_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 3 generation failed: {e}")

    print()

    # Level 4: Morpheme Composition
    print("Level 4: Morpheme Composition")
    try:
        level4_affix_app = generator.generate_level4_affix_application(n=tasks_per_level, format="mcq")
        print(f"  ✓ Affix application (MCQ): {len(level4_affix_app)} tasks")
        all_tasks_mcq.extend(level4_affix_app)

        # Generative versions
        level4_gen = generator.generate_level4_affix_application(n=tasks_per_level, format="gen")
        all_tasks_gen.extend(level4_gen)
        print(f"  ✓ Generative: {len(level4_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 4 generation failed: {e}")

    print()

    # Level 5: Complex Reasoning
    print("Level 5: Complex Morphological Reasoning")
    try:
        level5_multi = generator.generate_level5_multi_step_transformation(n=tasks_per_level, format="mcq")
        print(f"  ✓ Multi-step transformation (MCQ): {len(level5_multi)} tasks")
        all_tasks_mcq.extend(level5_multi)

        # Generative versions
        level5_gen = generator.generate_level5_multi_step_transformation(n=tasks_per_level, format="gen")
        all_tasks_gen.extend(level5_gen)
        print(f"  ✓ Generative: {len(level5_gen)} tasks")
    except Exception as e:
        print(f"  ⚠ Level 5 generation failed: {e}")

    print()
    print(f"Total MCQ tasks: {len(all_tasks_mcq)}")
    print(f"Total Generative tasks: {len(all_tasks_gen)}")

    return all_tasks_mcq, all_tasks_gen


def save_tasks(tasks, output_file, format_type):
    """Save tasks to JSONL file."""
    with open(output_file, "w") as f:
        for task in tasks:
            if format_type == "mcq":
                f.write(json.dumps(task.to_mcq_dict(), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(task.to_gen_dict(), ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(tasks)} tasks to {output_file}")


def main():
    print("=" * 70)
    print("HIERARCHICAL PACUTE BENCHMARK GENERATION")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    words_df = load_words_for_tasks()
    print(f"  ✓ Loaded {len(words_df)} words")

    affixes_df = load_affix_annotations()
    print(f"  ✓ Loaded {len(affixes_df)} affix annotations")
    print("  ✓ Annotation format: prefix/infix/suffix columns")
    print()

    # Generate tasks
    mcq_tasks, gen_tasks = generate_all_levels(words_df, affixes_df, tasks_per_level=100)

    # Ensure output directory exists
    output_dir = project_root / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tasks
    print()
    print("Saving tasks...")
    save_tasks(mcq_tasks, output_dir / "hierarchical_mcq.jsonl", "mcq")
    save_tasks(gen_tasks, output_dir / "hierarchical_gen.jsonl", "gen")

    print()
    print("=" * 70)
    print("✅ Hierarchical benchmark generation complete!")
    print("=" * 70)

    # Statistics by level
    print()
    print("Tasks by Level:")
    for level in range(6):
        mcq_count = sum(1 for t in mcq_tasks if t.level == level)
        gen_count = sum(1 for t in gen_tasks if t.level == level)
        print(f"  Level {level}: {mcq_count} MCQ, {gen_count} generative")


if __name__ == "__main__":
    main()
