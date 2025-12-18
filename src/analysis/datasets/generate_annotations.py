"""
Create Morpheme Annotations for Hierarchical Tasks.

Generates a dataset of Filipino words with morpheme boundary annotations.
Uses inflections data and syllabified words to create annotations for:
- Affixation patterns (prefixes, infixes, suffixes, circumfixes)
- Morpheme boundaries for tokenization analysis
- Examples for hierarchical task generation

Output format (JSONL):
{
    "word": "tumakbo",
    "root": "takbo",
    "morphemes": ["um", "takbo"],
    "morpheme_types": ["infix", "root"],
    "affix": "um",
    "affix_type": "infix",
    "affix_position": "after_first_consonant"
}
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_affixes(affixes_file: str) -> List[str]:
    """Load Filipino affixes from file."""
    with open(affixes_file) as f:
        return [line.strip() for line in f if line.strip()]


def parse_aspect_affix(affix_str: str) -> Tuple[str, str]:
    """
    Parse affix string to extract affix and type.

    Examples:
        "mag-" → ("mag", "prefix")
        "-um-" → ("um", "infix")
        "-an" → ("an", "suffix")
        "mag--an" → ("mag-an", "circumfix")
    """
    affix_str = str(affix_str).strip()

    if pd.isna(affix_str) or not affix_str:
        return None, None

    # Remove quotes if present
    affix_str = affix_str.strip("'\"")

    # Circumfix: "mag--an"
    if "--" in affix_str:
        affix = affix_str.replace("--", "-")
        return affix, "circumfix"

    # Infix: "-um-", "-in-"
    if affix_str.startswith("-") and affix_str.endswith("-"):
        affix = affix_str.strip("-")
        return affix, "infix"

    # Prefix: "mag-", "nag-"
    if affix_str.endswith("-"):
        affix = affix_str.rstrip("-")
        return affix, "prefix"

    # Suffix: "-an", "-in"
    if affix_str.startswith("-"):
        affix = affix_str.lstrip("-")
        return affix, "suffix"

    # Default: treat as prefix if no markers
    return affix_str, "prefix"


def find_infix_position(inflected: str, root: str, infix: str) -> Optional[int]:
    """
    Find position where infix was inserted.

    Filipino infixes typically go after the first consonant:
    - takbo + um → tumakbo (after 't')
    - sulat + in → sinulat (after 's')
    """
    inflected_lower = inflected.lower()
    root_lower = root.lower()
    infix_lower = infix.lower()

    # Try to find where infix appears in inflected form
    for i in range(1, len(root_lower) + 1):
        # Try inserting infix at position i
        candidate = root_lower[:i] + infix_lower + root_lower[i:]
        if candidate == inflected_lower:
            return i

    return None


def annotate_from_inflections(inflections_df: pd.DataFrame) -> List[Dict]:
    """Create annotations from inflections data."""
    annotations = []

    for _, row in inflections_df.iterrows():
        root = str(row["root"]).lower()
        inflected = str(row["inflected"]).lower()
        aspect_affix = row["aspect_affix"]

        if pd.isna(aspect_affix):
            continue

        affix, affix_type = parse_aspect_affix(aspect_affix)

        if not affix or not affix_type:
            continue

        # Create annotation based on affix type
        annotation = {
            "word": inflected,
            "root": root,
            "affix": affix,
            "affix_type": affix_type,
            "aspect": str(row["aspect"]),
            "focus": str(row["focus"]),
        }

        if affix_type == "prefix":
            annotation["morphemes"] = [affix, root]
            annotation["morpheme_types"] = ["prefix", "root"]
            annotation["boundaries"] = [len(affix)]

        elif affix_type == "infix":
            pos = find_infix_position(inflected, root, affix)
            if pos:
                # Split root at infix position
                before = root[:pos]
                after = root[pos:]
                annotation["morphemes"] = [before, affix, after]
                annotation["morpheme_types"] = ["root_part", "infix", "root_part"]
                annotation["boundaries"] = [len(before), len(before) + len(affix)]
                annotation["infix_position"] = pos
            else:
                # Fallback: mark as infix but don't split
                annotation["morphemes"] = [affix, root]
                annotation["morpheme_types"] = ["infix", "root"]

        elif affix_type == "suffix":
            annotation["morphemes"] = [root, affix]
            annotation["morpheme_types"] = ["root", "suffix"]
            annotation["boundaries"] = [len(root)]

        elif affix_type == "circumfix":
            # Split circumfix: "mag-an" → ["mag", "an"]
            parts = affix.split("-")
            if len(parts) == 2:
                prefix, suffix = parts
                annotation["morphemes"] = [prefix, root, suffix]
                annotation["morpheme_types"] = ["prefix", "root", "suffix"]
                annotation["boundaries"] = [len(prefix), len(prefix) + len(root)]

        annotations.append(annotation)

    return annotations


def find_affixed_words_in_syllables(syllables_file: str, affixes: List[str], max_per_affix: int = 10) -> List[Dict]:
    """
    Find words in syllables.jsonl that contain known affixes.
    Creates simple annotations for prefix matches.
    """
    annotations = []
    affix_counts = defaultdict(int)

    with open(syllables_file) as f:
        for line in f:
            data = json.loads(line)
            word = data.get("normalized_word", "").lower()

            if not word or len(word) < 4:
                continue

            # Check for prefix matches
            for affix in affixes:
                if affix_counts[affix] >= max_per_affix:
                    continue

                if word.startswith(affix) and len(word) > len(affix):
                    root = word[len(affix) :]

                    # Simple heuristic: root should be at least 2 chars
                    if len(root) >= 2:
                        annotation = {
                            "word": word,
                            "root": root,
                            "affix": affix,
                            "affix_type": "prefix",
                            "morphemes": [affix, root],
                            "morpheme_types": ["prefix", "root"],
                            "boundaries": [len(affix)],
                            "source": "syllables_heuristic",
                            "syllables": data.get("normalized_syllable_list", []),
                        }
                        annotations.append(annotation)
                        affix_counts[affix] += 1
                        break

    return annotations


def main():
    """Create morpheme annotations for hierarchical tasks."""
    print("Creating affix annotations...")
    print()

    # Load data
    print("1. Loading inflections data...")
    inflections_df = pd.read_excel("data/corpora/pacute_data/inflections.xlsx")
    print(f"   Loaded {len(inflections_df)} inflections")

    print("2. Loading affixes...")
    affixes = load_affixes("data/affixes/filipino_affixes.txt")
    print(f"   Loaded {len(affixes)} affixes")

    # Create annotations from inflections
    print("3. Creating annotations from inflections...")
    inflection_annotations = annotate_from_inflections(inflections_df)
    print(f"   Created {len(inflection_annotations)} annotations from inflections")

    # Find more examples in syllables data
    print("4. Finding affixed words in syllables data...")
    syllable_annotations = find_affixed_words_in_syllables(
        "data/corpora/pacute_data/syllables.jsonl", affixes, max_per_affix=10
    )
    print(f"   Created {len(syllable_annotations)} annotations from syllables")

    # Combine and deduplicate
    all_annotations = inflection_annotations + syllable_annotations

    # Deduplicate by word
    seen_words = set()
    unique_annotations = []
    for ann in all_annotations:
        if ann["word"] not in seen_words:
            unique_annotations.append(ann)
            seen_words.add(ann["word"])

    print(f"5. Total unique annotations: {len(unique_annotations)}")

    # Statistics
    print()
    print("Annotation Statistics:")
    by_type = defaultdict(int)
    for ann in unique_annotations:
        by_type[ann["affix_type"]] += 1

    for affix_type, count in sorted(by_type.items()):
        print(f"  {affix_type}: {count}")

    # Save
    output_file = "data/corpora/affix_annotations.jsonl"
    with open(output_file, "w") as f:
        for ann in unique_annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    print()
    print(f"✓ Saved annotations to {output_file}")

    # Show examples
    print()
    print("Sample Annotations:")
    print("-" * 70)
    for ann in unique_annotations[:5]:
        print(f"Word: {ann['word']}")
        print(f"  Morphemes: {' + '.join(ann['morphemes'])}")
        print(f"  Types: {' + '.join(ann['morpheme_types'])}")
        print(f"  Affix: {ann['affix']} ({ann['affix_type']})")
        if "boundaries" in ann:
            print(f"  Boundaries: {ann['boundaries']}")
        print()


if __name__ == "__main__":
    main()
