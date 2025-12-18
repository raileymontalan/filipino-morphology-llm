"""
Context-Dependent Stress Tasks.

Tests understanding of stress patterns that distinguish homographs.
Filipino words like "pala" can have different meanings and stress patterns
depending on context.

Stress patterns:
- mabilis (á): acute accent - stressed last syllable
- malumi (à): grave accent - stressed penultimate
- maragsa (â): circumflex - glottal stop
- malumay: no accent - default stress
"""

import json
import random
from collections import defaultdict
from typing import Any, Dict, List


def load_homographs(syllables_file: str) -> Dict[str, List[Dict]]:
    """
    Load words that have multiple stress patterns (homographs).

    Returns:
        Dictionary mapping normalized word -> list of entries with different stress
    """
    word_entries = defaultdict(list)

    with open(syllables_file) as f:
        for line in f:
            data = json.loads(line)
            normalized = data.get("normalized_word", "").lower()
            if normalized:
                word_entries[normalized].append(data)

    # Filter to only homographs (multiple stress patterns)
    homographs = {}
    for word, entries in word_entries.items():
        stress_patterns = set(e.get("last_syllable_pronunciation", "") for e in entries)
        if len(stress_patterns) > 1:
            homographs[word] = entries

    return homographs


def create_synthetic_context(word: str, sense: str, pos: str) -> str:
    """
    Create a simple context sentence for a word.

    Uses part of speech to generate appropriate sentence frames.
    """
    # Simple sentence templates by part of speech
    templates = {
        "pnb": [  # pangngalan (noun)
            f"Ang {word} ay ...",
            f"May {word} sa ...",
            f"Nakita ko ang {word}.",
        ],
        "pdw": [  # pang-abay (adverb)
            f"{word.capitalize()} ako ...",
            f"Umuwi {word} ...",
            f"{word.capitalize()} siyang ...",
        ],
        "pdd": [  # pandamdam (interjection)
            f"{word.capitalize()}!",
            f'Sabi niya, "{word.capitalize()}!"',
        ],
        "pnr": [  # pang-uri (adjective)
            f"{word.capitalize()} ang ...",
            f"May {word} na ...",
        ],
        "pkw": [  # pandiwa (verb)
            f"Gusto kong {word}.",
            f"Kailangan {word} ...",
            f"Ayaw {word} ...",
        ],
    }

    # Get templates for this part of speech
    pos_templates = templates.get(pos, templates["pnb"])
    return random.choice(pos_templates)


def create_stress_identification_task(word: str, entries: List[Dict], format: str = "mcq") -> Dict[str, Any]:
    """
    Create task: "Which syllable has stress in 'pala' in this context?".

    Example:
        Context: "Nandiyan ka na pala."
        Question: Aling pantig ang may diin sa salitang "pala"?
        Answer: 2nd syllable (pa-LÁ)
    """
    # Pick one entry randomly
    entry = random.choice(entries)

    normalized = entry["normalized_word"]
    accented = entry["word"]
    syllables = entry["normalized_syllable_list"]
    stress_index = entry.get("accented_syllable_index", 0)
    sense = entry.get("word_sense", "")[:80]
    pos = entry.get("part_of_speech", "pnb")

    # Create context
    context = create_synthetic_context(normalized, sense, pos)

    # English and Tagalog prompts
    prompt_en = f'Which syllable is stressed in the word "{normalized}" in this context: "{context}"?'
    prompt_tl = f'Aling pantig ang may diin sa salitang "{normalized}" sa pangungusap na: "{context}"?'

    # Answer: ordinal (1st, 2nd, etc.)
    ordinals_en = ["1st", "2nd", "3rd", "4th", "5th"]
    ordinals_tl = ["una", "ikalawa", "ikatlo", "ikaapat", "ikalima"]

    if stress_index < len(ordinals_en):
        answer_en = ordinals_en[stress_index]
        answer_tl = ordinals_tl[stress_index]
    else:
        answer_en = f"{stress_index + 1}th"
        answer_tl = f"ika-{stress_index + 1}"

    if format == "mcq":
        # Options: all syllable positions
        options_en = [ordinals_en[i] for i in range(len(syllables))]
        options_tl = [ordinals_tl[i] for i in range(len(syllables))]

        return {
            "category": "stress_identification",
            "subcategory": "syllable_position",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "accented_form": accented,
            "answer_en": answer_en,
            "answer_tl": answer_tl,
            "options_en": options_en,
            "options_tl": options_tl,
            "syllables": syllables,
            "stress_index": stress_index,
        }
    else:  # generative
        return {
            "category": "stress_identification",
            "subcategory": "syllable_position",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "accented_form": accented,
            "answer_en": answer_en,
            "answer_tl": answer_tl,
            "syllables": syllables,
            "stress_index": stress_index,
        }


def create_stress_classification_task(word: str, entries: List[Dict], format: str = "mcq") -> Dict[str, Any]:
    """
    Create task: "What type of stress does 'pala' have in this context?".

    Example:
        Context: "Nandiyan ka na pala."
        Question: Anong klase ng diin ang meron sa salitang "pala"?
        Answer: mabilis (stressed last syllable)
    """
    # Pick one entry randomly
    entry = random.choice(entries)

    normalized = entry["normalized_word"]
    accented = entry["word"]
    stress_type = entry.get("last_syllable_pronunciation", "malumay")
    sense = entry.get("word_sense", "")[:80]
    pos = entry.get("part_of_speech", "pnb")

    # Create context
    context = create_synthetic_context(normalized, sense, pos)

    # Prompts
    prompt_en = f'What type of stress does the word "{normalized}" have in this context: "{context}"?'
    prompt_tl = f'Anong klase ng diin ang meron sa salitang "{normalized}" sa pangungusap na: "{context}"?'

    # Stress type mappings
    stress_labels = {
        "mabilis": "mabilis (acute á)",
        "malumi": "malumi (grave à)",
        "maragsa": "maragsa (circumflex â)",
        "malumay": "malumay (unmarked)",
    }

    answer = stress_labels.get(stress_type, stress_type)

    if format == "mcq":
        # All stress types as options
        options = list(stress_labels.values())
        random.shuffle(options)

        return {
            "category": "stress_classification",
            "subcategory": "stress_type",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "accented_form": accented,
            "answer": answer,
            "options": options,
            "stress_type": stress_type,
        }
    else:  # generative
        return {
            "category": "stress_classification",
            "subcategory": "stress_type",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "accented_form": accented,
            "answer": answer,
            "stress_type": stress_type,
        }


def create_deaccent_task(word: str, entries: List[Dict], format: str = "mcq") -> Dict[str, Any]:
    """
    Create task: Given context, provide the correctly accented form.

    Example:
        Word: pala (no accents)
        Context: "Nandiyan ka na pala."
        Question: How should "pala" be written with accent marks in this context?
        Answer: palá
    """
    # Pick one entry randomly
    entry = random.choice(entries)

    normalized = entry["normalized_word"]
    accented = entry["word"]
    sense = entry.get("word_sense", "")[:80]
    pos = entry.get("part_of_speech", "pnb")

    # Create context
    context = create_synthetic_context(normalized, sense, pos)

    # Prompts
    prompt_en = f'How should the word "{normalized}" be written with proper accent marks in this context: "{context}"?'
    prompt_tl = (
        f'Paano isusulat ang salitang "{normalized}" na may tamang marka ng diin sa pangungusap na: "{context}"?'
    )

    answer = accented

    if format == "mcq":
        # Options: all variants of this word with different accents
        options = [e["word"] for e in entries if e["word"] != accented]

        # If we don't have enough variants, create some
        while len(options) < 3:
            options.append(normalized)  # Unaccented version

        options = [answer] + options[:3]
        random.shuffle(options)

        return {
            "category": "deaccent",
            "subcategory": "accent_placement",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "answer": answer,
            "options": options,
        }
    else:  # generative
        return {
            "category": "deaccent",
            "subcategory": "accent_placement",
            "prompt_en": prompt_en,
            "prompt_tl": prompt_tl,
            "context": context,
            "word": normalized,
            "answer": answer,
        }


def generate_stress_benchmark(syllables_file: str, output_file: str, n_per_task: int = 100, format: str = "mcq"):
    """
    Generate stress task benchmark.

    Args:
        syllables_file: Path to syllables.jsonl
        output_file: Where to save tasks
        n_per_task: Number of tasks per type
        format: "mcq" or "gen"
    """
    print(f"Loading homographs from {syllables_file}...")
    homographs = load_homographs(syllables_file)
    print(f"Found {len(homographs)} homographs")

    all_tasks = []

    # Generate stress identification tasks
    print(f"\nGenerating {n_per_task} stress identification tasks...")
    sampled = random.sample(list(homographs.items()), min(n_per_task, len(homographs)))
    for word, entries in sampled:
        task = create_stress_identification_task(word, entries, format)
        all_tasks.append(task)

    # Generate stress classification tasks
    print(f"Generating {n_per_task} stress classification tasks...")
    sampled = random.sample(list(homographs.items()), min(n_per_task, len(homographs)))
    for word, entries in sampled:
        task = create_stress_classification_task(word, entries, format)
        all_tasks.append(task)

    # Generate deaccent tasks
    print(f"Generating {n_per_task} deaccent tasks...")
    sampled = random.sample(list(homographs.items()), min(n_per_task, len(homographs)))
    for word, entries in sampled:
        task = create_deaccent_task(word, entries, format)
        all_tasks.append(task)

    # Save
    print(f"\nSaving {len(all_tasks)} tasks to {output_file}...")
    with open(output_file, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(all_tasks)} {format} stress tasks")
    print("\nTask breakdown:")
    print(f"  - Stress identification: {n_per_task}")
    print(f"  - Stress classification: {n_per_task}")
    print(f"  - De-accent: {n_per_task}")


if __name__ == "__main__":
    # Generate MCQ and generative versions
    generate_stress_benchmark(
        syllables_file="data/corpora/pacute_data/syllables.jsonl",
        output_file="data/benchmarks/stress_mcq.jsonl",
        n_per_task=100,
        format="mcq",
    )

    generate_stress_benchmark(
        syllables_file="data/corpora/pacute_data/syllables.jsonl",
        output_file="data/benchmarks/stress_gen.jsonl",
        n_per_task=100,
        format="gen",
    )
