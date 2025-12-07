"""
CUTE: Character Understanding Test Evaluation

Tests character-level understanding through orthographic manipulation tasks.
Based on Edman et al. (2024) "CUTE: Measuring LLMs' Understanding of Their Tokens"

14 task types:
- Character-level: spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char
- Word-level: contains_word, ins_word, del_word, swap_word, sub_word
- Semantic/Orthographic: orth, sem

Total: 14,000 examples (1,000 per task)

Dataset: https://huggingface.co/datasets/leukas/cute
"""
import random


def load_cute(split="test", task_types=None, **kwargs):
    """
    Load CUTE benchmark from HuggingFace.

    Args:
        split: Not used (all data treated as test)
        task_types: List of task types to include. Options:
                   ['spell', 'spell_inverse', 'contains_char', 'contains_word',
                    'orth', 'sem', 'ins_char', 'ins_word', 'del_char', 'del_word',
                    'sub_char', 'sub_word', 'swap_char', 'swap_word']
                   If None, loads all task types.

    Yields:
        For MCQ format compatibility:
        - prefix: The prompt (question)
        - ground_truth: The correct answer
        - false_options: Empty list (generative task, not MCQ)

    Note: CUTE is a generative benchmark, not MCQ. The false_options will be empty.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for CUTE benchmark. "
            "Install it with: pip install datasets"
        )

    # Load from HuggingFace
    dataset = load_dataset("leukas/cute")

    # The dataset is organized by task types, not by train/test splits
    # Collect all tasks across all splits

    # Available task types (derived from task naming in prompts)
    all_task_types = [
        'spell', 'spell_inverse', 'contains_char', 'contains_word',
        'orth', 'sem', 'ins_char', 'ins_word', 'del_char', 'del_word',
        'sub_char', 'sub_word', 'swap_char', 'swap_word'
    ]

    if task_types is None:
        task_types = all_task_types

    # Collect tasks from the dataset
    # The dataset is organized by task type as splits
    tasks = []
    for task_type in all_task_types:
        if task_type in dataset and (task_types is None or task_type in task_types):
            for item in dataset[task_type]:
                tasks.append({
                    'prompt': item['prompt'],
                    'answer': item['answer'],
                    'task_type': task_type
                })

    total = len(tasks)
    print(f"CUTE: Loaded {total} character understanding tasks from HuggingFace.")
    print(f"Note: CUTE is a generative benchmark (prompt → answer), not MCQ format.")

    # Shuffle tasks
    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        prefix = task['prompt']
        ground_truth = task['answer']
        false_options = []  # Generative task, no MCQ options

        yield prefix, ground_truth, false_options
