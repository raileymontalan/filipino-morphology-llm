"""
Generate Multi-digit Addition Benchmark

Creates simple arithmetic tasks for evaluating numerical reasoning.
Output: data/benchmarks/multi_digit_addition_gen.jsonl (evaluation data only)
"""

import os
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path
# Go up 5 levels: scripts -> datasets -> evaluation -> src -> filipino-morphology-llm
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def generate_simple_questions(num_digits, train_size, val_size):
    """Generate simple arithmetic questions without tokenization."""
    total_size = train_size + val_size
    all_number_pairs = np.arange(10**(2*num_digits-1), 10**(2*num_digits))
    assert len(all_number_pairs) >= total_size, f"{len(all_number_pairs)} < {total_size}"
    np.random.shuffle(all_number_pairs)
    
    questions = []
    answers = []
    
    for number_pair in all_number_pairs[:total_size]:
        number1 = number_pair // 10**num_digits
        number2 = number_pair % 10**num_digits
        answer = str(number1 + number2)
        question = f"{number1}+{number2}="
        questions.append(question)
        answers.append(answer)
    
    train_questions = questions[:train_size]
    train_answers = answers[:train_size]
    val_questions = questions[train_size:]
    val_answers = answers[train_size:]
    
    return train_questions, train_answers, val_questions, val_answers


def main():
    """Generate multi-digit addition benchmark."""
    print("="*70)
    print("MULTI-DIGIT ADDITION BENCHMARK GENERATION")
    print("="*70)
    
    # Configuration
    num_digits = 3
    total_numbers = 10**(2*num_digits) - 10**(2*num_digits-1)
    val_size = total_numbers // 10
    train_size = total_numbers - val_size
    
    print(f"Configuration:")
    print(f"  Number of digits: {num_digits}")
    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    print()
    
    # Generate simple questions (always works)
    print("Generating arithmetic questions...")
    train_questions, train_answers, val_questions, val_answers = generate_simple_questions(
        num_digits, train_size, val_size
    )
    print(f"✓ Generated {len(train_questions)} train and {len(val_questions)} val samples")
    print()
    
    # Save as JSONL for benchmarking
    benchmarks_path = project_root / "data" / "benchmarks"
    benchmarks_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving JSONL benchmark to {benchmarks_path}...")
    
    # Evaluation data only (no train split needed)
    gen_jsonl_path = benchmarks_path / "multi_digit_addition_gen.jsonl"
    with open(gen_jsonl_path, 'w', encoding='utf-8') as f:
        for i in range(len(val_questions)):
            item = {
                "question": val_questions[i],
                "answer": val_answers[i],
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(val_questions)} samples to {gen_jsonl_path}")
    print()
    
    print("="*70)
    print(f"✓ Multi-digit addition benchmark generation complete!")
    print(f"✓ Output location: {benchmarks_path}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
