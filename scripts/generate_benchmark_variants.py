"""
Generate additional benchmark variants:
1. LangGame GEN: Convert MCQ format to generative
2. Multi-digit Addition MCQ: Add distractor options to generative format
3. CUTE GEN: Save local copy from HuggingFace
"""
import json
import random
import os
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def generate_langgame_gen():
    """Convert LangGame MCQ to generative format"""
    print("Generating LangGame GEN variant...")
    
    benchmarks_dir = Path("data/benchmarks")
    
    input_file = benchmarks_dir / "langgame_mcq.jsonl"
    output_file = benchmarks_dir / "langgame_gen.jsonl"
    
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # For GEN format, just need question and answer (first option)
            gen_sample = {
                "question": data["question"],
                "answer": data["answer"]
            }
            samples.append(gen_sample)
    
    # Save generative version
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Created {output_file} with {len(samples)} samples")
    print()


def generate_incorrect_answer(correct_answer):
    """Generate an incorrect answer for multi-digit addition"""
    correct = int(correct_answer)
    strategies = [
        # Strategy 1: Off by small amount
        lambda x: str(x + random.randint(1, 20)),
        lambda x: str(x - random.randint(1, 20)),
        # Strategy 2: Digit manipulation (swap digits)
        lambda x: ''.join(random.sample(str(x), len(str(x)))) if len(str(x)) > 1 else str(x + 1),
        # Strategy 3: Add/remove a digit
        lambda x: str(x)[:-1] if len(str(x)) > 1 else str(x * 10),
        lambda x: str(x) + str(random.randint(0, 9)),
        # Strategy 4: Replace random digit
        lambda x: str(x)[:random.randint(0, len(str(x))-1)] + str(random.randint(0, 9)) + str(x)[random.randint(0, len(str(x))-1)+1:] if len(str(x)) > 1 else str(x+1),
        # Strategy 5: Common arithmetic errors (carry errors, etc.)
        lambda x: str(x + 10) if x % 10 < 5 else str(x - 10),
        lambda x: str(x + 100) if x >= 100 else str(x + 10),
    ]
    
    # Try to generate a unique incorrect answer
    for _ in range(10):
        strategy = random.choice(strategies)
        try:
            incorrect = strategy(correct)
            # Ensure it's different from correct and is a valid number
            if incorrect != correct_answer and incorrect.isdigit() and int(incorrect) > 0:
                return incorrect
        except:
            continue
    
    # Fallback: just add/subtract a random amount
    offset = random.randint(1, 50)
    if random.random() > 0.5:
        return str(correct + offset)
    else:
        return str(max(1, correct - offset))


def generate_multi_digit_addition_mcq():
    """Convert multi-digit addition GEN to MCQ format with distractors"""
    print("Generating Multi-digit Addition MCQ variant...")
    
    benchmarks_dir = Path("data/benchmarks")
    
    gen_file = benchmarks_dir / "multi_digit_addition_gen.jsonl"
    mcq_file = benchmarks_dir / "multi_digit_addition_mcq.jsonl"
    
    # Generate MCQ version (limit to 1000 samples)
    samples = []
    max_samples = 1000
    
    with open(gen_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= max_samples:
                break
                
            data = json.loads(line.strip())
            correct_answer = data["answer"]
            
            # Generate 3 incorrect options
            incorrect_options = []
            attempts = 0
            while len(incorrect_options) < 3 and attempts < 20:
                incorrect = generate_incorrect_answer(correct_answer)
                if incorrect not in incorrect_options and incorrect != correct_answer:
                    incorrect_options.append(incorrect)
                attempts += 1
            
            # Ensure we have exactly 3 incorrect options
            while len(incorrect_options) < 3:
                offset = random.randint(1, 100)
                incorrect = str(int(correct_answer) + offset)
                if incorrect not in incorrect_options:
                    incorrect_options.append(incorrect)
            
            # Create MCQ sample with correct answer first, then incorrect options
            mcq_sample = {
                "question": data["question"],
                "answer": correct_answer,
                "options": [correct_answer] + incorrect_options
            }
            samples.append(mcq_sample)
    
    # Save MCQ version
    with open(mcq_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Created {mcq_file} with {len(samples)} samples")
    print()


def generate_cute_if_missing():
    """Generate CUTE if it doesn't exist yet"""
    benchmarks_dir = Path("data/benchmarks")
    output_file = benchmarks_dir / "cute_gen.jsonl"
    
    if output_file.exists():
        print("CUTE dataset already exists, skipping...")
        print()
        return
    
    print("CUTE dataset not found, generating...")
    print("Note: CUTE should be generated with: python src/evaluation/datasets/scripts/generate_cute_benchmark.py")
    print("      or: python scripts/generate_benchmarks.py --benchmarks cute")
    print()
    
    try:
        from evaluation.datasets.scripts.generate_cute_benchmark import main as generate_cute_main
        generate_cute_main()
    except Exception as e:
        print(f"  ⚠ Could not generate CUTE: {e}")
        print()


def main():
    print("="*70)
    print("Generating Benchmark Variants")
    print("="*70)
    print()
    
    # 1. Generate LangGame generative version
    generate_langgame_gen()
    
    # 2. Generate multi-digit addition MCQ version
    generate_multi_digit_addition_mcq()
    
    # 3. Check/generate CUTE if missing
    generate_cute_if_missing()
    
    print("="*70)
    print("Summary of Generated Files:")
    print("="*70)
    benchmarks_dir = Path("data/benchmarks")
    
    files_to_check = [
        "langgame_gen.jsonl",
        "langgame_mcq.jsonl",
        "multi_digit_addition_gen.jsonl",
        "multi_digit_addition_mcq.jsonl",
        "cute_gen.jsonl",
    ]
    
    for filename in files_to_check:
        filepath = benchmarks_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            with open(filepath, 'r') as f:
                count = sum(1 for _ in f)
            print(f"  ✓ {filename:<40} {count:>6} samples  {size:>8.1f} KB")
        else:
            print(f"  ✗ {filename:<40} NOT FOUND")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
