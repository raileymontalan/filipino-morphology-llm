"""
Loader for Hierarchical benchmark - diagnostic tasks across 6 compositional levels.
"""
import json
from pathlib import Path


def load_hierarchical(format="mcq", **kwargs):
    """
    Load hierarchical benchmark.
    
    Args:
        format: 'mcq' or 'gen'
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of task dictionaries
    """
    benchmarks_dir = Path("data/benchmarks")
    filepath = benchmarks_dir / f"hierarchical_{format}.jsonl"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Hierarchical benchmark file not found: {filepath}\n"
            f"Generate it with: python src/evaluation/datasets/scripts/generate_hierarchical_benchmark.py"
        )
    
    tasks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line.strip())
            tasks.append(task)
    
    print(f"Loaded {len(tasks)} hierarchical tasks ({format} format)")
    return tasks
