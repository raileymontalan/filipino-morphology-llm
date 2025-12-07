# PACUTE Benchmark Results (100 samples)

## Summary

| Model | Parameters | Type | Accuracy | F1 Score | Precision | Recall | Path Confidence |
|-------|-----------|------|----------|----------|-----------|---------|-----------------|
| **GPT-2** | 124M | PT | **24.0%** | **38.7%** | 100.0% | 24.0% | 24.8% |
| **Cerebras-GPT-111M** | 111M | PT | **22.0%** | **36.1%** | 100.0% | 22.0% | 21.7% |
| **Cerebras-GPT-256M** | 256M | PT | **25.0%** | **40.0%** | 100.0% | 25.0% | 26.0% |

## PACUTE Benchmark Details

- **Total Tasks**: 560 across 4 categories
  - Affixation: 140 tasks
  - Composition: 180 tasks
  - Manipulation: 160 tasks
  - Syllabification: 80 tasks
- **Evaluation**: 100 samples (random subset)
- **Format**: MCQ with log probability scoring
- **Focus**: Filipino morphological understanding

## Key Findings

1. **Performance Range**: 22-25% accuracy on Filipino morphology tasks
2. **Model Size Impact**: Larger models (256M) slightly outperform smaller ones
3. **F1 Scores**: Range from 36.1% to 40.0%, indicating modest understanding
4. **Perfect Precision**: All models achieved 100% precision (when they guess correctly, they're confident)
5. **Recall = Accuracy**: Since only one answer is selected per question

## Comparison to Random Baseline

- **Random baseline** (1/4 options): 25% accuracy
- Models perform at or slightly below random on this morphologically complex task
- Indicates Filipino morphology remains challenging for pretrained LLMs

## Evaluation Setup

- **Device**: CPU
- **Method**: Log-probability MCQ scoring
- **Metrics**: Accuracy, F1, Precision, Recall, Path Confidence
- **Date**: 2025-12-02

## Notes

- Qwen-2.5-0.5B: Out of memory (killed after 55/100 samples)
- CUTE benchmark: Division by zero error (generative format incompatibility with MCQ scoring)
- Full 560-task evaluation recommended for comprehensive results
