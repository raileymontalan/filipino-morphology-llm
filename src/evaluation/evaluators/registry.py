"""
Given an evaluator name, load the evaluator.

NOTE: This registry is for the stochastok training pipeline.
      For standalone evaluation, use the loaders and evaluators directly.
"""

# Legacy imports - use new evaluation modules instead
# from evals.evaluator_interface import EvaluationInterface
# from evals.mcqs.mcq_evaluator import MCQEvaluator
# from evals.generation_evaluator_math import GenerationEvaluatorMath

from evaluation.evaluators.math_evaluator import GenerationEvaluatorMath
from evaluation.evaluators.mcq_evaluator import MCQEvaluator

EVALUATORS_DICT = {
    "mcq": MCQEvaluator,
    "math_generation": GenerationEvaluatorMath,
}


def load_evaluator(evaluator_name, model, **kwargs):
    """
    Given the evaluator name, load the evaluator.

    Args:
        evaluator_name: Name of the evaluator to load ('mcq' or 'math_generation')
        model: The model instance to evaluate
        **kwargs: Additional arguments to pass to the evaluator

    Returns:
        An evaluator instance
    """
    return EVALUATORS_DICT[evaluator_name](model, **kwargs)
