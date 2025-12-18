"""
A collection of metrics for evaluating models
"""

import torch
from trainers.utils import aggregate_value


def accuracy_metric(confidences):
    """
    Calculate the accuracy of the model over a path_prob.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        accuracy: float of accuracy
    """
    _, predicted = torch.max(confidences, 1)
    return aggregate_value((predicted == 0).float().mean())


def path_confidence(confidences):
    """
    Calculate the path confidence of the model.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        path_confidence: float of path confidences
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0]
    return aggregate_value(softmaxed.mean())


def ground_confidence(confidences):
    """
    Calculate the confidence of the model on the ground truth.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        ground_confidence: float of confidence on ground truth
    See: https://arxiv.org/pdf/2406.04391 - this is equivalent to
    $$P_\\theta^{\\text{Choices}}(\\text{Ground Truth})$$ over the
    Path probabilities. (takeaway 3)
    """
    return confidences[:, 0].mean()


def num_options(confidences):
    """Calculate the number of options."""
    return confidences.shape[1]


def normalized_accuracy(confidences):
    """
    Accuracy of model normalized so that 0 is random guessing and 1 is perfect accuracy.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        accuracy: float of accuracy
    """
    _, predicted = torch.max(confidences, 1)
    accuracy = (predicted == 0).float().mean()
    num_options = confidences.shape[1]
    normalized_accuracy = (accuracy * num_options - 1) / (num_options - 1)
    return aggregate_value(normalized_accuracy)


def normalized_path_confidence(confidences):
    """
    Path confidence normalized so that 0 is all having the same confidence and 1
    is much higher confidence on the ground truth.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        path_confidence: float of path confidences
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0].mean()
    num_options = confidences.shape[1]
    normalized_path_confidence = (softmaxed * num_options - 1) / (num_options - 1)
    return aggregate_value(normalized_path_confidence)


def f1_score_metric(confidences):
    """
    Calculate F1 score for binary classification (correct vs incorrect)
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        f1_score: float F1 score
    """
    _, predicted = torch.max(confidences, 1)
    # True positives: predicted 0 (correct) and it is 0 (correct)
    # Since ground truth is always at index 0, TP is when predicted == 0
    tp = (predicted == 0).float().sum()

    # False positives: predicted 0 but ground truth is not 0
    # Since ground truth is always 0, FP = 0 in this setup
    fp = 0

    # False negatives: predicted != 0 but ground truth is 0
    # This is when we fail to select the correct answer
    fn = (predicted != 0).float().sum()

    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return aggregate_value(f1)


def precision_metric(confidences):
    """
    Calculate precision for MCQ.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        precision: float precision score
    """
    _, predicted = torch.max(confidences, 1)
    tp = (predicted == 0).float().sum()
    fp = 0  # No false positives since we only predict one answer
    precision = tp / (tp + fp + 1e-10)
    return aggregate_value(precision)


def recall_metric(confidences):
    """
    Calculate recall for MCQ.

    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        recall: float recall score
    """
    _, predicted = torch.max(confidences, 1)
    tp = (predicted == 0).float().sum()
    fn = (predicted != 0).float().sum()
    recall = tp / (tp + fn + 1e-10)
    return aggregate_value(recall)


MCQ_METRIC_DICT = {
    "accuracy": accuracy_metric,
    "f1_score": f1_score_metric,
    "precision": precision_metric,
    "recall": recall_metric,
    "path_confidence": path_confidence,
    "ground_confidence": ground_confidence,
    "num_options": num_options,
    "normalized_accuracy": normalized_accuracy,
    "normalized_path_confidence": normalized_path_confidence,
}
