import numpy as np
from typing import List


def calculate_mcc_score(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate the Matthews Correlation Coefficient (MCC).
    
    MCC is a balanced measure for binary classification that considers true/false positives
    and true/false negatives, and is generally regarded as a balanced measure for binary
    classification even when the classes are of very different sizes.
    
    Args:
        y_true (List[int]): List of true binary labels (0 or 1).
        y_pred (List[int]): List of predicted binary labels (0 or 1).
    
    Returns:
        float: The MCC score ranging from -1 to 1, where:
            1 = perfect prediction
            0 = random prediction
            -1 = inverse prediction
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Convert lists to numpy arrays for easier computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate MCC using the formula
    # MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    # Handle the case where denominator is 0 to avoid division by zero
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def calculate_mr_score(mcc_solution: float, error_location_accuracy: float, solution_accuracy: float) -> float:
    """Calculate the Meta-Reasoning (MR) score for MR-GSM8K benchmark.
    
    This score combines multiple metrics to provide an overall evaluation of the model's
    meta-reasoning capabilities, particularly in identifying errors in mathematical solutions.
    
    Args:
        mcc_solution (float): MCC score for solution correctness prediction.
        error_location_accuracy (float): Accuracy in identifying the location of the first error.
        solution_accuracy (float): Accuracy in predicting whether the solution steps are correct.
    
    Returns:
        float: The combined MR score, scaled to be between 0 and 1, where higher is better.
    """
    # Normalize MCC from [-1, 1] to [0, 1]
    normalized_mcc = (mcc_solution + 1) / 2
    
    # Assign weights to different components based on their importance
    # MCC solution is weighted highest as it balances performance across classes
    # Error location accuracy is weighted second as it's the core meta-reasoning capability
    # Solution accuracy is included for completeness
    weights = {
        'mcc_solution': 0.5,
        'error_location_accuracy': 0.3,
        'solution_accuracy': 0.2
    }
    
    # Calculate weighted score
    mr_score = (
        weights['mcc_solution'] * normalized_mcc +
        weights['error_location_accuracy'] * error_location_accuracy +
        weights['solution_accuracy'] * solution_accuracy
    )
    
    # Ensure score is within [0, 1] range
    mr_score = max(0.0, min(1.0, mr_score))
    
    return mr_score


def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate the accuracy score.
    
    Args:
        y_true (List[int]): List of true binary labels (0 or 1).
        y_pred (List[int]): List of predicted binary labels (0 or 1).
    
    Returns:
        float: The accuracy score ranging from 0 to 1.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0