from typing import List, Dict, Any

def evaluate_responses(responses: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    metrics = {
        "accuracy": calculate_accuracy(responses, ground_truth),
        "precision": calculate_precision(responses, ground_truth),
        "recall": calculate_recall(responses, ground_truth),
        "f1_score": calculate_f1_score(responses, ground_truth),
    }
    return metrics

def calculate_accuracy(responses: List[str], ground_truth: List[str]) -> float:
    correct = sum(1 for response, truth in zip(responses, ground_truth) if response == truth)
    return correct / len(ground_truth) if ground_truth else 0.0

def calculate_precision(responses: List[str], ground_truth: List[str]) -> float:
    true_positive = sum(1 for response in responses if response in ground_truth)
    predicted_positive = len(responses)
    return true_positive / predicted_positive if predicted_positive > 0 else 0.0

def calculate_recall(responses: List[str], ground_truth: List[str]) -> float:
    true_positive = sum(1 for response in responses if response in ground_truth)
    actual_positive = len(ground_truth)
    return true_positive / actual_positive if actual_positive > 0 else 0.0

def calculate_f1_score(responses: List[str], ground_truth: List[str]) -> float:
    precision = calculate_precision(responses, ground_truth)
    recall = calculate_recall(responses, ground_truth)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def generate_report(metrics: Dict[str, Any]) -> str:
    report = "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())
    return report