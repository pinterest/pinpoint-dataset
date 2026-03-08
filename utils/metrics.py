"""Retrieval metrics calculation utilities."""

from typing import List, Dict


def calculate_metrics_at_k(
    retrieved: List[str], relevant: List[str], k: int
) -> Dict[str, float]:
    """
    Calculate precision, recall, and Average Precision (AP) at k.

    Args:
        retrieved: List of retrieved item identifiers
        relevant: List of relevant item identifiers
        k: Cutoff value for top-k metrics

    Returns:
        Dictionary with 'precision', 'recall', and 'ap' keys
    """
    retrieved_k = retrieved[:k]

    if not relevant:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "ap": 0.0
        }

    relevant_set = set(relevant)

    # Calculate precision@k
    hits = sum(1 for item in retrieved_k if item in relevant_set)
    precision = hits / k if k > 0 else 0.0

    # Calculate recall@k
    recall = hits / len(relevant) if relevant else 0.0

    # Calculate AP@k
    ap = 0.0
    hits_so_far = 0
    for i, item in enumerate(retrieved_k):
        if item in relevant_set:
            hits_so_far += 1
            ap += hits_so_far / (i + 1)

    ap = ap / min(len(relevant), k) if relevant else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "ap": ap
    }


def calculate_neg_recall_at_k(
    retrieved: List[str], negative: List[str], k: int
) -> float:
    """
    Calculate negative recall at k (fraction of negatives retrieved).
    
    Args:
        retrieved: List of retrieved item identifiers
        negative: List of negative item identifiers
        k: Cutoff value for top-k
        
    Returns:
        Negative recall value (0.0 to 1.0)
    """
    if not negative:
        return 0.0

    negative_set = set(negative)
    retrieved_k = retrieved[:k]
    neg_hits = sum(1 for item in retrieved_k if item in negative_set)
    return neg_hits / min(len(negative), k)

