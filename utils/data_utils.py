"""Data processing utilities."""

from typing import List


def normalize_query_id(query_id) -> str:
    """
    Normalize query ID to 5-digit string format.
    
    Args:
        query_id: Query ID (can be string, int, or other)
        
    Returns:
        Normalized 5-digit string (e.g., "00001")
    """
    if isinstance(query_id, str):
        if query_id.startswith("query_"):
            return query_id.replace("query_", "").zfill(5)
        return query_id.zfill(5)
    return str(query_id).zfill(5)


def filter_out_negatives(
    retrieved: List[str], negative: List[str]
) -> List[str]:
    """
    Remove negative examples from retrieved list, preserving order.
    
    Args:
        retrieved: List of retrieved item identifiers
        negative: List of negative item identifiers to remove
        
    Returns:
        Filtered list with negatives removed
    """
    if not retrieved or not negative:
        return retrieved
    neg_set = set(negative)
    return [item for item in retrieved if item not in neg_set]


