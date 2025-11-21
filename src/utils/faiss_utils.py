"""FAISS index utilities for loading and saving indices."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import faiss
import numpy as np


def load_faiss_index(
    index_dir: Path
) -> Tuple[faiss.Index, np.ndarray, Dict[str, Any]]:
    """
    Load FAISS index, identifiers, and metadata.
    
    Args:
        index_dir: Directory containing the FAISS index files
        
    Returns:
        Tuple of (index, identifiers, metadata)
        
    Raises:
        FileNotFoundError: If index or identifiers file is missing
    """
    print(f"Loading FAISS index from {index_dir}...")

    index_path = index_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(index_path))

    identifiers_path = index_dir / "identifiers.npy"
    if not identifiers_path.exists():
        raise FileNotFoundError(
            f"Identifiers not found at {identifiers_path}")

    identifiers = np.load(identifiers_path)

    metadata_path = index_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    print(f"Loaded index with {index.ntotal} vectors")
    return index, identifiers, metadata


def save_faiss_index(
    index: faiss.Index,
    identifiers: np.ndarray,
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """
    Save FAISS index, identifiers, and metadata.
    
    Args:
        index: FAISS index to save
        identifiers: Array of image identifiers
        output_dir: Directory to save files
        metadata: Metadata dictionary to save
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "index.faiss"
    identifiers_path = output_dir / "identifiers.npy"
    metadata_path = output_dir / "metadata.json"

    print(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Saving identifiers to {identifiers_path}")
    np.save(identifiers_path, identifiers)

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

