import os
import faiss
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any

# ----------------------------
# INIT FAISS
# ----------------------------
def init_faiss(d: int = 768) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Initialize FAISS index (Inner Product for cosine similarity).
    Returns (index, metadata_list).
    """
    index = faiss.IndexFlatIP(d)  # cosine similarity via inner product (requires normalized vectors)
    metadata: List[Dict[str, Any]] = []
    return index, metadata


# ----------------------------
# SAVE FAISS
# ----------------------------
def save_faiss(index: faiss.IndexFlatIP, metadata: List[Dict[str, Any]], path: str):
    """
    Save FAISS index + metadata to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, path)

    # Save metadata
    meta_path = path + ".meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


# ----------------------------
# LOAD FAISS
# ----------------------------
def load_faiss(path: str) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Load FAISS index + metadata from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ FAISS index not found at {path}")

    index = faiss.read_index(path)

    meta_path = path + ".meta.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = []

    return index, metadata


# ----------------------------
# SEARCH FAISS
# ----------------------------
def search_faiss(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    metadata: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search FAISS index with query vector.
    Returns top-k results with metadata + scores.
    """
    if index.ntotal == 0:
        return []

    D, I = index.search(query_vector, top_k)  # distances, indices
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            entry = metadata[idx].copy()
            entry["score"] = float(dist)
            results.append(entry)

    return results
