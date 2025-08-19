from typing import List, Dict, Any
from embeddings import get_text_embedding, get_image_embedding
from vectorstore import load_faiss, search_faiss


def retrieve_documents(
    query: str,
    faiss_index_path: str = "data/faiss_index/index.faiss",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant text chunks for a given query string.
    """
    # Load index + metadata
    index, metadata = load_faiss(faiss_index_path)

    # Embed query
    query_vector = get_text_embedding(query)

    # Search FAISS
    results = search_faiss(index, query_vector, metadata, top_k=top_k)
    return results


def retrieve_with_image(
    image_bytes: bytes,
    faiss_index_path: str = "data/faiss_index/index.faiss",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks using an image query.
    """
    # Load index + metadata
    index, metadata = load_faiss(faiss_index_path)

    # Embed image
    image_vector = get_image_embedding(image_bytes)

    # Search FAISS
    results = search_faiss(index, image_vector, metadata, top_k=top_k)
    return results
