from typing import Optional, Tuple, List, Dict
from retriever import retrieve_documents, retrieve_with_image
from gemini_api import query_gemini


def multimodal_rag(
    text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    faiss_index_path: str = "data/faiss_index/index.faiss",
    top_k: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Multimodal Retrieval-Augmented Generation pipeline.
    
    Args:
        text (str): User query text
        image_bytes (bytes): Optional image for multimodal input
        faiss_index_path (str): Path to FAISS index
        top_k (int): Number of documents to retrieve
    
    Returns:
        answer (str): Gemini grounded answer
        retrieved (list[dict]): Retrieved chunks with metadata
    """

    retrieved = []

    # Retrieve text-based docs
    if text:
        retrieved = retrieve_documents(text, faiss_index_path, top_k=top_k)

    # Retrieve image-based docs
    if image_bytes:
        retrieved = retrieve_with_image(image_bytes, faiss_index_path, top_k=top_k)

    # Combine retrieved context
    context = "\n\n".join(
        [f"Source: {r.get('source', 'unknown')} | Chunk {r.get('chunk_id')}\n{r['content']}"
         for r in retrieved]
    )

    # Query Gemini with context + multimodal input
    answer = query_gemini(
        text=text,
        image_path=None,  # Optionally save image and pass path
        context=context
    )

    return answer, retrieved
