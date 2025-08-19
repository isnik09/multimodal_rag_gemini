from typing import Optional
from retriever import retrieve_documents, retrieve_with_image
from gemini_api import query_gemini


# ----------------------------
# MULTIMODAL RAG PIPELINE
# ----------------------------
def multimodal_rag(
    text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    faiss_index_path: str = "data/faiss_index/index.faiss",
    top_k: int = 5
) -> str:
    """
    Multimodal Retrieval-Augmented Generation pipeline.
    - If text provided → retrieve relevant chunks.
    - If image provided → retrieve based on image embedding.
    - Combine context with query.
    - Query Gemini for grounded answer.
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

    return answer
