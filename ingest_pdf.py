import os
import fitz  # PyMuPDF
from preprocess import clean_text
from chunking import chunk_text
from embeddings import get_text_embedding
from vectorstore import init_faiss, load_faiss, save_faiss
from config import FAISS_INDEX_PATH, EMBEDDING_DIM


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def ingest_pdf(pdf_path: str, faiss_index_path: str = FAISS_INDEX_PATH) -> str:
    """
    Ingest PDF into FAISS vector store.
    - Extracts text
    - Cleans & chunks
    - Embeds chunks
    - Saves to FAISS
    Returns: status message for UI.
    """
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        return f"⚠️ No extractable text found in {os.path.basename(pdf_path)}"

    # Preprocess & chunk
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    # Load or init FAISS
    if os.path.exists(faiss_index_path):
        index, metadata = load_faiss(faiss_index_path)
    else:
        index, metadata = init_faiss(EMBEDDING_DIM)

    # Embed & add chunks
    for i, chunk in enumerate(chunks):
        vector = get_text_embedding(chunk)
        index.add(vector)
        metadata.append({
            "content": chunk,
            "source": os.path.basename(pdf_path),
            "chunk_id": i
        })

    # Save index + metadata
    save_faiss(index, metadata, faiss_index_path)

    return f"✅ Ingested {len(chunks)} chunks from {os.path.basename(pdf_path)}"
