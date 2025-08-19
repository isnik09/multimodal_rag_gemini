import os
from pypdf import PdfReader
from utils.preprocess import clean_text
from utils.chunking import chunk_text, chunk_by_paragraphs
from embeddings import get_text_embedding
from vectorstore import init_faiss, save_faiss, load_faiss

# ----------------------------
# EXTRACT TEXT FROM PDF
# ----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and cleans text from PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ File not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    
    return clean_text(raw_text)


# ----------------------------
# INGEST PDF INTO FAISS
# ----------------------------
def ingest_pdf(
    pdf_path: str,
    faiss_index_path: str = "data/faiss_index/index.faiss",
    chunk_method: str = "words",
    chunk_size: int = 500,
    overlap: int = 50
):
    """
    Ingests PDF into FAISS index:
    - Extract text
    - Chunk
    - Embed
    - Store in FAISS
    """

    print(f"📥 Ingesting PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    # Choose chunking strategy
    if chunk_method == "words":
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    elif chunk_method == "paragraphs":
        chunks = chunk_by_paragraphs(text)
    else:
        raise ValueError(f"Unknown chunking method: {chunk_method}")

    print(f"📑 Extracted {len(chunks)} chunks")

    # Load or init FAISS
    if os.path.exists(faiss_index_path):
        index, metadata = load_faiss(faiss_index_path)
    else:
        index, metadata = init_faiss()

    # Embed & add to FAISS
    for i, chunk in enumerate(chunks):
        embedding = get_text_embedding(chunk)
        index.add(embedding)
        metadata.append({"content": chunk, "source": pdf_path, "chunk_id": i})

    # Save index
    save_faiss(index, metadata, faiss_index_path)
    print(f"✅ PDF {pdf_path} ingested successfully into {faiss_index_path}")
