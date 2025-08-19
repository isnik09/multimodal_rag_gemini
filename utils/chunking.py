from typing import List

# ----------------------------
# BASIC FIXED-SIZE CHUNKING
# ----------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks (word-based).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------------------
# CHARACTER-BASED CHUNKING
# ----------------------------

def chunk_by_chars(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks (character-based).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------------------
# PARAGRAPH-BASED CHUNKING
# ----------------------------

def chunk_by_paragraphs(
    text: str,
    max_paragraphs: int = 3
) -> List[str]:
    """
    Split text into chunks by paragraphs.
    Useful for PDFs or documents with natural breaks.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = "\n".join(paragraphs[i:i+max_paragraphs])
        chunks.append(chunk)
    return chunks
