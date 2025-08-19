import re
from typing import List
from PIL import Image
import io

# ----------------------------
# TEXT PREPROCESSING
# ----------------------------

def clean_text(text: str) -> str:
    """
    Normalize and clean input text:
    - Remove extra spaces, line breaks
    - Lowercase (optional)
    - Remove non-printable characters
    """
    if not text:
        return ""
    # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces/newlines
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # strip non-ASCII
    return text.strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    """
    text = clean_text(text)
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
# IMAGE PREPROCESSING
# ----------------------------

def preprocess_image(image_bytes: bytes, size: tuple = (224, 224)) -> Image.Image:
    """
    Preprocess uploaded image:
    - Convert to RGB
    - Resize
    - Return PIL Image object
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(size)
    return img
