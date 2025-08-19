import os
import requests
import numpy as np
from dotenv import load_dotenv
from utils.preprocess import preprocess_image
import io
from PIL import Image

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Set it in .env file.")

# Gemini embedding endpoint (example for "embedding-001")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

# ----------------------------
# HELPER: NORMALIZE VECTOR
# ----------------------------
def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length (required for cosine similarity in FAISS).
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# ----------------------------
# TEXT EMBEDDINGS
# ----------------------------
def get_text_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for text using Gemini Embeddings API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }

    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }

    response = requests.post(GEMINI_EMBED_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Gemini Embedding API Error: {response.status_code} - {response.text}")

    data = response.json()
    vector = np.array(data["embedding"]["values"], dtype=np.float32)
    return normalize_vector(vector).reshape(1, -1)


# ----------------------------
# IMAGE EMBEDDINGS
# ----------------------------
def get_image_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Convert image → embedding vector using Gemini multimodal embeddings.
    (Currently Gemini may not have direct embeddings for images, 
    so fallback can be CLIP or other vision encoders if needed.)
    """
    # Preprocess image
    image = preprocess_image(image_bytes, size=(224, 224))

    # Convert to byte array (JPEG)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # If Gemini supports multimodal embeddings:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }

    payload = {
        "model": "models/embedding-001",
        "content": {
            "parts": [
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_bytes.decode("latin1")  # base64 would be ideal if required
                }}
            ]
        }
    }

    response = requests.post(GEMINI_EMBED_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Gemini Image Embedding API Error: {response.status_code} - {response.text}")

    data = response.json()
    vector = np.array(data["embedding"]["values"], dtype=np.float32)
    return normalize_vector(vector).reshape(1, -1)
