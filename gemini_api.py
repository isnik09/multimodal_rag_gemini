import os
import base64
import requests
from dotenv import load_dotenv
from typing import Optional

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")

# Base endpoint (example for Gemini-pro API)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


# ----------------------------
# HELPER: ENCODE IMAGE
# ----------------------------
def encode_image(image_path: str) -> str:
    """
    Convert image file → base64 string for Gemini.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------------
# MAIN GEMINI QUERY
# ----------------------------
def query_gemini(
    text: str,
    image_path: Optional[str] = None,
    context: Optional[str] = None,
    temperature: float = 0.3
) -> str:
    """
    Query Gemini with text + optional image + context.
    Returns generated response as string.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

    # Build request
    parts = []
    if context:
        parts.append({"text": f"Context:\n{context}"})
    if text:
        parts.append({"text": text})
    if image_path:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encode_image(image_path)
            }
        })

    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
    }

    response = requests.post(
        GEMINI_API_URL,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
