import os
from dotenv import load_dotenv

# ----------------------------
# LOAD ENV VARIABLES
# ----------------------------
load_dotenv()

# ----------------------------
# API KEYS
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env file.")

# ----------------------------
# FAISS SETTINGS
# ----------------------------
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/index.faiss")

# ----------------------------
# EMBEDDING SETTINGS
# ----------------------------
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # Gemini embedding-001 default dim

# ----------------------------
# RAG SETTINGS
# ----------------------------
TOP_K = int(os.getenv("TOP_K", "5"))  # default number of retrieved docs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ----------------------------
# APP SETTINGS
# ----------------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
