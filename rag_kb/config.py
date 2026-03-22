import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORE_DIR = DATA_DIR / "vector_store"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

HF_ENDPOINT = (os.getenv("HF_ENDPOINT") or "").strip().rstrip("/") or None

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.25"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)
