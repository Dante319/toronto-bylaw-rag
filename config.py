# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

# Anthropic
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
HYDE_MODEL = "claude-haiku-4-5-20251001"
GENERATION_MODEL = "claude-sonnet-4-6"

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # None in local dev
COLLECTION_NAME = "toronto_bylaws"

# Embedding
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIMENSION = 1024
EMBED_BATCH_SIZE = 32
EMBED_DIMENSION = 1024 if "large" in EMBED_MODEL else 384

# Retrieval
TOP_K_DENSE = 10
TOP_K_RERANK = 3

# Chunking
CHUNK_SIZE = 512        # tokens (approximate)
CHUNK_OVERLAP = 64

# Domains
DOMAINS = ["short_term_rental", "noise", "residential_tenancy"]

# Minimum similarity to confidently assign a single domain
# Below this → search all domains
DOMAIN_DETECTION_THRESHOLD = 0.75

DOMAIN_DESCRIPTIONS = {
    "short_term_rental": (
        "Toronto short-term rental regulations, Airbnb licensing, "
        "operator registration, rental company permits, hosting rules, "
        "short-term rental by-law Chapter 547"
    ),
    "noise": (
        "Toronto noise by-laws, quiet hours, construction noise, "
        "amplified sound, disturbance complaints, noise permits, "
        "noise by-law Chapter 591"
    ),
}