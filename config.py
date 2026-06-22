"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# Gemini OpenAI-compatible endpoint can be used through the OpenAI SDK.
LLM_API_KEY = OPENAI_API_KEY or GOOGLE_API_KEY or GEMINI_API_KEY
LLM_BASE_URL = OPENAI_BASE_URL or (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
    if (GOOGLE_API_KEY or GEMINI_API_KEY)
    else ""
)
ANSWER_MODEL = os.getenv(
    "ANSWER_MODEL",
    "gemini-2.5-flash-lite" if LLM_BASE_URL else "gpt-4o-mini",
)
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", ANSWER_MODEL)


def create_llm_client():
    from openai import OpenAI

    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
