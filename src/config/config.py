"""
config.py — Runtime configuration loader

Loads environment variables from .env and exposes
configuration values used across the RAG pipeline.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ── Data ──────────────────────────────────────────────────────
DATA_DIR = "data"


# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536
EMBEDDING_BATCH_SIZE = 32


# ── Vector Store ──────────────────────────────────────────────
VECTOR_DB_TYPE = "faiss"
VECTOR_DB_PERSIST_PATH = "vectorstore/faiss_index"
VECTOR_DB_BATCH_SIZE = 500


# ── Retriever ─────────────────────────────────────────────────
RETRIEVER_TOP_K = 10
RETRIEVER_SCORE_THRESHOLD = 0.2


# ── Reranker (Cohere) ─────────────────────────────────────────
RERANKER_TOP_N = 5
RERANKER_MODEL = "rerank-english-v3.0"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# ── Context Builder ───────────────────────────────────────────
CONTEXT_MAX_TOKENS = 2000
CONTEXT_CHUNK_SEPARATOR = "\n\n---\n\n"


# ── API (FastAPI / Uvicorn) ───────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000


# ── Observability (LangSmith) ─────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "RAG_AI"
LANGSMITH_TRACING_ENABLED = True


# ── LLM (OpenAI) ──────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
