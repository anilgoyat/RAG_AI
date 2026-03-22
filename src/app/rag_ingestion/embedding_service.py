"""
embedding_service.py — Step 3: Embedding

Responsibility:
    Convert chunk Documents into float vectors using OpenAI embeddings.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from src.config.config import EMBEDDING_MODEL, VECTOR_DIMENSION, EMBEDDING_BATCH_SIZE

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Embedding Service
# ──────────────────────────────────────────────

class EmbeddingService:
    """
    Converts chunk Documents into float vectors using OpenAI embeddings.

    Attributes:
        model_name (str)
        batch_size (int)
        vector_dim (int)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.vector_dim = VECTOR_DIMENSION

        logger.info(f"Loading OpenAI embedding model: '{model_name}' ...")

        self._model = OpenAIEmbeddings(
            model=model_name,
        )

        logger.info(
            f"Embedding model loaded. "
            f"vector_dim={self.vector_dim}, batch_size={self.batch_size}"
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed batch of text strings.
        """
        return self._model.embed_documents(texts)

    def embed(self, chunks: list[Document]) -> list[list[float]]:
        """
        Embed chunk Documents → vectors
        """

        if not chunks:
            raise ValueError("No chunks provided to embed.")

        logger.info(
            f"Embedding {len(chunks)} chunk(s) "
            f"in batches of {self.batch_size}..."
        )

        texts = [chunk.page_content for chunk in chunks]
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Batch {batch_num}/{total_batches} "
                f"— {len(batch)} chunk(s)"
            )

            batch_vectors = self._embed_batch(batch)
            embeddings.extend(batch_vectors)

        assert len(embeddings) == len(chunks), (
            f"Embedding count mismatch: got {len(embeddings)} vectors "
            f"for {len(chunks)} chunks."
        )

        logger.info(
            f"Embedding complete. "
            f"Total vectors: {len(embeddings)}, dimension: {self.vector_dim}"
        )

        return embeddings