"""
vectordb_factory.py — Step 4: Vector Storage

Responsibility:
    Accept chunk Documents and their corresponding embeddings, write them
    into a FAISS index in batches, and persist the index to local disk.

Input:
    chunks     : list[Document]       — chunks from chunk_strategies.py
    embeddings : list[list[float]]    — vectors from embedding_service.py

Output:
    FAISS index saved to disk under:
        <project_root>/vectorstore/faiss_index/

    Two files are written by FAISS:
        faiss_index/index.faiss   — the binary vector index
        faiss_index/index.pkl     — the docstore (metadata + page_content)

Design:
    VectorStore         — abstract base class enforcing store() + persist()
    FAISSVectorStore    — concrete FAISS implementation with batch writes
    VectorDBFactory     — registry-based factory; swap backends via config
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from src.config.config import VECTOR_DB_PERSIST_PATH, VECTOR_DB_BATCH_SIZE

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class VectorStore(ABC):
    """
    Abstract base class for all vector store backends.

    Enforces a two-method contract on every concrete backend:
        store()   — write chunks + embeddings into the index
        persist() — flush the index to durable storage
    """

    @abstractmethod
    def store(
        self,
        chunks: list[Document],
        embeddings: list[list[float]]
    ) -> None:
        """
        Write chunk Documents and their embeddings into the vector index.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persist the vector index to durable storage.
        """
        pass


# ──────────────────────────────────────────────
# Concrete FAISS Implementation
# ──────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """
    FAISS-backed vector store that writes embeddings in batches and saves
    the index to local disk via LangChain's FAISS.save_local().
    """

    def __init__(
        self,
        persist_path: str = VECTOR_DB_PERSIST_PATH,
        embedding_model: OpenAIEmbeddings | None = None,
        batch_size: int = VECTOR_DB_BATCH_SIZE,
    ):
        """
        Args:
            persist_path (str): Directory path for saving the index.
            embedding_model (OpenAIEmbeddings): Same embedding model used
                during ingestion (required for retrieval later).
            batch_size (int): Chunk/vector pairs per write batch.
        """

        if embedding_model is None:
            raise ValueError(
                "embedding_model must be provided to FAISSVectorStore"
            )

        self.persist_path = persist_path
        self.batch_size = batch_size
        self._embedding_model = embedding_model
        self._index: FAISS | None = None

        Path(persist_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FAISSVectorStore initialized. "
            f"persist_path='{persist_path}', batch_size={batch_size}"
        )


    def store(
        self,
        chunks: list[Document],
        embeddings: list[list[float]]
    ) -> None:
        """
        Write all chunks and embeddings into the FAISS index in batches.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        logger.info(
            f"Storing {len(chunks)} chunk(s) into FAISS "
            f"in batches of {self.batch_size}..."
        )

        total_batches = (
            len(chunks) + self.batch_size - 1
        ) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):

            batch_chunks = chunks[i: i + self.batch_size]
            batch_embeddings = embeddings[i: i + self.batch_size]

            batch_num = i // self.batch_size + 1

            text_embedding_pairs = [
                (chunk.page_content, embedding)
                for chunk, embedding in zip(batch_chunks, batch_embeddings)
            ]

            metadatas = [
                chunk.metadata
                for chunk in batch_chunks
            ]

            if self._index is None:

                self._index = FAISS.from_embeddings(
                    text_embeddings=text_embedding_pairs,
                    embedding=self._embedding_model,
                    metadatas=metadatas,
                )

                logger.info(
                    f"Batch {batch_num}/{total_batches} — index created."
                )

            else:

                self._index.add_embeddings(
                    text_embeddings=text_embedding_pairs,
                    metadatas=metadatas,
                )

                logger.info(
                    f"Batch {batch_num}/{total_batches} — merged into index."
                )

        self.persist()


    def persist(self) -> None:
        """
        Save the FAISS index to disk.
        """

        if self._index is None:
            raise RuntimeError(
                "Cannot persist: no data has been stored yet."
            )

        self._index.save_local(self.persist_path)

        logger.info(
            f"FAISS index persisted to '{self.persist_path}' "
            f"(index.faiss + index.pkl)"
        )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class VectorDBFactory:
    """
    Registry-based factory for creating vector store backends.
    """

    def __init__(self):

        self._registry: dict[str, type[VectorStore]] = {}

        self.register("faiss", FAISSVectorStore)


    def register(
        self,
        db_type: str,
        store_class: type[VectorStore]
    ) -> None:

        self._registry[db_type.lower()] = store_class

        logger.debug(
            f"Registered vector store: "
            f"'{db_type}' → {store_class.__name__}"
        )


    def create(
        self,
        db_type: str,
        **kwargs
    ) -> VectorStore:

        store_class = self._registry.get(db_type.lower())

        if store_class is None:

            raise ValueError(
                f"Unknown vector DB type: '{db_type}'. "
                f"Available: {list(self._registry.keys())}"
            )

        logger.info(
            f"Creating vector store: '{db_type}'"
        )

        return store_class(**kwargs)