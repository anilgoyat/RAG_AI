"""
retriever.py — Step 2: FAISS Retrieval

Responsibility:
    Load the persisted FAISS index, run a similarity search against the
    rewritten query, convert L2 distances to [0, 1] similarity scores,
    and return only high-confidence chunks above the score threshold.

Input  : query (str)
Output : list[Document]

Similarity conversion:
    similarity = 1 / (1 + l2_distance)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.config import (
    VECTOR_DB_PERSIST_PATH,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    RETRIEVER_SCORE_THRESHOLD,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────

class Retriever:
    """
    Loads FAISS index and retrieves relevant chunks.

    Converts L2 distance → similarity score.

    Returns only chunks above similarity threshold.
    """

    def __init__(
        self,
        index_path: str = VECTOR_DB_PERSIST_PATH,
        top_k: int = RETRIEVER_TOP_K,
        score_threshold: float = RETRIEVER_SCORE_THRESHOLD,
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        """
        Load FAISS index and embedding model.
        """

        self.top_k = top_k
        self.score_threshold = score_threshold
        self.index_path = index_path

        logger.info(
            f"Loading embedding model '{embedding_model_name}' for retrieval..."
        )

        self._embedding_model = OpenAIEmbeddings(
            model=embedding_model_name
        )

        logger.info(f"Loading FAISS index from '{index_path}'...")

        try:

            self._index = FAISS.load_local(
                index_path,
                self._embedding_model,
                allow_dangerous_deserialization=True,
            )

            logger.info("FAISS index loaded successfully.")

        except Exception as e:

            raise RuntimeError(
                f"Failed to load FAISS index from '{index_path}': {e}\n"
                "Run ingestion pipeline first."
            ) from e


    # ──────────────────────────────────────────
    # L2 → Similarity conversion
    # ──────────────────────────────────────────

    def _l2_to_similarity(self, l2_distance: float) -> float:

        return 1.0 / (1.0 + l2_distance)


    # ──────────────────────────────────────────
    # Retrieve
    # ──────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:

        if not query.strip():

            raise ValueError("Query is empty — cannot retrieve.")

        logger.info(
            f"Retrieving top {self.top_k} chunks for query: "
            f"'{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        try:

            results: list[tuple[Document, float]] = (
                self._index.similarity_search_with_score(
                    query,
                    k=self.top_k
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"FAISS search failed: {e}"
            ) from e


        logger.info(
            f"FAISS returned {len(results)} candidate(s). Applying score filter..."
        )

        scored_chunks: list[Document] = []


        for doc, l2_distance in results:

            try:

                similarity = self._l2_to_similarity(
                    l2_distance
                )

            except Exception as e:

                logger.warning(
                    f"Score conversion failed for chunk "
                    f"(source={doc.metadata.get('source', '?')}): {e}. "
                    f"Using raw L2={l2_distance:.4f} as score."
                )

                similarity = l2_distance


            doc.metadata["similarity_score"] = round(
                similarity,
                4
            )

            doc.metadata["l2_distance"] = round(
                l2_distance,
                4
            )


            if similarity >= self.score_threshold:

                scored_chunks.append(doc)

            else:

                logger.debug(
                    f"Chunk filtered out — score={similarity:.4f} < "
                    f"threshold={self.score_threshold}"
                )


        scored_chunks.sort(
            key=lambda d: d.metadata["similarity_score"],
            reverse=True
        )


        logger.info(
            f"Retrieval complete. "
            f"{len(scored_chunks)}/{len(results)} chunk(s) passed threshold."
        )

        return scored_chunks


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    query = "what is the attention mechanism in transformers?"

    retriever = Retriever()

    chunks = retriever.retrieve(query)

    print(f"\n{'='*55}")
    print(f"Query   : {query}")
    print(f"Chunks returned : {len(chunks)}")
    print(f"{'='*55}")

    for i, doc in enumerate(chunks):

        print(f"\n[Chunk {i+1}]")

        print(
            f"  Score     : {doc.metadata.get('similarity_score')}"
        )

        print(
            f"  L2 dist   : {doc.metadata.get('l2_distance')}"
        )

        print(
            f"  Source    : {doc.metadata.get('source', 'N/A')}"
        )

        print(
            f"  Page/Row  : "
            f"{doc.metadata.get('page', doc.metadata.get('row', 'N/A'))}"
        )

        print(
            f"  Content   : "
            f"{doc.page_content[:120].strip()}..."
        )