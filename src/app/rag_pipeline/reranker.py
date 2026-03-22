"""
reranker.py — Step 3: Cohere Reranking

Responsibility:
    Accept candidate chunks returned by retriever.py, send them
    with the query to the Cohere Rerank API, and return the top-N chunks
    sorted by relevance_score (highest first).
"""

import logging
import sys
from pathlib import Path
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.config import (
    COHERE_API_KEY,
    RERANKER_MODEL,
    RERANKER_TOP_N,
)

from langchain_cohere import CohereRerank
from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker using Cohere Rerank API.

    Produces higher-quality semantic ranking than embedding similarity alone.
    """

    def __init__(
        self,
        top_n: int = RERANKER_TOP_N,
        model: str = RERANKER_MODEL,
    ):

        self.top_n = top_n
        self.model = model

        logger.info(
            f"Initialising Cohere reranker "
            f"(model={model}, top_n={top_n})"
        )

        try:
            api_key: str | None = COHERE_API_KEY

            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY missing. Set it inside config.py or .env"
                )
            self._reranker = CohereRerank(
                cohere_api_key=SecretStr(api_key),
                model=model,
                top_n=top_n,
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed to initialise CohereRerank: {e}"
            ) from e

        logger.info("Cohere reranker initialised successfully.")


    # ──────────────────────────────────────────
    # Rerank
    # ──────────────────────────────────────────

    def rerank(
        self,
        query: str,
        chunks: list[Document]
    ) -> list[Document]:

        if not query.strip():

            raise ValueError(
                "Query is empty — cannot rerank."
            )

        if not chunks:

            logger.warning(
                "No chunks provided to reranker."
            )

            return []

        logger.info(
            f"Reranking {len(chunks)} chunk(s) "
            f"→ returning top {self.top_n}"
        )

        try:

            reranked: list[Document] = list(
                self._reranker.compress_documents(
                    chunks,
                    query
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"Cohere reranking failed: {e}"
            ) from e


        reranked.sort(
            key=lambda d: d.metadata.get(
                "relevance_score",
                0.0
            ),
            reverse=True,
        )


        logger.info(
            f"Reranking complete. "
            f"Returned {len(reranked)} chunk(s)."
        )

        return reranked


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    from llm_service import LLMService
    from query_understanding import QueryUnderstanding
    from retriever import Retriever


    raw_query = "what is the attention mechanism in transformers?"


    # Step 1 — rewrite query
    qu = QueryUnderstanding(
        llm=LLMService()
    )

    rewritten_query = qu.rewrite(
        raw_query
    )


    # Step 2 — retrieve
    retriever = Retriever()

    chunks = retriever.retrieve(
        rewritten_query
    )


    print("\n" + "=" * 60)

    print(f"Raw query      : {raw_query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Chunks retrieved: {len(chunks)}")

    print("=" * 60)


    # Step 3 — rerank
    reranker = Reranker()

    top_chunks = reranker.rerank(
        rewritten_query,
        chunks
    )


    print(f"\nTop {len(top_chunks)} chunk(s) after reranking:")

    print("=" * 60)


    for i, doc in enumerate(top_chunks):

        print(f"\n[Chunk {i+1}]")

        print(
            f"  Relevance score : "
            f"{doc.metadata.get('relevance_score')}"
        )

        print(
            f"  Similarity score: "
            f"{doc.metadata.get('similarity_score')}"
        )

        print(
            f"  Source          : "
            f"{doc.metadata.get('source')}"
        )

        print(
            f"  Page/Row        : "
            f"{doc.metadata.get('page', doc.metadata.get('row'))}"
        )

        print(
            f"  Content         : "
            f"{doc.page_content[:120]}..."
        )