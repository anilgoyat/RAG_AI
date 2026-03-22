"""
context_builder.py — Step 4: Context Assembly

Responsibility:
    Accept reranked Documents, clean them, deduplicate, enforce token budget,
    and format into final context string for LLM prompt injection.
"""

import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.config import (
    CONTEXT_CHUNK_SEPARATOR,
    CONTEXT_MAX_TOKENS,
)

from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Context Builder
# ──────────────────────────────────────────────

class ContextBuilder:

    def __init__(
        self,
        max_tokens: int = CONTEXT_MAX_TOKENS,
        separator: str = CONTEXT_CHUNK_SEPARATOR,
    ):

        self.max_tokens = max_tokens
        self.separator = separator

        logger.info(
            f"ContextBuilder initialised — max_tokens={max_tokens}"
        )


    # ──────────────────────────────────────────
    # Token estimation
    # ──────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:

        return max(0, len(text) // 4)


    # ──────────────────────────────────────────
    # Clean chunk text
    # ──────────────────────────────────────────

    def _clean(self, text: str) -> str:

        text = text.strip()

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(
            r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]",
            "",
            text
        )

        return text


    # ──────────────────────────────────────────
    # Deduplication fingerprint
    # ──────────────────────────────────────────

    def _fingerprint(self, text: str) -> str:

        normalised = re.sub(
            r"\s+",
            " ",
            text.lower()
        ).strip()

        return hashlib.sha256(
            normalised.encode("utf-8")
        ).hexdigest()


    # ──────────────────────────────────────────
    # Format chunk
    # ──────────────────────────────────────────

    def _format_chunk(
        self,
        doc: Document,
        index: int
    ) -> str:

        meta = doc.metadata

        parts = [f"Chunk {index}"]


        source = meta.get("source")

        if source:

            parts.append(
                f"Source: {Path(source).name}"
            )


        page = meta.get("page")

        if page is not None:

            parts.append(
                f"Page: {page}"
            )


        row = meta.get("row")

        if row is not None:

            parts.append(
                f"Row: {row}"
            )


        score = meta.get(
            "relevance_score"
        ) or meta.get(
            "similarity_score"
        )


        if isinstance(score, (int, float)):

            parts.append(
                f"Score: {score:.4f}"
            )


        header = "[" + " | ".join(parts) + "]"

        return f"{header}\n{doc.page_content}"


    # ──────────────────────────────────────────
    # Build final context
    # ──────────────────────────────────────────

    def build(
        self,
        chunks: list[Document]
    ) -> str:

        if chunks is None:

            raise ValueError(
                "chunks must not be None."
            )


        if not chunks:

            logger.warning(
                "ContextBuilder received empty chunk list."
            )

            return ""


        logger.info(
            f"Building context from {len(chunks)} chunk(s)..."
        )


        # Stage 1 — Clean

        for doc in chunks:

            doc.page_content = self._clean(
                doc.page_content
            )


        # Stage 2 — Deduplicate

        seen = set()

        unique_chunks = []


        for doc in chunks:

            fp = self._fingerprint(
                doc.page_content
            )

            if fp not in seen:

                seen.add(fp)

                unique_chunks.append(doc)


        logger.info(
            f"Stage 2 — Deduplicate: "
            f"{len(unique_chunks)} unique chunk(s) remain."
        )


        # Stage 3 — Order

        unique_chunks.sort(

            key=lambda d: (

                d.metadata.get(
                    "relevance_score",
                    0.0
                ),

                d.metadata.get(
                    "similarity_score",
                    0.0
                ),

            ),

            reverse=True,

        )


        logger.info(
            "Stage 3 — Order: sorted by relevance_score"
        )


        # Stage 4 — Trim token budget

        trimmed = []

        running_tokens = 0


        for doc in unique_chunks:

            tokens = self._estimate_tokens(
                doc.page_content
            )

            if running_tokens + tokens > self.max_tokens:

                break


            trimmed.append(doc)

            running_tokens += tokens


        logger.info(
            f"Stage 4 — Trim: kept {len(trimmed)} chunk(s)"
        )


        if not trimmed:

            logger.warning(
                "All chunks removed by token budget."
            )

            return ""


        # Stage 5 — Format

        formatted_blocks = [

            self._format_chunk(doc, i + 1)

            for i, doc in enumerate(trimmed)

        ]


        context = self.separator.join(
            formatted_blocks
        )


        logger.info(
            f"Stage 5 — Format complete "
            f"(~{self._estimate_tokens(context)} tokens)"
        )


        return context


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    from llm_service import LLMService
    from query_understanding import QueryUnderstanding
    from retriever import Retriever
    from reranker import Reranker


    raw_query = "what is the attention mechanism in transformers?"


    qu = QueryUnderstanding(
        llm=LLMService()
    )


    rewritten_query = qu.rewrite(
        raw_query
    )


    retriever = Retriever()

    chunks = retriever.retrieve(
        rewritten_query
    )


    reranker = Reranker()

    top_chunks = reranker.rerank(
        rewritten_query,
        chunks
    )


    builder = ContextBuilder()

    context = builder.build(
        top_chunks
    )


    print("\n" + "=" * 60)

    print(f"Query          : {raw_query}")

    print(f"Chunks in      : {len(top_chunks)}")

    print(f"Context tokens : ~{len(context)//4}")

    print("=" * 60)

    print("\n--- CONTEXT BLOCK ---\n")

    print(context)

    print("\n--- END CONTEXT ---")