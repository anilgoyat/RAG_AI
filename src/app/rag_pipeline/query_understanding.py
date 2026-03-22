"""
query_understanding.py — Step 1: Query Cleaning & Rewriting

Responsibility:
    Accept a raw user query, clean it, and rewrite it for better retrieval.

Two functions:
    1. clean(query)   — strips extra whitespace, normalizes spaces, lowercases
    2. rewrite(query) — uses the LLM (via LLMService) to expand/rephrase the
                        query into a retrieval-optimized version

Input  : raw query string
Output : cleaned query string
         rewritten query string (LLM output)

Design:
    QueryUnderstanding
    ├── clean(query: str)   → str
    └── rewrite(query: str) → str   (calls LLMService internally)
"""

import logging
import re
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.app.rag_pipeline.llm_service import LLMService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = (
    "You are a search query reviewer for a retrieval-augmented generation system. "
    "Your job is to evaluate whether the user's query needs improvement. "
    "Rules:\n"
    "- If the query is already clear, specific, and self-contained → return it EXACTLY as given, word for word.\n"
    "- Only rewrite if the query is vague, has typos, is too short to be meaningful, or is ambiguous.\n"
    "- When you do rewrite, make minimal targeted changes — do NOT expand, add keywords, or make it longer.\n"
    "- ALWAYS preserve the original punctuation at the end of the query (e.g. ?, !, .).\n"
    "- Do NOT add, remove, or change any punctuation unless fixing a typo.\n"
    "- Return ONLY the final query string — no explanation, no preamble, no quotes, no bullet points."
)


# ──────────────────────────────────────────────
# Query Understanding Class
# ──────────────────────────────────────────────

class QueryUnderstanding:
    """
    Cleans and rewrites a raw user query before retrieval.

    clean()   → rule-based normalization
    rewrite() → LLM-based rewriting
    """

    def __init__(self, llm: LLMService):
        """
        Args:
            llm (LLMService): Injected LLM service instance.
        """
        self.llm = llm

        logger.info("QueryUnderstanding initialized.")


    # ──────────────────────────────────────────
    # Clean Query
    # ──────────────────────────────────────────

    def clean(self, query: str) -> str:
        """
        Normalize whitespace + lowercase query.
        """

        logger.info(
            f"Cleaning query: '{query[:80]}{'...' if len(query) > 80 else ''}'"
        )

        cleaned = re.sub(r"\s+", " ", query.strip()).lower()

        if not cleaned:
            raise ValueError("Query is empty after cleaning.")

        logger.info(f"Cleaned query : '{cleaned}'")

        return cleaned


    # ──────────────────────────────────────────
    # Rewrite Query via LLM
    # ──────────────────────────────────────────

    def rewrite(self, query: str) -> str:
        """
        Rewrite query using OpenAI LLM via LLMService.
        """

        cleaned = self.clean(query)

        logger.info("Rewriting query via LLM...")

        messages = [
            {
                "role": "system",
                "content": REWRITE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": cleaned
            }
        ]

        try:

            rewritten = self.llm.generate(messages).strip()

        except Exception as e:

            raise RuntimeError(
                f"LLM rewrite failed: {e}"
            ) from e

        logger.info(f"Rewritten query: '{rewritten}'")

        return rewritten


# ──────────────────────────────────────────────
# Quick Smoke Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    raw_query = "  what   is   attention   mechanism    in   transformers?  "

    qu = QueryUnderstanding(
        llm=LLMService()
    )

    cleaned = qu.clean(raw_query)

    print(f"\n{'='*50}")
    print(f"Raw query     : '{raw_query}'")
    print(f"Cleaned query : '{cleaned}'")

    rewritten = qu.rewrite(raw_query)

    print(f"Rewritten     : '{rewritten}'")
    print(f"{'='*50}")