"""
prompt_builder.py — Step 5: Prompt Assembly & LLM Call

Responsibility:
    Accept rewritten query + formatted context block,
    assemble messages list,
    call LLMService,
    return generated answer.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.app.rag_pipeline.llm_service import LLMService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# System Prompt Template
# ──────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a precise, factual question-answering assistant for a book knowledge base.

You are given a set of relevant excerpts (context) retrieved from the knowledge base.
Your job is to answer the user's question using ONLY the information in the context below.

Rules:
- Answer strictly from the provided context. Do NOT use outside knowledge.
- If the answer is fully or partially present in the context, provide it clearly and concisely.
- If the context does not contain enough information to answer, respond with:
  "I could not find a sufficient answer in the provided context."
- Where possible, cite the source (e.g. "According to [Source: book.pdf | Page: 42]...").
- Do NOT speculate, hallucinate, or add information not present in the context.
- Keep your answer focused and factual.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

class PromptBuilder:
    """
    Builds LLM message payload and generates final answer.
    """

    def __init__(self, llm: LLMService):

        self.llm = llm

        logger.info("PromptBuilder initialised.")


    # ──────────────────────────────────────────
    # Build messages
    # ──────────────────────────────────────────

    def build_messages(
    self,
    query: str,
    context: str,
    chat_history: list[dict] | None = None) -> list[dict]:

        if not query or not query.strip():

            raise ValueError(
                "Query is empty — cannot build prompt."
            )

        if not context or not context.strip():

            raise ValueError(
                "Context is empty — no content to answer from."
            )


        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            context=context
        )


        messages = [

            {
                "role": "system",
                "content": system_content,
            }
        ]


        # ── Inject session chat history (NEW) ─────────────────────

        if chat_history:

            messages.extend(chat_history)


        # ── Append current query (always last user message) ───────

        messages.append(

            {
                "role": "user",
                "content": query,
            }
        )


        logger.info(
            f"Prompt built — system approx "
            f"{len(system_content)//4} tokens"
        )


        return messages


    # ──────────────────────────────────────────
    # Generate answer
    # ──────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> str:

        messages = self.build_messages(
            query,
            context,
            chat_history
        )

        logger.info(
            "Calling LLM to generate answer..."
        )

        try:

            answer = self.llm.generate(
                messages
            )

        except Exception as e:

            raise RuntimeError(
                f"LLM answer generation failed: {e}"
            ) from e


        logger.info(
            f"Answer received — {len(answer)} character(s)."
        )

        return answer


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    from query_understanding import QueryUnderstanding
    from retriever import Retriever
    from reranker import Reranker
    from context_builder import ContextBuilder


    raw_query = "what is the attention mechanism in transformers?"


    llm = LLMService()


    # Step 1 — rewrite query
    qu = QueryUnderstanding(
        llm=llm
    )

    rewritten_query = qu.rewrite(
        raw_query
    )


    # Step 2 — retrieve
    retriever = Retriever()

    chunks = retriever.retrieve(
        rewritten_query
    )


    # Step 3 — rerank
    reranker = Reranker()

    top_chunks = reranker.rerank(
        rewritten_query,
        chunks
    )


    # Step 4 — build context
    context = ContextBuilder().build(
        top_chunks
    )


    # Step 5 — generate answer
    prompt_builder = PromptBuilder(
        llm=llm
    )

    answer = prompt_builder.generate(
        query=rewritten_query,
        context=context
    )


    print("\n" + "=" * 60)

    print(f"Raw query      : {raw_query}")

    print(f"Rewritten query: {rewritten_query}")

    print(f"Chunks used    : {len(top_chunks)}")

    print("=" * 60)

    print("\nAnswer:\n")

    print(answer)

    print("=" * 60)