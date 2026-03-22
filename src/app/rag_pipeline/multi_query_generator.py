"""
multi_query_generator.py

Generates multiple retrieval-friendly variants of a user query
using an LLM to improve recall during vector search.
"""

import logging

from src.app.rag_pipeline.llm_service import LLMService


logger = logging.getLogger(__name__)


MULTI_QUERY_PROMPT = """
You are an expert search assistant.

Rewrite the user's question into 3 alternative versions
that improve document retrieval quality.

Rules:
- Keep meaning identical
- Use different phrasing
- Focus on technical clarity
- Return ONLY the rewritten queries
- One per line
"""


class MultiQueryGenerator:

    def __init__(self, llm: LLMService):

        self.llm = llm

        logger.info("MultiQueryGenerator initialized.")


    def generate(self, query: str) -> list[str]:

        messages = [

            {
                "role": "system",
                "content": MULTI_QUERY_PROMPT,
            },

            {
                "role": "user",
                "content": query,
            },
        ]


        response = self.llm.generate(messages)


        queries = [

            q.strip()

            for q in response.split("\n")

            if q.strip()
        ]


        queries.append(query)


        logger.info(
            f"Generated {len(queries)} retrieval queries."
        )


        return queries