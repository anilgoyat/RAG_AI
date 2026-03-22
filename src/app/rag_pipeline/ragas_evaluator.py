"""
ragas_evaluator.py
Production-safe evaluator compatible with latest ragas + langchain_openai
"""

import logging
from typing import Dict, Any

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


logger = logging.getLogger(__name__)


class RagasEvaluator:

    def __init__(self):

        # IMPORTANT:
        # Do NOT pass api_key here
        # langchain_openai automatically reads OPENAI_API_KEY from .env

        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )
        )

        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
            )
        )


    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> Dict[str, Any]:

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        logger.info("Running RAGAS evaluation...")

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )

        # latest ragas returns Executor (lazy execution)
        if callable(result):
            result = result()

        scores = result.scores # type: ignore

        logger.info(f"RAGAS scores: {scores}")

        return scores # type: ignore