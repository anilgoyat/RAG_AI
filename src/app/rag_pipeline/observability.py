"""
observability.py — Step 7: Pipeline Observability & Tracing

Tracks:
- step latency
- token usage
- retrieval quality metrics
- reranker effectiveness
- LangSmith tracing

Works automatically with:
- OpenAI LLM
- HuggingFace embeddings
- FAISS
- Cohere reranker
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING_ENABLED,
)

from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Pipeline Metrics Dataclass
# ============================================================

@dataclass
class PipelineMetrics:

    query: str

    rewritten_query: str = ""

    latency_query_understanding: float = 0.0
    latency_retrieval: float = 0.0
    latency_reranking: float = 0.0
    latency_context_building: float = 0.0
    latency_llm: float = 0.0
    latency_total: float = 0.0

    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    avg_retrieval_score: float = 0.0
    avg_rerank_score: float = 0.0

    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_total_tokens: int = 0

    relevancy_score: float | None = None
    hallucination_rate: float | None = None

    answer_length: int = 0


    # --------------------------------------------------------
    # Convert metrics → dict
    # --------------------------------------------------------

    def to_dict(self) -> dict:

        return {

            "query": self.query,
            "rewritten_query": self.rewritten_query,

            "latency_query_understanding_s":
                round(self.latency_query_understanding, 3),

            "latency_retrieval_s":
                round(self.latency_retrieval, 3),

            "latency_reranking_s":
                round(self.latency_reranking, 3),

            "latency_context_building_s":
                round(self.latency_context_building, 3),

            "latency_llm_s":
                round(self.latency_llm, 3),

            "latency_total_s":
                round(self.latency_total, 3),

            "chunks_retrieved":
                self.chunks_retrieved,

            "chunks_after_rerank":
                self.chunks_after_rerank,

            "avg_retrieval_score":
                round(self.avg_retrieval_score, 4),

            "avg_rerank_score":
                round(self.avg_rerank_score, 4),

            "estimated_prompt_tokens":
                self.estimated_prompt_tokens,

            "estimated_completion_tokens":
                self.estimated_completion_tokens,

            "estimated_total_tokens":
                self.estimated_total_tokens,

            "relevancy_score":
                self.relevancy_score,

            "hallucination_rate":
                self.hallucination_rate,

            "answer_length":
                self.answer_length,
        }


# ============================================================
# Observability Controller
# ============================================================

class Observability:

    STEP_LATENCY_MAP = {

        "query_understanding":
            "latency_query_understanding",

        "retrieval":
            "latency_retrieval",

        "reranking":
            "latency_reranking",

        "context_building":
            "latency_context_building",

        "llm":
            "latency_llm",
    }


    # --------------------------------------------------------
    # Init LangSmith tracing
    # --------------------------------------------------------

    def __init__(self):

        self.project = LANGSMITH_PROJECT

        self.tracing_enabled = (
            LANGSMITH_TRACING_ENABLED
            and bool(LANGSMITH_API_KEY)
        )

        if self.tracing_enabled:

            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if LANGSMITH_API_KEY:
                os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
            else:
                raise ValueError("LANGSMITH_API_KEY is not set")
            os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

            logger.info(
                f"LangSmith tracing enabled (project='{LANGSMITH_PROJECT}')"
            )

        else:

            logger.warning(
                "LangSmith tracing disabled "
                "(missing key or disabled in config)."
            )


    # --------------------------------------------------------
    # Start Run
    # --------------------------------------------------------

    def start_run(self, query: str) -> PipelineMetrics:

        logger.info(
            f"Observability run started for query: '{query[:60]}'"
        )

        return PipelineMetrics(query=query)


    # --------------------------------------------------------
    # Finish Run
    # --------------------------------------------------------

    def finish_run(self, metrics: PipelineMetrics):

        logger.info(

            "\n"
            "================ PIPELINE SUMMARY ================\n"

            f"Query: {metrics.query[:70]}\n"

            "\nLatency\n"
            f"Query rewrite: {metrics.latency_query_understanding:.3f}s\n"
            f"Retrieval: {metrics.latency_retrieval:.3f}s\n"
            f"Reranking: {metrics.latency_reranking:.3f}s\n"
            f"Context build: {metrics.latency_context_building:.3f}s\n"
            f"LLM: {metrics.latency_llm:.3f}s\n"
            f"Total: {metrics.latency_total:.3f}s\n"

            "\nRetrieval quality\n"
            f"Chunks retrieved: {metrics.chunks_retrieved}\n"
            f"Chunks reranked: {metrics.chunks_after_rerank}\n"
            f"Avg similarity score: {metrics.avg_retrieval_score:.4f}\n"
            f"Avg rerank score: {metrics.avg_rerank_score:.4f}\n"

            "\nToken usage (estimated)\n"
            f"Prompt tokens: {metrics.estimated_prompt_tokens}\n"
            f"Completion tokens: {metrics.estimated_completion_tokens}\n"
            f"Total tokens: {metrics.estimated_total_tokens}\n"

            "\nQuality metrics\n"
            f"Relevancy score: {metrics.relevancy_score}\n"
            f"Hallucination rate: {metrics.hallucination_rate}\n"

            "\nAnswer length\n"
            f"{metrics.answer_length} characters\n"

            "=================================================="
        )


    # --------------------------------------------------------
    # Step Timer
    # --------------------------------------------------------

    @contextmanager
    def step_timer(
        self,
        step_name: str,
        metrics: PipelineMetrics
    ) -> Generator:

        if step_name not in self.STEP_LATENCY_MAP:

            raise ValueError(
                f"Invalid step '{step_name}'"
            )

        start = time.perf_counter()

        yield

        elapsed = time.perf_counter() - start

        setattr(
            metrics,
            self.STEP_LATENCY_MAP[step_name],
            elapsed
        )


    # --------------------------------------------------------
    # Retrieval Metrics
    # --------------------------------------------------------

    def record_retrieval_chunks(
        self,
        chunks: list[Document],
        metrics: PipelineMetrics
    ):

        metrics.chunks_retrieved = len(chunks)

        if chunks:

            scores = [

                c.metadata.get(
                    "similarity_score",
                    0.0
                )

                for c in chunks
            ]

            metrics.avg_retrieval_score = sum(scores) / len(scores)


    # --------------------------------------------------------
    # Reranker Metrics
    # --------------------------------------------------------

    def record_rerank_chunks(
        self,
        chunks: list[Document],
        metrics: PipelineMetrics
    ):

        metrics.chunks_after_rerank = len(chunks)

        if chunks:

            scores = [

                c.metadata.get(
                    "relevance_score",
                    0.0
                )

                for c in chunks
            ]

            metrics.avg_rerank_score = sum(scores) / len(scores)


    # --------------------------------------------------------
    # Token Metrics
    # --------------------------------------------------------

    def record_tokens(
        self,
        prompt_text: str,
        answer_text: str,
        metrics: PipelineMetrics
    ):

        prompt_tokens = len(prompt_text) // 4

        completion_tokens = len(answer_text) // 4

        metrics.estimated_prompt_tokens = prompt_tokens

        metrics.estimated_completion_tokens = completion_tokens

        metrics.estimated_total_tokens = (
            prompt_tokens + completion_tokens
        )

        metrics.answer_length = len(answer_text)


    # --------------------------------------------------------
    # External Quality Metrics
    # --------------------------------------------------------

    def set_quality_scores(
        self,
        metrics: PipelineMetrics,
        relevancy_score: float | None = None,
        hallucination_rate: float | None = None
    ):

        if relevancy_score is not None:

            if not 0 <= relevancy_score <= 1:

                raise ValueError(
                    "relevancy_score must be between 0 and 1"
                )

            metrics.relevancy_score = relevancy_score


        if hallucination_rate is not None:

            if not 0 <= hallucination_rate <= 1:

                raise ValueError(
                    "hallucination_rate must be between 0 and 1"
                )

            metrics.hallucination_rate = hallucination_rate