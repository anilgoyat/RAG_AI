"""
rag_pipeline_orc.py — RAG Query Pipeline Orchestrator
"""

import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.app.rag_pipeline.context_builder import ContextBuilder
from src.app.rag_pipeline.llm_service import LLMService
from src.app.rag_pipeline.observability import Observability
from src.app.rag_pipeline.prompt_builder import PromptBuilder, SYSTEM_PROMPT_TEMPLATE
from src.app.rag_pipeline.query_understanding import QueryUnderstanding
from src.app.rag_pipeline.reranker import Reranker
from src.app.rag_pipeline.retriever import Retriever
from src.app.memory.session_memory import SessionMemory
from src.app.rag_pipeline.multi_query_generator import MultiQueryGenerator
from src.app.rag_pipeline.bm25_retriever import BM25Retriever
from src.app.rag_pipeline.ragas_evaluator import RagasEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# RAG Pipeline Orchestrator
# ──────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):

        logger.info("Initialising RAG pipeline components...")

        self.obs = Observability()

        llm = LLMService()

        self.query_understanding = QueryUnderstanding(llm=llm)
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder(llm=llm)

        self.memory = SessionMemory()
        self.multi_query = MultiQueryGenerator(llm)
        self.bm25 = BM25Retriever()

        self.ragas = RagasEvaluator()

        logger.info("RAG pipeline ready.")


    # ──────────────────────────────────────────
    # Standard RAG Pipeline
    # ──────────────────────────────────────────

    def run(self, query: str) -> str:

        if not query.strip():
            raise ValueError("Query is empty — cannot run the pipeline.")

        pipeline_start = time.perf_counter()

        metrics = self.obs.start_run(query)

        logger.info("\n" + "=" * 60)
        logger.info(f"RAG pipeline started for query: '{query[:80]}'")
        logger.info("=" * 60)


        # Step 1 — Query Understanding

        logger.info("[Step 1/5] Query Understanding...")

        with self.obs.step_timer("query_understanding", metrics):

            rewritten_query = self.query_understanding.rewrite(query)

        metrics.rewritten_query = rewritten_query


        # Step 2 — Retrieval

        logger.info("[Step 2/5] Retrieval...")

        with self.obs.step_timer("retrieval", metrics):

            chunks = self.retriever.retrieve(rewritten_query)

        self.obs.record_retrieval_chunks(chunks, metrics)

        if not chunks:

            metrics.latency_total = time.perf_counter() - pipeline_start
            self.obs.finish_run(metrics)

            return "No relevant content found."


        # Step 3 — Reranking

        logger.info("[Step 3/5] Reranking...")

        with self.obs.step_timer("reranking", metrics):

            top_chunks = self.reranker.rerank(
                rewritten_query,
                chunks
            )

        self.obs.record_rerank_chunks(top_chunks, metrics)


        # Step 4 — Context Building

        logger.info("[Step 4/5] Building context...")

        with self.obs.step_timer("context_building", metrics):

            context = self.context_builder.build(top_chunks)

        if not context.strip():

            metrics.latency_total = time.perf_counter() - pipeline_start
            self.obs.finish_run(metrics)

            return "Context empty after processing."


        # Step 5 — LLM Generation

        logger.info("[Step 5/5] Generating answer...")

        with self.obs.step_timer("llm", metrics):

            answer = self.prompt_builder.generate(
                query=rewritten_query,
                context=context,
            )


        # ─────────────────────────────────────────
        # RAGAS Evaluation (NEW)
        # ─────────────────────────────────────────

        contexts = [doc.page_content for doc in top_chunks]

        ragas_scores = self.ragas.evaluate(
            query=rewritten_query,
            answer=answer,
            contexts=contexts
        )

        self.obs.set_quality_scores(
            metrics,
            relevancy_score=ragas_scores.get("answer_relevancy"),
        )

        faithfulness_score = ragas_scores.get("faithfulness")

        if faithfulness_score is not None:

            hallucination_rate = 1 - faithfulness_score

            self.obs.set_quality_scores(
                metrics,
                hallucination_rate=hallucination_rate,
            )


        # Token metrics

        prompt_text = (
            SYSTEM_PROMPT_TEMPLATE.format(context=context)
            + "\n"
            + rewritten_query
        )

        self.obs.record_tokens(prompt_text, answer, metrics)

        metrics.latency_total = time.perf_counter() - pipeline_start

        self.obs.finish_run(metrics)

        logger.info("Pipeline complete.")

        return answer


    # ──────────────────────────────────────────
    # Chat Pipeline (Memory + Hybrid + MultiQuery)
    # ──────────────────────────────────────────

    def run_chat(self, session_id: str, query: str) -> str:

        history = self.memory.get_history(session_id)

        rewritten_query = self.query_understanding.rewrite(query)

        expanded_queries = self.multi_query.generate(rewritten_query)

        all_chunks = []

        for q in expanded_queries:

            faiss_chunks = self.retriever.retrieve(q)
            bm25_chunks = self.bm25.retrieve(q)

            logger.info(
                f"Hybrid retrieval → FAISS={len(faiss_chunks)} BM25={len(bm25_chunks)}"
            )

            all_chunks.extend(faiss_chunks)
            all_chunks.extend(bm25_chunks)


        if not all_chunks:
            return "No relevant content found."


        top_chunks = self.reranker.rerank(
            rewritten_query,
            all_chunks
        )


        context = self.context_builder.build(top_chunks)


        answer = self.prompt_builder.generate(
            query=rewritten_query,
            context=context,
            chat_history=history,
        )


        # ─────────────────────────────────────────
        # RAGAS Evaluation for Chat Mode (NEW)
        # ─────────────────────────────────────────

        contexts = [doc.page_content for doc in top_chunks]

        ragas_scores = self.ragas.evaluate(
            query=rewritten_query,
            answer=answer,
            contexts=contexts
        )

        logger.info(f"Chat RAGAS scores: {ragas_scores}")


        # Save conversation memory

        self.memory.add_user_message(session_id, query)
        self.memory.add_ai_message(session_id, answer)

        return answer