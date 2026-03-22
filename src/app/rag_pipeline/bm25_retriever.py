"""
bm25_retriever.py

Keyword-based retrieval using BM25 ranking.

Used alongside FAISS semantic retrieval
to implement hybrid search architecture.
"""

import logging
import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class BM25Retriever:

    def __init__(
        self,
        persist_path: str = "vectorstore/faiss_index/documents.pkl",
        top_k: int = 5,
    ):

        self.persist_path = Path(persist_path)
        self.top_k = top_k

        if not self.persist_path.exists():

            raise RuntimeError(
                f"BM25 corpus file missing: {persist_path}"
            )

        logger.info(
            f"Loading BM25 corpus from '{persist_path}'..."
        )

        with open(self.persist_path, "rb") as f:

            self.documents: List[Document] = pickle.load(f)

        tokenized_corpus = [

            doc.page_content.split()

            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            f"BM25 index ready with {len(self.documents)} chunks."
        )


    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_pairs = sorted(

            zip(self.documents, scores),

            key=lambda x: x[1],

            reverse=True,
        )

        top_docs = [

            doc

            for doc, score in ranked_pairs[: self.top_k]
        ]

        logger.info(
            f"BM25 retrieved {len(top_docs)} chunks."
        )

        return top_docs