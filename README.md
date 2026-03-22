📘RAG — Production-Grade Retrieval-Augmented Generation System

A production-ready modular Retrieval-Augmented Generation (RAG) pipeline that answers questions from documents using:

Hybrid Retrieval (FAISS + BM25)
Multi-Query Expansion
Cohere Cross-Encoder Reranking
Session-Based Conversational Memory
LangSmith Observability
Automatic Evaluation using RAGAS

Built with:

LangChain + FAISS + BM25 + Cohere + OpenAI + FastAPI + LangSmith + RAGAS

Designed following enterprise GenAI architecture principles.

🧠 Architecture Overview

The system contains two independent pipelines

1️⃣ Ingestion Pipeline
Documents → Chunking → Embeddings → FAISS Index + BM25 Corpus

Supports:

PDF
TXT
CSV

Produces:

vectorstore/faiss_index/
documents.pkl (BM25 corpus)
2️⃣ Query Pipeline
User Query
   │
   ▼
Query Understanding (clean + rewrite)
   │
   ▼
Multi-Query Expansion (LLM generates search variations)
   │
   ▼
Hybrid Retrieval
   ├── FAISS (semantic search)
   └── BM25  (keyword search)
   │
   ▼
Cohere Cross-Encoder Reranking
   │
   ▼
Context Builder
   ├── deduplication
   ├── ranking
   └── token-budget trimming
   │
   ▼
Prompt Builder
   ├── system prompt injection
   └── session memory injection
   │
   ▼
LLM Generation (OpenAI)
   │
   ▼
RAGAS Evaluation
   ├── Faithfulness
   └── Answer Relevancy
   │
   ▼
Answer + Observability + Metrics
✨ Key Features
🔎 Retrieval Improvements
Multi-Query Retrieval
Hybrid Search (FAISS + BM25)
Cohere Cross-Encoder Reranker

Improves recall + precision significantly compared to basic RAG.

💬 Conversational Memory

Supports:

Session-based memory
Chat history injection
Multi-turn reasoning

Enables chatbot-style interactions across requests.

📊 Observability (LangSmith)

Tracks:

Pipeline latency per step
Retrieval scores
Prompt tokens
Completion tokens
Execution traces

Accessible via:

https://smith.langchain.com
📈 Automatic Evaluation (RAGAS)

Pipeline automatically computes:

Faithfulness
Answer relevancy

Example:

faithfulness = 1.0
answer_relevancy = 0.68

Ensures grounded responses and reduces hallucinations.

📂 Project Structure
BookRAGProject/

config/
 └── config.py

data/
 └── source documents

vectorstore/
 └── faiss_index/
     └── documents.pkl

src/

 ├── api/
 │   └── main.py

 └── app/

     rag_ingestion/
     ├── base_loader.py
     ├── chunk_strategies.py
     ├── embedding_service.py
     ├── vectordb_factory.py
     └── ingestion_pipeline.py

     rag_pipeline/
     ├── query_understanding.py
     ├── retriever.py
     ├── bm25_retriever.py
     ├── multi_query_generator.py
     ├── reranker.py
     ├── context_builder.py
     ├── prompt_builder.py
     ├── llm_service.py
     ├── ragas_evaluator.py
     ├── observability.py
     └── rag_pipeline_orc.py

     memory/
     └── session_memory.py
⚙️ Tech Stack
Layer	Technology
Document loading	LangChain loaders
Chunking	RecursiveCharacterTextSplitter
Embeddings	OpenAI text-embedding-3-small
Vector search	FAISS
Keyword search	BM25
Hybrid retrieval	FAISS + BM25
Query expansion	LLM multi-query generator
Reranking	Cohere rerank-english-v3.0
LLM	OpenAI gpt-4o-mini
Evaluation	RAGAS
Memory	Session-based memory
API	FastAPI
Observability	LangSmith
Package manager	uv
Deployment	Docker-ready
🔐 Environment Setup

Create .env

OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=RAG_AI
LANGSMITH_TRACING_ENABLED=true
🚀 Quickstart
1️⃣ Install dependencies
uv sync
2️⃣ Add documents

Place files inside:

data/

Supported formats:

PDF
TXT
CSV
3️⃣ Run ingestion pipeline

Builds FAISS + BM25 index

uv run python src/app/rag_ingestion/ingestion_pipeline.py

Output:

vectorstore/faiss_index/
documents.pkl
4️⃣ Start API server
uv run uvicorn src.api.main:app --reload

Open Swagger UI:

http://localhost:8000/docs
📡 API Endpoints
Health Check
GET /health

Response:

{
  "status": "ok",
  "pipeline_ready": true
}
Ask Question (Stateless)
POST /ask

Example:

{
 "query": "What is transformer?"
}
Chat Endpoint (Session Memory Enabled)
POST /chat

Example:

{
 "session_id": "abc123",
 "query": "Explain transformers"
}

Supports multi-turn conversation context.

📊 Example Pipeline Metrics

Example output:

Latency
Query rewrite: 1.6s
Retrieval: 0.9s
Reranking: 0.5s
Context build: 0.04s
LLM: 2.2s
Total: 5.4s

Retrieval quality
Chunks retrieved: 5
Chunks reranked: 5

Token usage
Prompt tokens: 476
Completion tokens: 84
📈 RAGAS Evaluation Example
Faithfulness = 1.0
Answer relevancy = 0.68

Meaning:

Answer fully grounded in retrieved context
Moderately aligned with question intent
🐳 Docker Deployment

Build image

docker build -t bookrag .

Run container

docker run -p 8000:8000 bookrag
🧪 Production-Grade Capabilities Implemented

This system includes features typically found in enterprise GenAI stacks:

Hybrid retrieval (vector + keyword)
Multi-query expansion
Cross-encoder reranking
Session conversational memory
Automatic hallucination detection
Automatic answer relevance scoring
LangSmith observability
Token tracking
Latency tracking
FastAPI production server
Docker-ready deployment
👨‍💻 Author

Built as part of an advanced production-grade GenAI RAG system architecture implementation focusing on:

retrieval quality
evaluation reliability
observability
modular pipeline design
enterprise readiness