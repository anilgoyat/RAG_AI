"""
main.py — BookRAG FastAPI Application

Exposes the RAG pipeline over HTTP:

    GET  /health
    POST /ask

Production-ready features:
    - request_id tracking
    - structured logging
    - latency headers
    - FastAPI lifespan singleton pipeline
    - Pydantic validation
    - custom exception handlers
    - OpenAPI docs
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Path Setup
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAG_PIPELINE_DIR = _PROJECT_ROOT / "src" / "app" / "rag_pipeline"

sys.path.insert(0, str(_PROJECT_ROOT))


from src.config.config import API_HOST, API_PORT
from src.app.rag_pipeline.rag_pipeline_orc import RAGPipeline

#=============================================================
#chat request with session_id
#=============================================================
class ChatRequest(BaseModel):

    session_id: str

    query: str

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("bookrag.api")


# ============================================================
# Pipeline Singleton
# ============================================================

_pipeline: RAGPipeline | None = None


# ============================================================
# Lifespan (Startup / Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    global _pipeline

    logger.info(
        "Startup — loading RAG pipeline "
        "(FAISS index, embeddings, OpenAI LLM, Cohere reranker)..."
    )

    try:

        _pipeline = RAGPipeline()

        logger.info(
            "RAG pipeline loaded successfully. Server ready."
        )

    except Exception as e:

        logger.error(
            f"Failed to load pipeline: {e}"
        )

        raise

    yield

    logger.info("Shutdown — releasing pipeline resources.")

    _pipeline = None


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="BookRAG API",
    description="Retrieval-Augmented Generation API using OpenAI + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    logger.info(
        f"→ {request.method} {request.url.path} "
        f"| request_id={request_id}"
    )

    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

    logger.info(
        f"← {request.method} {request.url.path} "
        f"| status={response.status_code} "
        f"| latency={latency_ms:.2f}ms "
        f"| request_id={request_id}"
    )

    return response


# ============================================================
# Request Models
# ============================================================

class AskRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question to ask the RAG pipeline",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str):

        value = value.strip()

        if not value:

            raise ValueError(
                "Query must not be blank"
            )

        return value


class AskResponse(BaseModel):

    request_id: str
    query: str
    answer: str
    latency_ms: float


class HealthResponse(BaseModel):

    status: str
    pipeline_ready: bool
    version: str


class ErrorResponse(BaseModel):

    request_id: str
    error: str
    detail: str | None = None


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError):

    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4())
    )

    return JSONResponse(

        status_code=422,

        content=ErrorResponse(
            request_id=request_id,
            error="Validation error",
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(RuntimeError)
async def handle_runtime_error(request: Request, exc: RuntimeError):

    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4())
    )

    logger.error(
        f"Pipeline error | request_id={request_id}",
        exc_info=True
    )

    return JSONResponse(

        status_code=500,

        content=ErrorResponse(
            request_id=request_id,
            error="Pipeline error",
            detail=str(exc)
        ).model_dump()
    )


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():

    return HealthResponse(
        status="ok",
        pipeline_ready=_pipeline is not None,
        version="1.0.0",
    )


# ============================================================
# Ask Endpoint
# ============================================================

@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):

    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4())
    )

    if _pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready"
        )

    start = time.perf_counter()

    answer = _pipeline.run(body.query)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return AskResponse(

        request_id=request_id,

        query=body.query,

        answer=answer,

        latency_ms=round(latency_ms, 2)
    )

# ============================================================
# Chat EndPoint with memory
# ============================================================

@app.post("/chat")

async def chat(
    body: ChatRequest,
    request: Request):

    if _pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready",
        )

    answer = _pipeline.run_chat(
        session_id=body.session_id,
        query=body.query,
    )

    return {
        "session_id": body.session_id,
        "query": body.query,
        "answer": answer,
    }

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "main:app",

        host=API_HOST,

        port=API_PORT,

        reload=False,

        log_level="info"
    )