from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from .config import Settings
from .storage.file_storage import FileStorage
from .storage.index_storage import IndexStorage
from .services.document_service import DocumentService
from .services.indexing_service import IndexingService
from .services.retrieval_service import RetrievalService
from .services.comparison_service import ComparisonService
from .services.runtime import Runtime
from .api.routes import health, documents, query, comparison

logger = logging.getLogger("queryadaptiverag.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    runtime = None
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
        encoder = SentenceTransformer(settings.embedding_model, device=settings.device)
        # The experiment's encoder configuration is 512 positions (M2 contains
        # 510 content tokens plus tokenizer special tokens). Match it here.
        position_limit = getattr(getattr(encoder, "_first_module", lambda: None)(), "auto_model", None)
        position_limit = getattr(getattr(position_limit, "config", None), "max_position_embeddings", 512)
        if position_limit < 512:
            raise RuntimeError("Configured embedding model supports fewer than 512 input positions")
        encoder.max_seq_length = 512
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer is required for source offsets and token budgets")
        runtime = Runtime(encoder, tokenizer, settings.embedding_model, {}, {}, {})
    except Exception:
        logger.exception("Embedding runtime initialization failed")
    file_storage = FileStorage(settings.upload_dir)
    index_storage = IndexStorage(settings.index_dir)
    app.state.runtime = runtime
    app.state.index_storage = index_storage
    app.state.document_service = DocumentService(settings, file_storage, index_storage, runtime)
    if runtime is not None:
        for method in ("M1", "M2", "M3", "M4-v3"):
            bundle = index_storage.load(method)
            if bundle and bundle[3] == settings.embedding_model:
                runtime.load_bundle(method, bundle)
    if runtime is None:
        app.state.indexing_service = None
        app.state.retrieval_service = None
        app.state.comparison_service = None
    else:
        app.state.indexing_service = IndexingService(settings, file_storage, index_storage, runtime)
        app.state.retrieval_service = RetrievalService(settings, runtime)
        app.state.comparison_service = ComparisonService(app.state.retrieval_service)
    app.state.services = {"settings": settings, "runtime": runtime}
    try:
        yield
    finally:
        app.state.runtime = None


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(title="QueryAdaptiveRAG API", version="1.0.0", lifespan=lifespan)
    application.state.settings = settings or Settings.from_env()
    application.add_middleware(CORSMiddleware, allow_origins=list(application.state.settings.cors_origins),
                               allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
    application.include_router(health.router, prefix="/api/v1", tags=["health"])
    application.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
    application.include_router(query.router, prefix="/api/v1/query", tags=["query"])
    application.include_router(comparison.router, prefix="/api/v1/compare", tags=["comparison"])

    @application.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": {"code": "internal_error", "message": "An unexpected server error occurred."}})
    return application


app = create_app()
