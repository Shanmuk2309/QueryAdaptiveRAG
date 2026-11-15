from __future__ import annotations

from time import perf_counter

from query_adaptive_rag.chunking.fixed_size import fixed_size_chunks
from query_adaptive_rag.chunking.semantic import semantic_chunks
from query_adaptive_rag.retrieval.dense import DenseRetriever

from ..config import Settings
from ..storage.file_storage import FileStorage
from ..storage.index_storage import IndexStorage
from ..utils.errors import BackendError
from .runtime import Runtime


class IndexingService:
    def __init__(self, settings: Settings, documents: FileStorage, indexes: IndexStorage, runtime: Runtime):
        self.settings, self.documents, self.storage, self.runtime = settings, documents, indexes, runtime

    def index(self, method: str, document_ids: list[str] | None = None) -> dict:
        started = perf_counter()
        available = self.documents.list()
        selected = [row for row in available if document_ids is None or row["document_id"] in set(document_ids)]
        if document_ids and len({row["document_id"] for row in selected}) != len(set(document_ids)):
            raise BackendError(404, "document_not_found", "One or more requested documents were not found.")
        if not selected:
            raise BackendError(422, "no_documents", "Upload at least one document before indexing.")
        selected_ids = {row["document_id"] for row in selected}
        # Keep chunks for documents outside a partial re-index request. This makes
        # the method endpoint an update operation instead of silently dropping data.
        preserved = [chunk for chunk in self.runtime.chunks_by_method.get(method, [])
                     if chunk.get("doc_id") not in selected_ids]
        chunks = list(preserved)
        for row in selected:
            base = row["text"]
            if method == "M1":
                part = fixed_size_chunks(base, row["document_id"], self.runtime.tokenizer, 256, 32)
            elif method == "M2":
                part = fixed_size_chunks(base, row["document_id"], self.runtime.tokenizer, 510, 32)
            else:
                part = semantic_chunks(base, row["document_id"], self.runtime.tokenizer, self.runtime.encoder,
                                       0.45, 80, 400, self.settings.data_dir / "sentence_embeddings", self.settings.embedding_model)
            for chunk in part:
                chunk["filename"] = row["filename"]
                chunk["source_document_id"] = row["document_id"]
            chunks.extend(part)
        if not chunks:
            raise BackendError(422, "no_chunks", "No indexable text chunks were produced.")
        retriever = DenseRetriever(self.runtime.encoder, self.settings.embedding_model)
        try:
            retriever.build(chunks)
            self.storage.save(method, retriever.index, retriever.chunks, retriever.vectors, self.settings.embedding_model)
        except Exception as exc:
            raise BackendError(500, "indexing_failed", "Unable to build or persist this method index.") from exc
        self.runtime.indexes[method] = retriever
        self.runtime.chunks_by_method[method] = retriever.chunks
        self.runtime.vectors_by_method[method] = {chunk["chunk_id"]: retriever.vectors[i] for i, chunk in enumerate(retriever.chunks)}
        indexed_ids = {chunk["doc_id"] for chunk in chunks}
        for row in available:
            methods = set(row.get("indexed_methods", []))
            if row["document_id"] in indexed_ids:
                methods.add(method)
            else:
                methods.discard(method)
            row["indexed_methods"] = sorted(methods)
            self.documents.update(row)
        return {"method": method, "document_count": len(indexed_ids), "chunk_count": len(chunks),
                "embedding_model": self.settings.embedding_model, "indexing_duration_ms": (perf_counter()-started)*1000,
                "index_status": "ready"}
