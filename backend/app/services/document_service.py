from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from fastapi import UploadFile

from ..config import Settings
from ..storage.file_storage import FileStorage
from ..storage.index_storage import IndexStorage
from ..utils.errors import BackendError


class DocumentService:
    def __init__(self, settings: Settings, storage: FileStorage, indexes: IndexStorage, runtime=None):
        self.settings, self.storage, self.indexes, self.runtime = settings, storage, indexes, runtime

    @staticmethod
    def _extract(name: str, payload: bytes) -> str:
        if name.lower().endswith(".txt"):
            try:
                return payload.decode("utf-8-sig").strip()
            except UnicodeDecodeError as exc:
                raise BackendError(422, "invalid_text", "TXT file must be UTF-8 encoded.") from exc
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(payload), strict=True)
            if reader.is_encrypted:
                raise BackendError(422, "pdf_encrypted", "Encrypted PDF files are not supported.")
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(422, "invalid_pdf", "PDF is corrupted or text could not be extracted.") from exc

    async def upload(self, files: list[UploadFile]) -> list[dict]:
        if not files:
            raise BackendError(422, "empty_upload", "At least one file is required.")
        output = []
        for file in files:
            filename = Path((file.filename or "").replace("\\", "/")).name
            suffix = Path(filename).suffix.lower()
            mime = (file.content_type or "").lower()
            allowed = {".txt": {"text/plain", "application/octet-stream"}, ".pdf": {"application/pdf", "application/octet-stream"}}
            if suffix not in allowed or mime not in allowed[suffix]:
                raise BackendError(415, "unsupported_file", "Only PDF and TXT files with matching MIME types are supported.")
            payload = await file.read(self.settings.max_upload_size_bytes + 1)
            if not payload:
                raise BackendError(422, "empty_file", "Uploaded files must not be empty.")
            if len(payload) > self.settings.max_upload_size_bytes:
                raise BackendError(413, "file_too_large", f"File exceeds the {self.settings.max_upload_size_mb} MB upload limit.")
            text = self._extract(filename, payload)
            if not text:
                raise BackendError(422, "empty_document", "No extractable text was found in the document.")
            doc_id = uuid.uuid4().hex
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[:180] or f"document{suffix}"
            row = {"document_id": doc_id, "filename": safe_name, "stored_name": f"{doc_id}{suffix}",
                   "text": text, "text_length": len(text), "processing_status": "ready", "indexed_methods": [],
                   "content_type": mime, "created_at": datetime.now(timezone.utc).isoformat()}
            self.storage.save(row, payload, suffix)
            output.append(row)
        return output

    def list(self) -> list[dict]:
        return self.storage.list()

    def get(self, doc_id: str) -> dict:
        return self.storage.get(doc_id)

    def delete(self, doc_id: str) -> None:
        self.storage.delete(doc_id)
        self.indexes.delete_document(doc_id)
        if self.runtime is not None:
            for method in ("M1", "M2", "M3", "M4-v3"):
                self.runtime.indexes.pop(method, None)
                self.runtime.chunks_by_method.pop(method, None)
                self.runtime.vectors_by_method.pop(method, None)
        for row in self.storage.list():
            if row.get("indexed_methods"):
                row["indexed_methods"] = []
                self.storage.update(row)
