from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from ..utils.errors import BackendError


class FileStorage:
    def __init__(self, upload_dir: Path):
        self.root = upload_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _metadata(self, document_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", document_id):
            raise BackendError(404, "document_not_found", "Document was not found.")
        return self.root / f"{document_id}.json"

    def save(self, document: dict[str, Any], payload: bytes, suffix: str) -> None:
        data_path = self.root / f"{document['document_id']}{suffix}"
        data_path.write_bytes(payload)
        try:
            self._metadata(document["document_id"]).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        except Exception:
            data_path.unlink(missing_ok=True)
            raise

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def get(self, document_id: str) -> dict[str, Any]:
        path = self._metadata(document_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BackendError(404, "document_not_found", "Document was not found.")
        except (OSError, json.JSONDecodeError):
            raise BackendError(500, "storage_error", "Document metadata could not be read.")

    def source_path(self, document: dict[str, Any]) -> Path:
        path = (self.root / document["stored_name"]).resolve()
        if path.parent != self.root:
            raise BackendError(500, "storage_error", "Stored document reference is invalid.")
        return path

    def delete(self, document_id: str) -> None:
        row = self.get(document_id)
        self.source_path(row).unlink(missing_ok=True)
        self._metadata(document_id).unlink(missing_ok=True)

    def update(self, document: dict[str, Any]) -> None:
        self._metadata(document["document_id"]).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
