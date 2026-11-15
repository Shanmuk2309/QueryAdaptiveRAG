from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class IndexStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, method: str) -> Path:
        path = self.root / method
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, method: str, index: Any, chunks: list[dict], vectors: np.ndarray, model_name: str) -> None:
        target = self.directory(method)
        faiss.write_index(index, str(target / "index.faiss"))
        np.save(target / "vectors.npy", vectors, allow_pickle=False)
        (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps({"method": method, "model_name": model_name}), encoding="utf-8")

    def load(self, method: str) -> tuple[Any, list[dict], np.ndarray, str] | None:
        target = self.root / method
        required = [target / name for name in ("index.faiss", "vectors.npy", "chunks.json", "manifest.json")]
        if not all(path.is_file() for path in required):
            return None
        try:
            manifest = json.loads(required[3].read_text(encoding="utf-8"))
            if manifest.get("method") != method:
                return None
            return (faiss.read_index(str(required[0])), json.loads(required[2].read_text(encoding="utf-8")),
                    np.load(required[1], allow_pickle=False), manifest["model_name"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def delete_document(self, document_id: str) -> None:
        # Rebuilding method indexes on next explicit indexing keeps persisted metadata consistent.
        for method in ("M1", "M2", "M3", "M4-v3"):
            (self.root / method / "index.faiss").unlink(missing_ok=True)
            (self.root / method / "vectors.npy").unlink(missing_ok=True)
            (self.root / method / "chunks.json").unlink(missing_ok=True)
            (self.root / method / "manifest.json").unlink(missing_ok=True)
