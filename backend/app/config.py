from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, fallback: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else fallback
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    upload_dir: Path
    index_dir: Path
    max_upload_size_mb: int
    cors_origins: tuple[str, ...]
    embedding_model: str
    context_token_budget: int
    top_k_default: int
    device: str
    max_upload_size_bytes: int

    def __post_init__(self) -> None:
        if self.context_token_budget < 1:
            raise ValueError("CONTEXT_TOKEN_BUDGET must be positive")
        if self.top_k_default < 1:
            raise ValueError("TOP_K_DEFAULT must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _path_env("DATA_DIR", Path("data"))
        max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
        if max_mb < 1:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
        return cls(
            data_dir=data_dir,
            upload_dir=_path_env("UPLOAD_DIR", data_dir / "uploads"),
            index_dir=_path_env("INDEX_DIR", data_dir / "indexes"),
            max_upload_size_mb=max_mb,
            cors_origins=tuple(x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "3072")),
            top_k_default=int(os.getenv("TOP_K_DEFAULT", "5")),
            device=os.getenv("DEVICE", "cpu"),
            max_upload_size_bytes=max_mb * 1024 * 1024,
        )

