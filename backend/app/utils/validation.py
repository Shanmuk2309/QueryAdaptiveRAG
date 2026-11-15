from __future__ import annotations

METHODS = {"M1", "M2", "M3", "M4-v3"}


def normalize_method(method: str) -> str:
    if method not in METHODS:
        from .errors import BackendError
        raise BackendError(422, "invalid_method", f"Method must be one of {', '.join(sorted(METHODS))}.")
    return method
