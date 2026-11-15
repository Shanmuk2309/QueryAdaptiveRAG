"""Optional real-model integration check: QUERY_RAG_REAL_INTEGRATION=1 pytest backend/tests/test_real_integration.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.skipif(os.getenv("QUERY_RAG_REAL_INTEGRATION") != "1", reason="set QUERY_RAG_REAL_INTEGRATION=1 to load the real embedding model")
def test_real_upload_index_query_compare_and_restart(tmp_path):
    from app.config import Settings
    from app.main import create_app

    settings = Settings(tmp_path, tmp_path / "uploads", tmp_path / "indexes", 2,
                        ("http://localhost:3000",), "sentence-transformers/all-MiniLM-L6-v2",
                        3072, 5, "cpu", 2 * 1024 * 1024)
    body = ("Climate change is caused by greenhouse gas emissions. Burning coal, oil, and gas releases carbon dioxide. "
            "Agriculture and land use also contribute methane and nitrous oxide. Climate impacts include rising temperatures. " * 18)
    with TestClient(create_app(settings)) as client:
        uploaded = client.post("/api/v1/documents/upload", files={"files": ("sample.txt", body.encode(), "text/plain")})
        assert uploaded.status_code == 201, uploaded.text
        document_id = uploaded.json()["documents"][0]["document_id"]
        for method in ("M1", "M2", "M3", "M4-v3"):
            indexed = client.post("/api/v1/documents/index", json={"method": method, "document_ids": [document_id]})
            assert indexed.status_code == 200, indexed.text
            result = client.post("/api/v1/query", json={"query": "What causes climate change and what impacts result?", "method": method, "top_k": 3})
            assert result.status_code == 200, result.text
            assert result.json()["retrieved_chunks"]
            assert result.json()["metrics"]["context_tokens"] <= 3072
        compared = client.post("/api/v1/compare", json={"query": "What causes climate change?", "methods": ["M1", "M2", "M3", "M4-v3"], "top_k": 3})
        assert compared.status_code == 200, compared.text
        assert all("error" not in item for item in compared.json()["results"].values())

    with TestClient(create_app(settings)) as restarted:
        available = restarted.get("/api/v1/health").json()["indexes"]
        assert all(available[method] for method in ("M1", "M2", "M3", "M4-v3"))
