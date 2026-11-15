from __future__ import annotations
import json, re, sys, types
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class TinyTokenizer:
    is_fast = True
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping: result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result

class TinyEncoder:
    def __init__(self, *_args, **_kwargs): pass
    def get_sentence_embedding_dimension(self): return 2
    def encode(self, texts, **_kwargs):
        return np.asarray([[1.0, float((sum(map(ord, text)) % 11) + 1)] for text in texts], dtype=np.float32)

class AutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs): return TinyTokenizer()

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=TinyEncoder))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    from app.config import Settings
    from app.main import create_app
    settings = Settings(tmp_path, tmp_path / "uploads", tmp_path / "indexes", 2,
                        ("http://localhost:3000",), "fake-model", 1000, 5, "cpu", 2 * 1024 * 1024)
    with TestClient(create_app(settings)) as test_client: yield test_client

def upload(client):
    text = ("Climate change is driven by greenhouse gas emissions. Burning fossil fuels releases carbon dioxide. Methane is another greenhouse gas. " * 30)
    response = client.post("/api/v1/documents/upload", files={"files": ("climate.txt", text.encode(), "text/plain")})
    assert response.status_code == 201, response.text
    return response.json()["documents"][0]["document_id"]

def test_health_and_document_lifecycle(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    openapi = client.app.openapi()
    request_body = openapi["paths"]["/api/v1/documents/upload"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]
    ref = request_body["content"]["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[-1]
    files = openapi["components"]["schemas"][ref]["properties"]["files"]
    files_array = next(branch for branch in files["anyOf"] if branch.get("type") == "array")
    assert files_array["items"] == {"type": "string", "format": "binary"}
    file = openapi["components"]["schemas"][ref]["properties"]["file"]
    file_binary = next(branch for branch in file["anyOf"] if branch.get("format") == "binary")
    assert file_binary == {"type": "string", "format": "binary"}
    doc_id = upload(client)
    assert client.get("/api/v1/documents").json()[0]["document_id"] == doc_id
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 200
    assert client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": [doc_id]}).status_code == 200
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
    assert client.post("/api/v1/query", json={"query": "climate", "method": "M1"}).status_code == 404
    assert client.get("/api/v1/documents/../../outside").status_code in (404, 422)

def test_upload_accepts_multiple_actual_txt_files(client):
    response = client.post("/api/v1/documents/upload", files=[
        ("files", ("first.txt", b"First uploaded text file.", "text/plain")),
        ("files", ("second.txt", b"Second uploaded text file.", "text/plain")),
    ])
    assert response.status_code == 201, response.text
    documents = response.json()["documents"]
    assert len(documents) == 2
    assert {row["filename"] for row in documents} == {"first.txt", "second.txt"}
    assert all(row["text_length"] > 0 and row["processing_status"] == "ready" for row in documents)

def test_single_txt_upload_returns_metadata_and_persists_file(client):
    text = b"A small stored TXT document."
    response = client.post("/api/v1/documents/upload", files={"files": ("stored.txt", text, "text/plain")})
    assert response.status_code == 201, response.text
    document = response.json()["documents"][0]
    assert document["filename"] == "stored.txt"
    assert document["text_length"] == len(text.decode("utf-8"))
    assert document["processing_status"] == "ready"
    stored_path = client.app.state.settings.upload_dir / f"{document['document_id']}.txt"
    metadata_path = client.app.state.settings.upload_dir / f"{document['document_id']}.json"
    assert stored_path.read_bytes() == text
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["text"] == text.decode("utf-8")

def test_direct_file_upload_field_accepts_txt(client):
    text = b"Uploaded through the direct Swagger file chooser."
    response = client.post("/api/v1/documents/upload", files={"file": ("direct.txt", text, "text/plain")})
    assert response.status_code == 201, response.text
    document = response.json()["documents"][0]
    assert document["filename"] == "direct.txt"
    assert document["text_length"] == len(text.decode("utf-8"))
    stored_path = client.app.state.settings.upload_dir / f"{document['document_id']}.txt"
    assert stored_path.read_bytes() == text

def test_oversized_upload_is_rejected(client):
    object.__setattr__(client.app.state.settings, "max_upload_size_bytes", 2)
    response = client.post("/api/v1/documents/upload", files={"files": ("large.txt", b"too large", "text/plain")})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"

@pytest.mark.parametrize("name,mime,data", [("bad.doc", "application/msword", b"x"), ("empty.txt", "text/plain", b"")])
def test_invalid_upload(client, name, mime, data):
    response = client.post("/api/v1/documents/upload", files={"files": (name, data, mime)})
    assert response.status_code in (415, 422)

def test_pdf_upload(client):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    import io
    writer = PdfWriter(); page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject(); stream.set_data(b"BT /F1 12 Tf 20 100 Td (PDF climate document text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO(); writer.write(buf)
    response = client.post("/api/v1/documents/upload", files={"files": ("blank.pdf", buf.getvalue(), "application/pdf")})
    assert response.status_code == 201, response.text
    assert response.json()["documents"][0]["text_length"] > 0

def test_corrupt_pdf_rejected(client):
    response = client.post("/api/v1/documents/upload", files={"files": ("broken.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 422

@pytest.mark.parametrize("method", ["M1", "M2", "M3", "M4-v3"])
def test_index_query_all_methods(client, method):
    doc_id = upload(client)
    indexed = client.post("/api/v1/documents/index", json={"method": method, "document_ids": [doc_id]})
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["chunk_count"] > 0
    response = client.post("/api/v1/query", json={"query": "What are causes of climate change?", "method": method, "top_k": 2})
    assert response.status_code == 200, response.text
    assert response.json()["metrics"]["context_tokens"] <= 1000
    assert response.json()["retrieved_chunks"]

def test_validation_missing_index_and_comparison(client):
    assert client.post("/api/v1/query", json={"query": "x", "method": "M1"}).status_code == 404
    assert client.post("/api/v1/query", json={"query": " ", "method": "M1"}).status_code == 422
    assert client.post("/api/v1/query", json={"query": "x", "method": "nope"}).status_code == 422
    assert client.post("/api/v1/query", json={"query": "x", "method": "M1", "top_k": 0}).status_code == 422
    missing = client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": ["missing-document"]})
    assert missing.status_code == 404
    assert client.delete("/api/v1/documents/missing-document").status_code == 404
    doc_id = upload(client)
    for method in ("M1", "M2", "M3", "M4-v3"):
        assert client.post("/api/v1/documents/index", json={"method": method, "document_ids": [doc_id]}).status_code == 200
    result = client.post("/api/v1/compare", json={"query": "What causes climate change?", "methods": ["M1", "M2", "M3", "M4-v3"]})
    assert result.status_code == 200, result.text
    assert list(result.json()["results"]) == ["M1", "M2", "M3", "M4-v3"]

def test_token_budget_is_enforced(client):
    doc_id = upload(client)
    assert client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": [doc_id]}).status_code == 200
    object.__setattr__(client.app.state.retrieval_service.settings, "context_token_budget", 10)
    response = client.post("/api/v1/query", json={"query": "What causes climate change?", "method": "M1", "top_k": 5})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "context_budget_exceeded"

def test_index_persistence_after_restart(client, tmp_path):
    from app.config import Settings
    from app.main import create_app
    doc_id = upload(client)
    assert client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": [doc_id]}).status_code == 200
    settings = Settings(tmp_path, tmp_path / "uploads", tmp_path / "indexes", 2,
                        ("http://localhost:3000",), "fake-model", 1000, 5, "cpu", 2 * 1024 * 1024)
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/v1/health").json()["indexes"]["M1"] is True
        assert restarted.post("/api/v1/query", json={"query": "climate", "method": "M1"}).status_code == 200

def test_reindex_is_idempotent_and_deletion_invalidates_indexes(client):
    doc_id = upload(client)
    first = client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": [doc_id]})
    second = client.post("/api/v1/documents/index", json={"method": "M1", "document_ids": [doc_id]})
    assert first.status_code == second.status_code == 200
    assert first.json()["chunk_count"] == second.json()["chunk_count"]
    assert client.get(f"/api/v1/documents/{doc_id}").json()["indexed_methods"] == ["M1"]

    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
    assert client.get("/api/v1/health").json()["indexes"]["M1"] is False
    assert client.post("/api/v1/query", json={"query": "climate", "method": "M1"}).status_code == 404
