# QueryAdaptiveRAG API

FastAPI integration for the existing research implementation. The backend calls the project's `src/query_adaptive_rag` chunking, classification, FAISS retrieval, context assembly, and M4-v3 expansion functions directly. It does not import experiment datasets or evaluation CSVs at runtime.

## Architecture

- `app/api/routes`: versioned HTTP endpoints.
- `app/services`: upload/extraction, indexing, retrieval, and method comparison.
- `app/storage`: local uploads and method-isolated FAISS/vector/chunk bundles.
- `app/main.py`: application lifecycle, cached embedding/tokenizer model, persistent index restoration, CORS, and safe error handling.
- `tests`: API tests with deterministic model doubles; no research evaluation data is involved.

M1 uses the existing fixed chunker with 256 tokens and 32 overlap. M2 uses 510 and 32. M3 uses the existing semantic chunker and its 0.45/80/400 settings. M4-v3 indexes those semantic units separately and invokes the existing query-only classifier and fact-level expander. Its default expansion rules and V3 parameters match `configs/day3_heldout_10000_m4v3.yaml`.

## Prerequisites and install

Use Python 3.11 and run from the `backend/` directory. The research dependencies are included through the parent requirements file.

FastAPI is intentionally constrained to `<0.129.1`: releases beginning with 0.129.1 changed the generated uploaded-file JSON Schema from `format: binary` to `contentMediaType`, which causes the bundled Swagger UI to show a text field. The endpoint itself uses the native `UploadFile`/`File` API and the schema is generated without a custom OpenAPI rewrite.

```powershell
cd C:\Users\shanm\OneDrive\Desktop\QueryAdaptiveRAG\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set environment variables in the shell or load `.env` with your preferred environment manager. `Settings` reads process environment variables; it intentionally does not add a dotenv dependency.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | Base directory for runtime data and semantic embedding cache |
| `UPLOAD_DIR` | `$DATA_DIR/uploads` | Uploaded file and extracted document metadata storage |
| `INDEX_DIR` | `$DATA_DIR/indexes` | Method-specific FAISS index bundles |
| `MAX_UPLOAD_SIZE_MB` | `25` | Per-file upload cap |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed browser origins |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformer and tokenizer identifier |
| `CONTEXT_TOKEN_BUDGET` | `3072` | Hard assembled-context budget |
| `TOP_K_DEFAULT` | `5` | Documented client default (request schema default is 5) |
| `DEVICE` | `cpu` | Sentence Transformers device; CPU is the supported default |

The model is initialized once per app process during lifespan startup. Model download/cache availability is required on first startup. Index bundles are loaded only if the persisted model identifier matches the configured model.

## Start and API docs

```powershell
python -m uvicorn app.main:app --reload
```

Interactive Swagger is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs); OpenAPI JSON is at `/openapi.json`.

All API routes are under `/api/v1`.

### Health

`GET /api/v1/health` reports API version/status, supported methods, per-method index availability, and embedding-model load state.

### Upload/list/get/delete

- `POST /api/v1/documents/upload` multipart field `files`, repeatable for multiple PDF/TXT files.
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}`
- `DELETE /api/v1/documents/{document_id}`

PowerShell upload:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/upload `
  -F "files=@C:\docs\climate.txt;type=text/plain" `
  -F "files=@C:\docs\report.pdf;type=application/pdf"
```

The response provides `document_id`, safe display filename, extracted text length, processing status, and indexed methods. It does not expose storage paths.

### Index

`POST /api/v1/documents/index` accepts a method and optional document ID list. Omitting IDs indexes all uploaded documents. Re-indexing replaces that method's bundle atomically at the logical bundle level; each method is isolated.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/index `
  -H "Content-Type: application/json" `
  -d '{"method":"M4-v3"}'
```

Methods are `M1`, `M2`, `M3`, and `M4-v3`. The response reports document/chunk counts, model identifier, duration, and status.

### Query

`POST /api/v1/query` accepts `{ "query": "...", "method": "M1", "top_k": 5 }`. Use the same shape with each method name. M4-v3 returns query-only predicted complexity, expanded chunks, and accepted/rejected candidate decisions. Other methods return no predicted complexity and empty expansion results. Metrics include retrieval/expansion latency and actual assembled token count; normal requests do not calculate evaluation metrics.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What are the main causes of climate change?","method":"M4-v3","top_k":5}'
```

Response shape:

```json
{
  "query": "What are the main causes of climate change?",
  "method": "M4-v3",
  "predicted_complexity": "single_fact",
  "retrieved_chunks": [{"chunk_id":"...","document_id":"...","filename":"climate.txt","text":"...","score":0.7,"rank":1,"start":0,"end":100}],
  "expanded_chunks": [],
  "final_context": "...",
  "sources": [{"document_id":"...","filename":"climate.txt"}],
  "metrics": {"retrieval_latency_ms": 1.2,"expansion_latency_ms": 0.1,"context_tokens": 20,"retrieved_chunk_count": 1,"expanded_chunk_count": 0},
  "candidate_decisions": []
}
```

The values above illustrate field types and are not benchmark results.

### Compare

`POST /api/v1/compare` accepts a query, unique methods, and top-k. Results are keyed by the exact method name. If one requested index is missing, that method gets an `error` object and other method results are still returned.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/compare `
  -H "Content-Type: application/json" `
  -d '{"query":"What are the main causes of climate change?","methods":["M1","M2","M3","M4-v3"],"top_k":5}'
```

## Tests

From `backend/` run:

```powershell
python -m pytest -q
```

The backend tests use temporary data directories and model doubles. Existing research tests remain independent and unchanged; from the repository root run `python -m pytest -q` to execute them alongside backend tests.

## Limitations

- PDF text extraction uses `pypdf`; image-only PDFs require OCR and are reported as empty text. OCR is not added because no OCR implementation is present in the research project.
- A M4-v3 index is its own semantic-chunk index bundle, separate from M3, so M4-v3 calls the existing expansion algorithm without overwriting M3. Both use the existing semantic chunker settings.
- Deleting any document invalidates all method index bundles because their vectors contain document-specific chunks. Re-index the desired methods after deletion.
- Index writes use separate files and are suitable for a single local API process; concurrent multi-worker index writes are not coordinated. Run one worker for indexing.
- Persistence stores extracted text in private per-document metadata JSON and FAISS vectors/chunks in the configured local data directory. Protect these directories with appropriate OS permissions and backups.
- The research code has no native FAISS deserializer. This backend persists FAISS through `faiss.write_index/read_index` and stores the matching chunk/vector arrays alongside it.
