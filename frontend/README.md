# QueryAdaptiveRAG frontend

Responsive React and Vite workspace for the existing QueryAdaptiveRAG FastAPI backend. The UI calls backend endpoints for health, documents, indexing, retrieval, and comparison; it does not implement research algorithms or generate answers.

## Prerequisites

- Node.js 20 or newer with npm.
- Python 3.11 and the backend dependencies installed.
- The FastAPI service available at `http://127.0.0.1:8001` by default.

## Install and configure

From this directory:

```powershell
npm install
Copy-Item .env.example .env
```

Set `VITE_API_BASE_URL` in `.env` if the backend uses another address. For example:

```env
VITE_API_BASE_URL=http://127.0.0.1:8001
```

The Vite development server binds to `localhost:3000`, which is included in the backend's default CORS origins. If you change the frontend origin, add it to `CORS_ORIGINS` in the backend environment and restart FastAPI.

## Start the applications

Terminal 1, from the project root:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Terminal 2, from `frontend/`:

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend Swagger UI is at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

## Features

- **Overview:** live API connection, uploaded document count, available methods, and currently loaded index status.
- **Documents:** multi-file PDF/TXT upload with drag and drop, selection preview, upload progress, document metadata, details, and confirmed deletion.
- **Indexing:** select one or multiple uploaded documents and one or more methods. The UI sends a separate request for each selected method and reports each result independently.
- **Query lab:** select M1, M2, M3, or M4-v3 and inspect returned context, source chunks, scores, timing, complexity, and M4-v3 candidate decisions.
- **Compare methods:** submit a shared query across selected methods and inspect per-method context, chunks, context token counts, latency, complexity, and expansion decisions.

The API does not support filtering queries by document, so query pages explicitly use the selected method's full index. Uploaded document listing exposes extracted character counts and processing/index state, but not original upload byte sizes or extracted full text. The frontend shows original file sizes before upload and does not invent metadata the API does not return. Deleting a document invalidates all backend indexes; the confirmation dialog explains that the indexes must be rebuilt.

## API integration

All requests use the centralized Axios client in `src/api/client.js`. The base URL is read from `VITE_API_BASE_URL`. Backend validation errors and network failures are translated to readable messages. A failed health check leaves the rest of the application navigable and clearly marks the API offline. The Compare page displays per-method missing-index errors returned by the backend without hiding other successful methods.

The interface labels API output as retrieved context. It does not generate an answer, calculate Recall@K/MRR/context precision, use gold evidence, or claim that M4-v3 improves retrieval.

## Build and validation

```powershell
npm run build
```

There is no frontend test or lint configuration in this project yet. The production Vite build is the available frontend compilation check.
