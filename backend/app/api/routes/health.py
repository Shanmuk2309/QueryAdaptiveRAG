from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    state = request.app.state
    runtime = getattr(state, "runtime", None)
    methods = ("M1", "M2", "M3", "M4-v3")
    return {"status": "ok", "version": "1.0.0", "available_methods": list(methods),
            "indexes": {method: bool(runtime and method in runtime.indexes) for method in methods},
            "embedding_model": {"name": state.settings.embedding_model, "status": "loaded" if runtime else "unavailable"}}
