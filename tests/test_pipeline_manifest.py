import json
from dataclasses import replace

import numpy as np

from query_adaptive_rag import pipeline
from query_adaptive_rag.config import load_config


class TinyTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re

        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


class TinyModel:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, sentences, **kwargs):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(sentences), 1))


def test_pipeline_writes_manifest_with_effective_cli_limit(tmp_path, monkeypatch):
    base = load_config("configs/day1.yaml")
    config = replace(base, output_dir=tmp_path / "processed", cache_dir=tmp_path / "cache")
    records = [{
        "id": "one",
        "dataset": "squad",
        "question": "What?",
        "answers": ["alpha"],
        "documents": [{"doc_id": "squad:doc", "title": "doc", "text": "alpha beta."}],
        "gold_evidence": [{"doc_id": "squad:doc", "start": 0, "end": 5}],
        "complexity": "single-fact",
    }]
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(pipeline, "SentenceTransformer", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(pipeline, "load_configured_datasets", lambda settings, seed: records)

    output_dir = pipeline.run("configs/day1.yaml", 5, "python -m query_adaptive_rag.pipeline --limit 5")
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["random_seed"] == config.seed
    assert manifest["cli_overrides"]["limit_per_dataset"] == 5
    assert manifest["effective_config"]["datasets"]["squad"]["limit"] == 5
    assert manifest["model_identifier"] == config.model_name
    assert manifest["tokenizer_identifier"] == config.tokenizer_name
    assert manifest["executed_command"].endswith("--limit 5")
    assert manifest["dataset_counts"] == {"total": 1, "by_dataset": {"squad": 1}}
    assert manifest["chunk_counts"]["semantic"] == 1
    assert manifest["dependency_versions"]["numpy"]
