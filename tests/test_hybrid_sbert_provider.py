from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_hybrid_sbert_provider_uses_cache(monkeypatch, tmp_path: Path):
    # Use isolated cache dir for determinism
    monkeypatch.setenv("OMNIVERSE_EMBED_CACHE_DIR", str(tmp_path / "emb"))
    calls = {"n": 0}

    # Fake sbert embedder
    def fake_sbert_embed(texts, model_name=None):
        import numpy as np

        calls["n"] += 1
        # simple 8-dim vector based on length
        arr = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i, len(t) % 8] = 1.0
        return arr

    monkeypatch.setattr("embeddings.sbert.embed_texts", fake_sbert_embed)

    payload = {
        "query": "abc",
        "top_k": 2,
        "strategy": "hybrid",
        "embedder": "sbert",
        "alpha": 0.5,
        "docs": [
            {"id": "a", "title": "A", "content": "aaa"},
            {"id": "b", "title": "B", "content": "bbb"},
        ],
    }

    # First call computes and caches
    r1 = client.post("/ax-search", json=payload)
    assert r1.status_code == 200
    called_after_first = calls["n"]
    assert called_after_first >= 1

    # Second call should hit cache (no additional calls)
    r2 = client.post("/ax-search", json=payload)
    assert r2.status_code == 200
    assert calls["n"] == called_after_first
