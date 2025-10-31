from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_post_ax_search_hybrid_mode_returns_results():
    payload = {
        "query": "deep learning",
        "top_k": 2,
        "strategy": "hybrid",
        "alpha": 0.5,
        "docs": [
            {"id": "a", "title": "AI", "content": "Deep learning methods with neural networks"},
            {"id": "b", "title": "Cooking", "content": "Recipes and delicious meals"},
        ],
    }
    r = client.post("/ax-search", json=payload)
    assert r.status_code == 200
    data = r.json()
    ranked = data["ranked"]
    assert len(ranked) == 2
    assert ranked[0]["id"] == "a"
