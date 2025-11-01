from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_post_ax_search_with_docs_ranks_expected_top():
    payload = {
        "query": "python typing",
        "top_k": 2,
        "docs": [
            {
                "id": "a",
                "title": "Intro to Cooking",
                "content": "Recipes and ingredients for delicious meals.",
            },
            {
                "id": "b",
                "title": "Advanced Python",
                "content": "Typing in Python with mypy and type hints.",
            },
        ],
    }
    r = client.post("/ax-search", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "ranked" in data
    ranked = data["ranked"]
    assert len(ranked) == 2
    # Expect the Python-related doc to rank first
    assert ranked[0]["id"] == "b"
