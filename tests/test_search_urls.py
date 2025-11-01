from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_post_ax_search_with_urls(monkeypatch):
    async def fake_fetch_url(url: str, *, timeout: float = 10.0, user_agent: str = ""):
        if "a" in url:
            return (
                200,
                "<html><title>A</title><body>Python typing with hints</body></html>",
                "text/html",
            )
        return (
            200,
            "<html><title>B</title><body>Cooking recipes and food</body></html>",
            "text/html",
        )

    monkeypatch.setattr("ingestion.page_loader.fetch_url", fake_fetch_url)

    payload = {
        "query": "python typing",
        "top_k": 2,
        "urls": ["https://x.test/a", "https://x.test/b"],
    }
    r = client.post("/ax-search", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "ranked" in data
    ranked = data["ranked"]
    assert len(ranked) == 2
    assert ranked[0]["id"] == "https://x.test/a"
    assert ranked[0]["title"] == "A"
