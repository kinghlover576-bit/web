from fastapi.testclient import TestClient

from api.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ax_search_stub():
    client = TestClient(app)
    r = client.get("/ax-search", params={"q": "test"})
    assert r.status_code == 200
    body = r.json()
    assert "debated" in body and "ranked" in body
    assert isinstance(body["ranked"], list)
