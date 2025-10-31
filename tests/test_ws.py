from fastapi.testclient import TestClient

from api.main import app


def test_ws_stream_once():
    client = TestClient(app)
    with client.websocket_connect("/ws-stream") as ws:
        msg = ws.receive_json()
        assert msg["stream"] == "ax-update"
