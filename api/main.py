import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from ingestion import page_loader
from processing.text import extract_title, html_to_text
from search.pipeline import Doc as SearchDoc
from search.pipeline import search_docs

app = FastAPI(title="OmniVerse AX API", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ax-search")
def ax_search(q: str, preempt: bool = True) -> dict[str, Any]:
    # Stubbed MVP response; replace with hybrid retrieve + swarm + rank
    results: list[dict[str, Any]] = [
        {"url": "https://example.com/1", "title": "Example 1", "score": 0.9},
        {"url": "https://example.com/2", "title": "Example 2", "score": 0.85},
    ]
    return {
        "debated": "resolved-intent: stub",
        "preempt": ["next-hop: stub"] if preempt else [],
        "ranked": results,
    }


class DocIn(BaseModel):
    id: str
    title: str | None = None
    url: str | None = None
    content: str


class SearchRequest(BaseModel):
    query: str
    docs: list[DocIn] | None = None
    urls: list[str] | None = None
    top_k: int = 5


@app.post("/ax-search")
async def ax_search_post(body: SearchRequest) -> dict[str, Any]:
    # Prefer provided docs for deterministic, testable behavior; URL fetching not wired yet.
    docs_in = body.docs or []
    docs = [SearchDoc(id=d.id, title=d.title, url=d.url, content=d.content) for d in docs_in]

    # Fetch and extract URLs concurrently (best-effort)
    urls = body.urls or []
    if urls:
        results = await asyncio.gather(
            *(page_loader.fetch_url(u) for u in urls), return_exceptions=True
        )
        for u, res in zip(urls, results, strict=False):
            if isinstance(res, BaseException):
                continue
            status, html, _ctype = res
            if status != 200 or not html:
                continue
            text = html_to_text(html)
            if not text:
                continue
            title = extract_title(html)
            docs.append(SearchDoc(id=u, title=title, url=u, content=text))

    ranked = search_docs(docs, body.query, top_k=body.top_k) if docs else []
    return {
        "debated": "resolved-intent: tfidf",
        "preempt": [],
        "ranked": ranked,
    }


@app.websocket("/ws-stream")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    # Send a few heartbeat messages then close from server side
    for i in range(3):
        await websocket.send_json({"stream": "ax-update", "seq": i})
        await asyncio.sleep(0.05)
    await websocket.close()
