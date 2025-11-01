import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from index.persistent import get_index
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
    strategy: str | None = "tfidf"  # "tfidf" | "hybrid"
    alpha: float = 0.5
    embedder: str | None = "hashed"  # "hashed" | "sbert"
    model: str | None = None


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

    strategy = (body.strategy or "tfidf").lower()
    if docs:
        if strategy == "hybrid":
            from search.hybrid import HybridConfig, hybrid_search

            ranked = hybrid_search(
                docs,
                body.query,
                top_k=body.top_k,
                cfg=HybridConfig(
                    alpha=body.alpha,
                    provider=(body.embedder or "hashed").lower(),
                    model_name=body.model or "sentence-transformers/all-MiniLM-L6-v2",
                ),
            )
        else:
            ranked = search_docs(docs, body.query, top_k=body.top_k)
    else:
        ranked = []
    return {
        "debated": "resolved-intent: tfidf",
        "preempt": [],
        "ranked": ranked,
    }


class IndexRequest(BaseModel):
    docs: list[DocIn] | None = None
    urls: list[str] | None = None
    embedder: str | None = "hashed"  # "hashed" | "sbert"
    model: str | None = None
    dim: int | None = 256


@app.post("/index")
async def index_upsert(body: IndexRequest) -> dict[str, int]:
    docs_in = body.docs or []
    docs = [SearchDoc(id=d.id, title=d.title, url=d.url, content=d.content) for d in docs_in]

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

    idx = get_index(
        provider=(body.embedder or "hashed").lower(),
        model_name=body.model,
        dim=int(body.dim or 256),
    )
    added = idx.upsert(docs)
    return {"added": added, "total": idx.stats()["count"]}


@app.get("/index/stats")
def index_stats() -> dict[str, int | str]:
    idx = get_index()
    return idx.stats()


class IndexedSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    embedder: str | None = None
    model: str | None = None


@app.post("/search-indexed")
def search_indexed(body: IndexedSearchRequest) -> dict[str, list[dict]]:
    idx = get_index(provider=(body.embedder or "hashed"), model_name=body.model)
    ranked = idx.search(body.query, top_k=body.top_k)
    return {"ranked": ranked}


@app.websocket("/ws-stream")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    # Send a few heartbeat messages then close from server side
    for i in range(3):
        await websocket.send_json({"stream": "ax-update", "seq": i})
        await asyncio.sleep(0.05)
    await websocket.close()
