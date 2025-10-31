from typing import Any

from fastapi import FastAPI

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
