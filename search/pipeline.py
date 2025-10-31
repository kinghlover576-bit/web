from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from index.tfidf import TfidfIndex


@dataclass
class Doc:
    id: str
    title: str | None
    url: str | None
    content: str


def search_docs(docs: Iterable[Doc], query: str, top_k: int = 5) -> list[dict]:
    idx = TfidfIndex()
    ordered: list[Doc] = []
    for d in docs:
        ordered.append(d)
        idx.add_document(d.id, d.content)
    idx.build()
    ranked = idx.search(query, top_k=top_k)
    by_id = {d.id: d for d in ordered}

    # Start with positive scores
    out: list[dict] = []
    seen: set[str] = set()
    for doc_id, score in ranked:
        seen.add(doc_id)
        d = by_id[doc_id]
        snippet = d.content[:220]
        out.append(
            {
                "id": doc_id,
                "title": d.title or (d.content[:60] + ("…" if len(d.content) > 60 else "")),
                "url": d.url,
                "score": float(round(score, 6)),
                "snippet": snippet,
            }
        )

    # Fill with zero-score docs to satisfy top_k
    if len(out) < top_k:
        for d in ordered:
            if d.id in seen:
                continue
            out.append(
                {
                    "id": d.id,
                    "title": d.title or (d.content[:60] + ("…" if len(d.content) > 60 else "")),
                    "url": d.url,
                    "score": 0.0,
                    "snippet": d.content[:220],
                }
            )
            if len(out) >= top_k:
                break

    return out
