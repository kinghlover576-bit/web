from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from embeddings.hashing import embed_texts
from index.tfidf import TfidfIndex
from search.pipeline import Doc
from vector.store import InMemoryVectorStore


@dataclass
class HybridConfig:
    dim: int = 256
    alpha: float = 0.5  # weight for embedding score; (1-alpha) for tf-idf


def hybrid_search(
    docs: Iterable[Doc], query: str, *, top_k: int = 5, cfg: HybridConfig | None = None
) -> list[dict]:
    cfg = cfg or HybridConfig()

    ordered: list[Doc] = list(docs)

    # Build TF-IDF
    tf = TfidfIndex()
    for d in ordered:
        tf.add_document(d.id, d.content)
    tf.build()
    tf_scores = dict(tf.search(query, top_k=len(ordered)))

    # Build vector store
    store = InMemoryVectorStore(dim=cfg.dim)
    vectors = embed_texts([d.content for d in ordered], dim=cfg.dim)
    store.add([d.id for d in ordered], vectors)
    q_vec = embed_texts([query], dim=cfg.dim)
    v_scores = dict(store.search(q_vec, top_k=len(ordered)))

    # Combine
    alpha = float(cfg.alpha)
    combined: list[tuple[str, float]] = []
    for d in ordered:
        s_t = tf_scores.get(d.id, 0.0)
        s_v = v_scores.get(d.id, 0.0)
        s = alpha * s_v + (1.0 - alpha) * s_t
        combined.append((d.id, s))
    combined.sort(key=lambda x: x[1], reverse=True)

    by_id = {d.id: d for d in ordered}
    out: list[dict] = []
    for doc_id, score in combined[:top_k]:
        d = by_id[doc_id]
        out.append(
            {
                "id": d.id,
                "title": d.title or (d.content[:60] + ("…" if len(d.content) > 60 else "")),
                "url": d.url,
                "score": float(round(score, 6)),
                "snippet": d.content[:220],
            }
        )
    return out
