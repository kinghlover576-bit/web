from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from embeddings.cache import embed_with_cache
from embeddings.hashing import embed_texts
from index.tfidf import TfidfIndex
from search.pipeline import Doc
from vector.store import InMemoryVectorStore


@dataclass
class HybridConfig:
    dim: int = 256
    alpha: float = 0.5  # weight for embedding score; (1-alpha) for tf-idf
    provider: str = "hashed"  # "hashed" | "sbert"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str | None = None


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
    if cfg.provider == "sbert":
        try:
            from embeddings.sbert import embed_texts as sbert_embed

            def compute(batch: list[str]):
                return sbert_embed(batch, model_name=cfg.model_name)

            cache_ns = f"sbert:{cfg.model_name}"
            vectors = embed_with_cache(
                [d.content for d in ordered], cache_ns, compute, cache_dir=cfg.cache_dir
            )
            q_vec = embed_with_cache([query], cache_ns, compute, cache_dir=cfg.cache_dir)
            dim = int(vectors.shape[1]) if vectors.size else 0
        except Exception:
            # Fallback to hashed if sbert unavailable
            vectors = embed_texts([d.content for d in ordered], dim=cfg.dim)
            q_vec = embed_texts([query], dim=cfg.dim)
            dim = cfg.dim
    else:
        vectors = embed_texts([d.content for d in ordered], dim=cfg.dim)
        q_vec = embed_texts([query], dim=cfg.dim)
        dim = cfg.dim

    store = InMemoryVectorStore(dim=dim if dim else cfg.dim)
    v_scores = dict(store.search(q_vec, top_k=len(ordered)))
    if store.items == [] and vectors.size:
        # Add after init if not added yet
        store.add([d.id for d in ordered], vectors)
        v_scores = dict(store.search(q_vec, top_k=len(ordered)))
    else:
        store.add([d.id for d in ordered], vectors)
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
