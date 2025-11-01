from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def embed_with_cache(
    texts: Iterable[str],
    cache_namespace: str,
    compute: Callable[[list[str]], np.ndarray],
    *,
    cache_dir: str | os.PathLike | None = None,
) -> np.ndarray:
    """
    Cache per-text embeddings as .npy files under {cache_dir}/{cache_namespace}/sha1(text).npy.
    Returns stacked array matching the input order.
    """
    texts_list = list(texts)
    if not texts_list:
        return np.zeros((0, 0), dtype=np.float32)

    if cache_dir:
        root = Path(cache_dir)
    else:
        root = Path(os.getenv("OMNIVERSE_EMBED_CACHE_DIR", ".omniverse_cache/embeddings"))
    ns = root / cache_namespace
    ns.mkdir(parents=True, exist_ok=True)

    vectors: list[np.ndarray | None] = [None] * len(texts_list)
    missing_idx: list[int] = []
    missing_texts: list[str] = []

    # Load cached
    for i, t in enumerate(texts_list):
        p = ns / f"{_sha1(t)}.npy"
        if p.exists():
            try:
                vectors[i] = np.load(p)
                continue
            except Exception:
                # Treat as missing on load failure
                pass
        missing_idx.append(i)
        missing_texts.append(t)

    # Compute and save missing
    if missing_texts:
        new_vecs = compute(missing_texts)
        assert new_vecs.shape[0] == len(missing_texts)
        for j, i in enumerate(missing_idx):
            v = new_vecs[j]
            vectors[i] = v
            np.save(ns / f"{_sha1(texts_list[i])}.npy", v)

    # All vectors should be present now
    out = np.stack([v for v in vectors if v is not None], axis=0)
    return out.astype(np.float32)
