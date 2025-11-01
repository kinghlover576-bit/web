from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from index.tfidf import tokenize


def _hash32(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def embed_texts(texts: Iterable[str], dim: int = 256) -> np.ndarray:
    """
    Deterministic hashed bag-of-words embeddings.
    - Tokenize like TF-IDF
    - Hash tokens into `dim` bins
    - L2-normalize
    Returns float32 array of shape (N, dim)
    """
    mats = []
    for t in texts:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in tokenize(t):
            pos = _hash32(tok) % dim
            vec[pos] += 1.0
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        mats.append(vec)
    if not mats:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(mats, axis=0)
