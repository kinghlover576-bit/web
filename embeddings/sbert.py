from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_texts(texts: Iterable[str], *, model_name: str | None = None) -> np.ndarray:
    name = model_name or DEFAULT_MODEL
    model = _load_model(name)
    # Returns float32; normalize for cosine
    vecs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)
