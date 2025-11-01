from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class Item:
    id: str
    vec: np.ndarray  # shape (D,)


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.items: list[Item] = []

    def add(self, ids: Iterable[str], vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        for i, vid in enumerate(ids):
            v = vectors[i]
            # ensure normalized
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            self.items.append(Item(vid, v.astype(np.float32)))

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if vector.ndim == 2:
            q = vector[0]
        else:
            q = vector
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        if not self.items:
            return []
        mat = np.stack([it.vec for it in self.items], axis=0)  # (N,D)
        sims = mat @ q  # cosine due to normalization
        idx = np.argsort(-sims)[:top_k]
        out: list[tuple[str, float]] = []
        for i in idx.tolist():
            out.append((self.items[i].id, float(sims[i])))
        return out
