from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - handled at runtime
    faiss = None


@dataclass
class SearchHit:
    int_id: int
    score: float


class FaissStore:
    """Thin wrapper around FAISS with ID mapping and cosine similarity."""

    def __init__(self, dim: int) -> None:
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not available; install faiss-cpu")
        self.dim = dim
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: Iterable[int], vectors: np.ndarray) -> None:
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        assert vectors.shape[0] == ids_arr.shape[0]
        vecs = self._normalize(vectors.astype(np.float32))
        self.index.add_with_ids(vecs, ids_arr)

    def remove(self, ids: Iterable[int]) -> None:
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        sel = faiss.IDSelectorBatch(ids_arr.size, faiss.swig_ptr(ids_arr))
        self.index.remove_ids(sel)

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if vector.ndim == 1:
            q = vector[None, :]
        else:
            q = vector
        qn = self._normalize(q.astype(np.float32))
        dists, inds = self.index.search(qn, top_k)
        hits: list[SearchHit] = []
        for i, d in zip(inds[0].tolist(), dists[0].tolist(), strict=False):
            if i == -1:
                continue
            hits.append(SearchHit(int_id=int(i), score=float(d)))
        return hits

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))

    @classmethod
    def load(cls, path: str | Path) -> FaissStore:
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not available; install faiss-cpu")
        index = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.dim = int(index.d)
        obj.index = index
        return obj

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            norm = float(np.linalg.norm(v)) or 1.0
            return (v / norm).astype(np.float32)
        norms: np.ndarray = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (v / norms).astype(np.float32)
