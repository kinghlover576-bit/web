from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from embeddings.cache import embed_with_cache
from embeddings.hashing import embed_texts as hash_embed
from search.pipeline import Doc
from vector.faiss_store import FaissStore


@dataclass
class DocMeta:
    id: str
    title: str | None
    url: str | None
    content: str


class PersistentFaissIndex:
    def __init__(
        self,
        root_dir: str | os.PathLike[str] | None = None,
        *,
        provider: str = "hashed",
        model_name: str | None = None,
        dim: int = 256,
    ) -> None:
        env_default = os.getenv("OMNIVERSE_INDEX_DIR") or ".omniverse_store/faiss"
        self.root = Path(root_dir) if root_dir is not None else Path(env_default)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "vectors.faiss"
        self.meta_path = self.root / "meta.json"
        self.cfg_path = self.root / "config.json"

        self.provider = provider
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.dim = dim

        # mappings
        self.str2int: dict[str, int] = {}
        self.docs: dict[int, DocMeta] = {}
        self.next_int: int = 1

        self.store: FaissStore | None = None
        self._load()

    def _load(self) -> None:
        # meta
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text())
            self.str2int = {str(k): int(v) for k, v in data.get("str2int", {}).items()}
            self.docs = {int(k): DocMeta(**v) for k, v in data.get("docs", {}).items()}
            self.next_int = int(data.get("next_int", 1))

        # store
        if self.index_path.exists():
            self.store = FaissStore.load(self.index_path)
            self.dim = self.store.dim
        else:
            self.store = None

    def _save_meta(self) -> None:
        data = {
            "str2int": self.str2int,
            "docs": {str(k): asdict(v) for k, v in self.docs.items()},
            "next_int": self.next_int,
            "provider": self.provider,
            "model_name": self.model_name,
            "dim": self.dim,
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(data))

    def _ensure_store(self, dim: int | None = None) -> None:
        if self.store is None:
            d = dim or self.dim
            self.store = FaissStore(d)
            self.dim = d

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        if self.provider == "sbert":
            from embeddings.sbert import embed_texts as sbert_embed

            def compute(batch: list[str]):
                return sbert_embed(batch, model_name=self.model_name)

            return embed_with_cache(texts, f"sbert:{self.model_name}", compute)
        else:
            return hash_embed(texts, dim=self.dim)

    def upsert(self, docs: Iterable[Doc]) -> int:
        docs_list = list(docs)
        if not docs_list:
            return 0

        # Handle existing: collect removals
        to_remove: list[int] = []
        to_add: list[int] = []
        metas: list[DocMeta] = []
        texts: list[str] = []
        for d in docs_list:
            existing = self.str2int.get(d.id)
            if existing is not None:
                to_remove.append(existing)
                int_id = existing
            else:
                int_id = self.next_int
                self.next_int += 1
            self.str2int[d.id] = int_id
            to_add.append(int_id)
            metas.append(DocMeta(id=d.id, title=d.title, url=d.url, content=d.content))
            texts.append(d.content)

        vecs = self._compute_embeddings(texts)
        # Ensure store initialized with correct dim
        self._ensure_store(dim=int(vecs.shape[1]) if vecs.ndim == 2 else self.dim)

        if to_remove and self.store is not None:
            self.store.remove(to_remove)

        if self.store is not None and vecs.size:
            self.store.add(to_add, vecs)
            self.store.save(self.index_path)

        # Update metas
        for int_id, meta in zip(to_add, metas, strict=False):
            self.docs[int_id] = meta
        self._save_meta()
        return len(docs_list)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.store is None or self.store.ntotal == 0:
            return []
        qv = self._compute_embeddings([query])
        hits = self.store.search(qv, top_k=top_k)
        out: list[dict] = []
        for h in hits:
            meta = self.docs.get(h.int_id)
            if not meta:
                continue
            snippet = meta.content[:220]
            title = meta.title or (meta.content[:60] + ("…" if len(meta.content) > 60 else ""))
            out.append(
                {
                    "id": meta.id,
                    "title": title,
                    "url": meta.url,
                    "score": float(round(h.score, 6)),
                    "snippet": snippet,
                }
            )
        return out

    def stats(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "dim": int(self.dim),
            "count": int(self.store.ntotal) if self.store else 0,
            "path": str(self.root),
        }


_SINGLETONS: dict[str, PersistentFaissIndex] = {}


def get_index(
    *,
    provider: str = "hashed",
    model_name: str | None = None,
    dim: int = 256,
    root_dir: str | None = None,
) -> PersistentFaissIndex:
    key = f"{root_dir or ''}|{provider}|{model_name or ''}|{dim}"
    inst = _SINGLETONS.get(key)
    if inst is None:
        inst = PersistentFaissIndex(root_dir, provider=provider, model_name=model_name, dim=dim)
        _SINGLETONS[key] = inst
    return inst
