from __future__ import annotations

from pathlib import Path

from index.persistent import PersistentFaissIndex
from search.pipeline import Doc


def test_persistent_faiss_index_roundtrip(tmp_path: Path):
    root = tmp_path / "store"
    idx = PersistentFaissIndex(root_dir=str(root), provider="hashed", dim=16)

    docs = [
        Doc(id="d1", title="Doc 1", url=None, content="the quick brown fox"),
        Doc(id="d2", title="Doc 2", url=None, content="jumps over the lazy dog"),
    ]
    added = idx.upsert(docs)
    assert added == 2

    # Files persisted
    assert (root / "vectors.faiss").exists()
    assert (root / "meta.json").exists()

    # Stats
    s = idx.stats()
    assert s["count"] == 2
    assert s["dim"] == 16

    # Search should return at least one
    r = idx.search("quick fox", top_k=1)
    assert r and r[0]["id"] in {"d1", "d2"}

    # Reload from disk
    idx2 = PersistentFaissIndex(root_dir=str(root))
    s2 = idx2.stats()
    assert s2["count"] == 2
    r2 = idx2.search("lazy dog", top_k=1)
    assert r2 and r2[0]["id"] in {"d1", "d2"}

    # Upsert changing content should not duplicate entries
    docs2 = [Doc(id="d2", title="Doc 2", url=None, content="lazy dog sleeping")]
    added2 = idx2.upsert(docs2)
    assert added2 == 1
    s3 = idx2.stats()
    assert s3["count"] == 2  # still two docs total
