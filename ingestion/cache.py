from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from ingestion.page_loader import DEFAULT_UA, fetch_url


def _cache_dir() -> Path:
    root = Path(os.getenv("OMNIVERSE_CACHE_DIR", ".omniverse_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _path_for(url: str) -> Path:
    return _cache_dir() / f"{_key(url)}.json"


def load_cached(url: str, *, max_age_hours: int = 24) -> tuple[int, str, str | None] | None:
    p = _path_for(url)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    fetched_at = float(data.get("fetched_at", 0))
    if max_age_hours >= 0:
        age_h = (time.time() - fetched_at) / 3600.0
        if age_h > max_age_hours:
            return None
    return int(data.get("status", 0)), str(data.get("text", "")), data.get("content_type")


def save_cache(url: str, status: int, text: str, content_type: str | None) -> None:
    p = _path_for(url)
    payload = {
        "url": url,
        "status": status,
        "text": text,
        "content_type": content_type,
        "fetched_at": time.time(),
    }
    p.write_text(json.dumps(payload), encoding="utf-8")


async def fetch_url_cached(
    url: str,
    *,
    max_age_hours: int = 24,
    live_fetch: bool = False,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_UA,
    fetch: Callable[[str], Awaitable[tuple[int, str, str | None]]] | None = None,
) -> tuple[int, str, str | None]:
    if not live_fetch:
        cached = load_cached(url, max_age_hours=max_age_hours)
        if cached is not None:
            return cached

    fetcher = fetch or (lambda u: fetch_url(u, timeout=timeout, user_agent=user_agent))
    status, text, ctype = await fetcher(url)
    try:
        save_cache(url, status, text, ctype)
    except Exception:
        # best-effort cache
        pass
    return status, text, ctype
