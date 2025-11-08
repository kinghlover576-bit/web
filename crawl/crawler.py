from __future__ import annotations

import asyncio
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

from ingestion.page_loader import DEFAULT_UA, fetch_url
from processing.text import extract_title, html_to_text

HREF_RE = re.compile(r"href\s*=\s*\"([^\"]+)\"|href\s*=\s*'([^']+)'")


def _canon(url: str) -> str:
    p = urlparse(url)
    # strip fragment and normalize path
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme, p.netloc.lower(), path, "", p.query, ""))


def _same_host(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc.lower()) == (pb.scheme, pb.netloc.lower())


@dataclass
class Page:
    url: str
    status: int
    title: str | None
    content: str | None
    content_type: str | None
    error: str | None = None


async def crawl_site(
    start_url: str,
    *,
    max_pages: int = 100,
    max_depth: int = 2,
    concurrency: int = 8,
    timeout: float = 10.0,
    live_fetch: bool = True,  # reserved for future caching; currently unused
    respect_robots: bool = False,  # simple crawler, robots not enforced yet
    user_agent: str = DEFAULT_UA,
    fetch: Callable[[str], Awaitable[tuple[int, str, str | None]]] | None = None,
) -> list[Page]:
    """Crawl a site breadth-first starting from start_url and return fetched pages.

    - Restricts traversal to the same scheme+host as start_url
    - Parses links from <a href="..."> occurrences (no JS execution)
    - Extracts textual content and title for HTML pages
    - Designed to be testable: a custom `fetch` function can be injected
    """

    fetcher = fetch or (lambda u: fetch_url(u, timeout=timeout, user_agent=user_agent))

    origin = _canon(start_url)
    q: deque[tuple[str, int]] = deque([(origin, 0)])
    seen: set[str] = {origin}
    results: list[Page] = []
    sem = asyncio.Semaphore(concurrency)

    async def _get(u: str) -> tuple[int, str, str | None]:
        async with sem:
            return await fetcher(u)

    async def _visit(u: str, depth: int) -> None:
        try:
            status, body, ctype = await _get(u)
        except Exception as e:  # noqa: BLE001 - surface errors as part of result
            results.append(
                Page(url=u, status=0, title=None, content=None, content_type=None, error=str(e))
            )
            return

        title = None
        content_text = None
        if ctype and "html" in ctype.lower() and body:
            title = extract_title(body)
            content_text = html_to_text(body)
            # discover links for BFS
            if depth < max_depth:
                # light regex for anchor tags; BeautifulSoup not necessary for speed
                for m in HREF_RE.finditer(body):
                    href = m.group(1) or m.group(2)
                    if not href:
                        continue
                    abs_u = _canon(urljoin(u, href))
                    if not _same_host(origin, abs_u):
                        continue
                    if abs_u in seen:
                        continue
                    seen.add(abs_u)
                    q.append((abs_u, depth + 1))

        results.append(
            Page(url=u, status=status, title=title, content=content_text, content_type=ctype)
        )

    tasks: set[asyncio.Task[None]] = set()

    while q and len(results) + len(tasks) < max_pages:
        u, depth = q.popleft()
        tasks.add(asyncio.create_task(_visit(u, depth)))
        # opportunistically clean up finished tasks
        done, tasks = await asyncio.wait(tasks, timeout=0, return_when=asyncio.FIRST_COMPLETED)
        # ignore results here; _visit already records

    # wait for remaining tasks
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return results[:max_pages]
