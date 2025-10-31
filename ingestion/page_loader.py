from __future__ import annotations

import httpx

DEFAULT_TIMEOUT = 10.0
DEFAULT_UA = "OmniVerseBot/0.1 (+https://github.com/kinghlover576-bit/web)"


async def fetch_url(
    url: str, *, timeout: float = DEFAULT_TIMEOUT, user_agent: str = DEFAULT_UA
) -> tuple[int, str, str | None]:
    """
    Fetch a URL and return (status_code, text, content_type).

    Designed for short-lived API calls; not a full crawler. No robots handling.
    """
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = await client.get(url)
        ctype = resp.headers.get("content-type")
        return resp.status_code, resp.text, ctype
