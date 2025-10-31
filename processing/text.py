from __future__ import annotations

import re

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return _norm(text)


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    # 1) OpenGraph/Twitter/meta title
    for attr, val in (
        ("property", "og:title"),
        ("name", "og:title"),
        ("name", "twitter:title"),
        ("name", "title"),
    ):
        tag = soup.find("meta", attrs={attr: val})
        if tag:
            content = tag.get("content")
            if content:
                t = _norm(content)
                if t:
                    return t

    # 2) <title>
    if soup.title and soup.title.string:
        t = _norm(str(soup.title.string))
        if t:
            return t

    # 3) First h1/h2/h3
    for name in ("h1", "h2", "h3"):
        tag = soup.find(name)
        if tag:
            t = _norm(tag.get_text(" "))
            if t:
                return t

    # 4) Fallback: first 80 chars of visible text
    txt = html_to_text(html)
    if txt:
        return txt[:80]
    return None
