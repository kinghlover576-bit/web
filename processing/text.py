from __future__ import annotations

import re

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = _WS.sub(" ", text).strip()
    return text
