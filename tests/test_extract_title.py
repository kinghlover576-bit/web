from __future__ import annotations

from processing.text import extract_title


def test_extract_title_from_meta():
    html = '<html><head><meta property="og:title" content="OG Title"/></head><body></body></html>'
    assert extract_title(html) == "OG Title"


def test_extract_title_from_h1_when_no_title():
    html = "<html><body><h1>Heading Title</h1><p>Body text</p></body></html>"
    assert extract_title(html) == "Heading Title"
