from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and not p.isspace()]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t]


def summarize(query: str, docs: Iterable[str], max_sentences: int = 5) -> str:
    """Simple extractive summarization by scoring sentences against the query.

    - Tokenizes query and candidate sentences
    - Scores by term frequency weighted with IDF over the provided docs
    - Returns the top-N sentences joined in rank order
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        q_tokens = []

    # Build IDF over all docs
    N = 0
    df: Counter[str] = Counter()
    candidate_sents: list[str] = []
    for doc in docs:
        N += 1
        sents = _sentences(doc)
        candidate_sents.extend(sents)
        seen_terms = set(_tokenize(doc))
        for t in seen_terms:
            df[t] += 1

    if N == 0 or not candidate_sents:
        return ""

    def idf(t: str) -> float:
        return math.log((1 + N) / (1 + df.get(t, 0))) + 1.0

    def score_sent(s: str) -> float:
        toks = _tokenize(s)
        if not toks:
            return 0.0
        tf = Counter(toks)
        score = 0.0
        # emphasize query terms; if query empty, fallback to average IDF
        if q_tokens:
            for t in q_tokens:
                score += tf.get(t, 0) * idf(t)
        else:
            for t, c in tf.items():
                score += c * idf(t)
        # normalize by length to avoid bias toward long sentences
        return score / (len(toks) ** 0.5)

    ranked = sorted(candidate_sents, key=score_sent, reverse=True)[:max_sentences]
    return _norm(" ".join(ranked))


def research_digest(
    query: str, docs: Iterable[tuple[str, str, str | None]], *, max_highlights: int = 7
) -> dict:
    """Produce a lightweight structured research output.

    docs: iterable of (url, title, content)
    returns: {summary, highlights[], citations[]}
    """
    contents = [c or "" for _u, _t, c in docs]
    summary = summarize(query, contents, max_sentences=5)

    # pick top sentences from each doc as highlights
    highlights: list[str] = []
    citations: list[dict] = []
    for url, title, content in docs:
        sents = _sentences(content or "")
        if not sents:
            continue
        highlights.append(sents[0][:240])
        citations.append({"url": url, "title": title, "snippet": (sents[0][:240])})
        if len(highlights) >= max_highlights:
            break

    return {"summary": summary, "highlights": highlights, "citations": citations}
