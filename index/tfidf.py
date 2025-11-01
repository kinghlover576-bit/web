from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Doc:
    doc_id: str
    text: str


class TfidfIndex:
    def __init__(self) -> None:
        self.docs: list[Doc] = []
        self.df: dict[str, int] = defaultdict(int)
        self.doc_tf: list[Counter[str]] = []
        self.doc_vecs: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        self.docs.append(Doc(doc_id, text))
        self.doc_tf.append(tf)
        for term in tf:
            self.df[term] += 1

    def build(self) -> None:
        n = len(self.docs)
        # idf with smoothing to avoid div by zero
        self.idf = {term: math.log(1.0 + n / (1.0 + df)) for term, df in self.df.items()}

        self.doc_vecs = []
        for tf in self.doc_tf:
            vec: dict[str, float] = {}
            for term, freq in tf.items():
                idf = self.idf.get(term, 0.0)
                vec[term] = (1.0 + math.log(freq)) * idf
            # l2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for k in vec:
                vec[k] /= norm
            self.doc_vecs.append(vec)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_tf = Counter(tokenize(query))
        q_vec: dict[str, float] = {}
        for term, freq in q_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            q_vec[term] = (1.0 + math.log(freq)) * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        for k in list(q_vec.keys()):
            q_vec[k] /= q_norm

        scores: list[tuple[int, float]] = []
        for i, dvec in enumerate(self.doc_vecs):
            # dot product over intersection
            s = 0.0
            for term, qv in q_vec.items():
                dv = dvec.get(term)
                if dv:
                    s += qv * dv
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i].doc_id, score) for i, score in scores[:top_k]]
