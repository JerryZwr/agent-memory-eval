"""Local BM25 keyword retrieval, one chunk per message (METHODOLOGY 1.1).

Zero cost and deterministic, so CI and the demo can run a real retrieval
baseline.

Scoring is Lucene's BM25:

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(d, q) = sum over distinct t in q of
                  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

This is implemented here rather than taken from `rank_bm25`, whose Okapi idf,
ln((N - n + 0.5) / (n + 0.5)), is zero for a term in exactly half the
documents and negative above that. On small corpora (the fixture, or a short
evidence window) that silently drops or penalises genuine matches. The Lucene
idf is always positive, so a document scores > 0 exactly when it contains a
query term, and only such documents are returned. Ties are broken by message id.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from agentmemeval.arms.base import MemoryArm, RetrievedItem
from agentmemeval.data.models import Dataset, Message
from agentmemeval.errors import DatasetError, StageError
from agentmemeval.text import TOKENIZER_VERSION, tokenize

SCORING = "lucene-bm25"


class BM25Index:
    """Inverted index over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], *, k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, terms: list[str]) -> dict[int, float]:
        """Scores of documents containing at least one query term (all > 0)."""
        out: dict[int, float] = defaultdict(float)
        for term in set(terms):
            for i, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return dict(out)


def _document(message: Message, include_speaker: bool) -> str:
    return f"{message.speaker}: {message.text}" if include_speaker else message.text


class BM25Arm(MemoryArm):
    name = "bm25"

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, include_speaker: bool = True):
        self.k1 = k1
        self.b = b
        self.include_speaker = include_speaker
        self._messages: tuple[Message, ...] = ()
        self._index: BM25Index | None = None

    def params(self) -> dict[str, Any]:
        return {
            "scoring": SCORING,
            "k1": self.k1,
            "b": self.b,
            "include_speaker": self.include_speaker,
            "tokenizer": TOKENIZER_VERSION,
            "chunking": "message",
        }

    def _build(self, dataset: Dataset) -> None:
        if not dataset.messages:
            raise DatasetError("bm25: cannot index an empty corpus")
        self._messages = dataset.messages
        docs = [tokenize(_document(m, self.include_speaker)) for m in self._messages]
        self._index = BM25Index(docs, k1=self.k1, b=self.b)

    def add(self, dataset: Dataset) -> dict[str, Any]:
        self._build(dataset)
        return {
            "n_messages_indexed": len(self._messages),
            "message_ids_sha256": dataset.message_ids_sha256,
        }

    def attach(self, dataset: Dataset, summary: dict[str, Any]) -> None:
        if summary.get("message_ids_sha256") != dataset.message_ids_sha256:
            raise StageError(
                "bm25: the corpus differs from the one recorded at ingest; re-run ingest"
            )
        self._build(dataset)

    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        if self._index is None:
            raise RuntimeError("bm25: search() before add() or attach()")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        scores = self._index.scores(tokenize(query))
        ranked = sorted(scores, key=lambda i: (-scores[i], self._messages[i].id))
        return [
            RetrievedItem(
                message_id=m.id,
                date=m.date,
                time=m.time,
                group=m.group,
                speaker=m.speaker,
                text=m.text,
                score=round(scores[i], 6),
            )
            for i in ranked[:top_k]
            for m in (self._messages[i],)
        ]
