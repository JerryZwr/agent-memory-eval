"""Tokenisation shared by the BM25 arm and the scripted reader.

Versioned: changing the rules changes retrieval results, so bump
`TOKENIZER_VERSION` (it is part of the BM25 arm config and thus its hash).
"""

from __future__ import annotations

import re

TOKENIZER_VERSION = "simple-v1"

_TOKEN_RE = re.compile(r"[0-9a-z]+")

# Small English stopword list: function words plus question words ("who",
# "when", ...), which appear in almost every question but name nothing that a
# chat message could match on. Kept short on purpose; content words stay.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her his how i if in into is it its me my of on or our she so than that the
    their them then there these they this to was we were what when where which who
    whom why will with would you your about after before also any all just not no
    yes up out over please let us
    """.split()  # noqa: SIM905 - a word list reads better than a list literal
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed. Order and repeats kept."""
    return [t for t in _TOKEN_RE.findall(text.casefold()) if t not in STOPWORDS]


def normalize(text: str) -> str:
    """Collapse whitespace and casefold, for exact-text joins and containment checks."""
    return " ".join(text.split()).casefold()
