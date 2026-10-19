"""The control arm: stores nothing, always retrieves nothing (METHODOLOGY 1.1).

Its score is what the answering model gets from general knowledge, question
wording and multiple-choice guessing alone. Every other arm reads as "distance
from null".
"""

from __future__ import annotations

from typing import Any

from agentmemeval.arms.base import MemoryArm, RetrievedItem
from agentmemeval.data.models import Dataset


class NullArm(MemoryArm):
    name = "null"
    stores_data = False

    def params(self) -> dict[str, Any]:
        return {}

    def add(self, dataset: Dataset) -> dict[str, Any]:
        return {"n_messages_offered": len(dataset.messages), "n_messages_stored": 0}

    def attach(self, dataset: Dataset, summary: dict[str, Any]) -> None:
        return None

    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        return []
