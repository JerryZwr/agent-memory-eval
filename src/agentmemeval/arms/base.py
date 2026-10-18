"""Common interface for memory systems under test ("arms", METHODOLOGY 1.1).

Lifecycle, mirroring the pipeline stages:

* `add(dataset)` ingests the corpus and returns a JSON-able ingest summary,
  which the ingest stage writes to disk;
* `attach(dataset, summary)` makes a fresh arm instance searchable using only
  that summary (local arms rebuild deterministically; service-backed arms
  reconnect to the recorded namespace). This is what lets the search stage be
  re-run on its own;
* `search(query, top_k)` returns ranked items.

Infrastructure failures must be raised as `InfraError` subclasses so the
pipeline can classify them as `env_failure`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from agentmemeval.data.models import Dataset


class RetrievedItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    date: str
    time: str
    group: str
    speaker: str
    text: str
    score: float


class MemoryArm(ABC):
    name: ClassVar[str]

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """JSON-able parameters; part of `arm_config_hash`."""

    @abstractmethod
    def add(self, dataset: Dataset) -> dict[str, Any]:
        """Ingest the corpus; return a JSON-able summary for the ingest stage file."""

    @abstractmethod
    def attach(self, dataset: Dataset, summary: dict[str, Any]) -> None:
        """Prepare this instance for search from a previously written ingest summary."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        """Ranked results, best first. Empty if nothing relevant."""
