"""Memory systems under test. `build_arm` is the only place that maps names to classes."""

from __future__ import annotations

from typing import Any

from agentmemeval.arms.base import MemoryArm, RetrievedItem
from agentmemeval.arms.bm25 import BM25Arm
from agentmemeval.arms.null import NullArm

ARMS: dict[str, type[MemoryArm]] = {NullArm.name: NullArm, BM25Arm.name: BM25Arm}


def build_arm(name: str, params: dict[str, Any] | None = None) -> MemoryArm:
    try:
        cls = ARMS[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; known: {sorted(ARMS)}") from None
    return cls(**(params or {}))


__all__ = ["ARMS", "BM25Arm", "MemoryArm", "NullArm", "RetrievedItem", "build_arm"]
