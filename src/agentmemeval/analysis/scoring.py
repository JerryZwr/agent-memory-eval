"""Counting outcomes and verdicts into the two METHODOLOGY 1.4 metrics.

* answer accuracy   = correct / delivered
* end-to-end success = correct / (delivered + undelivered)

`env_failure` (and `agent_error`) enter neither denominator and are reported on
their own. A delivered answer without a verdict is refused, not skipped:
silently dropping it would shrink the denominator.

Intervals and significance tests are M2 (`analysis/stats.py`); this module
only counts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from agentmemeval.analysis.outcome import Outcome
from agentmemeval.errors import StageError
from agentmemeval.pipeline.records import JudgeRecord


@dataclass(frozen=True)
class Ratio:
    num: int
    den: int

    @property
    def value(self) -> float | None:
        return self.num / self.den if self.den else None

    def __str__(self) -> str:
        v = self.value
        return f"{self.num}/{self.den} = {v:.0%}" if v is not None else f"{self.num}/0 = n/a"


@dataclass(frozen=True)
class Tally:
    n: int = 0
    delivered: int = 0
    undelivered: int = 0
    env_failure: int = 0
    agent_error: int = 0
    correct: int = 0
    ambiguous_extraction: int = 0

    @property
    def accuracy(self) -> Ratio:
        return Ratio(self.correct, self.delivered)

    @property
    def end_to_end(self) -> Ratio:
        return Ratio(self.correct, self.delivered + self.undelivered)


def tally(records: Iterable[JudgeRecord]) -> Tally:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts["n"] += 1
        counts[r.outcome.value] += 1
        if r.outcome is Outcome.DELIVERED:
            if r.verdict is None:
                raise StageError(f"{r.qid}: delivered but not judged; refusing to score")
            counts["correct"] += r.verdict == "CORRECT"
            counts["ambiguous_extraction"] += r.ambiguous_extraction
        elif r.verdict is not None:
            raise StageError(f"{r.qid}: has a verdict but outcome is {r.outcome.value}")
    return Tally(**counts)


def tally_by(records: Iterable[JudgeRecord], key: Callable[[JudgeRecord], str]) -> dict[str, Tally]:
    groups: dict[str, list[JudgeRecord]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return {k: tally(v) for k, v in sorted(groups.items())}
