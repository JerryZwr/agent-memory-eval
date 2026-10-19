"""Lightweight span tracer (METHODOLOGY 2.2).

Span kinds: `run` / `route` / `llm_call` / `tool_call`. Each span records start
and end time, its parent, attributes (model, tool name, argument summary,
result count, error, TTFT for streamed LLM calls) and exports to JSON.

The clock is injectable. `RealClock` measures wall time; `FakeClock` is a
virtual clock that components *advance* to model latency, which makes latency
breakdowns exact and testable offline. A report built on a FakeClock must say
its timings are simulated.

`check_integrity` validates the observability data itself (dimension 8): a
trace that silently lost a span would make every latency number built on it
wrong without anyone noticing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

SpanKind = Literal["run", "route", "llm_call", "tool_call"]
SPAN_KINDS: tuple[str, ...] = ("run", "route", "llm_call", "tool_call")


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Virtual time: `sleep` advances it instantly."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot sleep a negative duration")
        self._t += seconds


@dataclass
class Span:
    span_id: int
    parent_id: int | None
    kind: SpanKind
    name: str
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration(self) -> float:
        if self.end is None:
            raise ValueError(f"span {self.span_id} ({self.name}) is not closed")
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "attributes": self.attributes,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        return cls(**d)


@dataclass
class Trace:
    spans: list[Span]

    def to_dict(self) -> dict[str, Any]:
        return {"spans": [s.to_dict() for s in self.spans]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        return cls([Span.from_dict(s) for s in d["spans"]])

    def of_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    @property
    def root(self) -> Span:
        roots = [s for s in self.spans if s.parent_id is None]
        if len(roots) != 1:
            raise ValueError(f"trace has {len(roots)} root spans, expected 1")
        return roots[0]


class Tracer:
    def __init__(self, clock: Clock | None = None):
        self.clock = clock or RealClock()
        self._spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, kind: SpanKind, name: str, **attributes: Any) -> Iterator[Span]:
        if kind not in SPAN_KINDS:
            raise ValueError(f"unknown span kind {kind!r}")
        span = Span(
            span_id=len(self._spans) + 1,
            parent_id=self._stack[-1].span_id if self._stack else None,
            kind=kind,
            name=name,
            start=self.clock.now(),
            attributes=dict(attributes),
        )
        self._spans.append(span)
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = self.clock.now()
            self._stack.pop()

    @property
    def trace(self) -> Trace:
        return Trace(list(self._spans))


# --------------------------------------------------------------------------- analysis


def check_integrity(
    trace: Trace, *, expected_llm_calls: int, expected_tool_calls: int
) -> list[str]:
    """Problems with the trace itself; [] means it can be trusted.

    * exactly one root `run` span; every parent exists;
    * every span is closed and ends after it starts;
    * children lie inside their parent's time range;
    * span counts equal the steps the agent actually took.
    """
    problems: list[str] = []
    by_id = {s.span_id: s for s in trace.spans}
    roots = [s for s in trace.spans if s.parent_id is None]
    if len(roots) != 1 or roots[0].kind != "run":
        problems.append(f"expected one root 'run' span, found {[r.kind for r in roots]}")
    for s in trace.spans:
        if s.end is None:
            problems.append(f"span {s.span_id} ({s.kind} {s.name}) was never closed")
            continue
        if s.end < s.start:
            problems.append(f"span {s.span_id} ends before it starts")
        if s.parent_id is not None:
            parent = by_id.get(s.parent_id)
            if parent is None:
                problems.append(f"span {s.span_id} has a missing parent {s.parent_id}")
            elif parent.end is not None and not (parent.start <= s.start and s.end <= parent.end):
                problems.append(f"span {s.span_id} ({s.name}) lies outside its parent's time range")
    n_llm = len(trace.of_kind("llm_call")) + len(trace.of_kind("route"))
    n_tool = len(trace.of_kind("tool_call"))
    if n_llm != expected_llm_calls:
        problems.append(f"{n_llm} LLM spans (route + llm_call) for {expected_llm_calls} LLM calls")
    if n_tool != expected_tool_calls:
        problems.append(f"{n_tool} tool spans for {expected_tool_calls} tool calls")
    return problems


@dataclass(frozen=True)
class LatencyBreakdown:
    total_ms: float
    llm_ms: float
    tool_ms: float
    framework_ms: float  # total - llm - tool: scheduling, context assembly, serialisation
    ttft_ms: tuple[float, ...]  # one per streamed LLM call

    def shares(self) -> dict[str, float]:
        if self.total_ms <= 0:
            return {"llm": 0.0, "tool": 0.0, "framework": 0.0}
        return {
            "llm": self.llm_ms / self.total_ms,
            "tool": self.tool_ms / self.total_ms,
            "framework": self.framework_ms / self.total_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "llm_ms": self.llm_ms,
            "tool_ms": self.tool_ms,
            "framework_ms": self.framework_ms,
            "ttft_ms": list(self.ttft_ms),
        }


def latency_breakdown(trace: Trace) -> LatencyBreakdown:
    """Split end-to-end time into llm / tool / framework (METHODOLOGY 2.3 dimension 9).

    Only direct children of the run span are summed, so nested spans are never
    counted twice. `framework` is the remainder: it is the share that a
    two-way llm/tool split would silently hide.
    """
    root = trace.root
    children = [s for s in trace.spans if s.parent_id == root.span_id]
    llm = sum(s.duration for s in children if s.kind in ("llm_call", "route"))
    tool = sum(s.duration for s in children if s.kind == "tool_call")
    total = root.duration
    ttft = tuple(
        s.attributes["ttft_s"] * 1000
        for s in children
        if s.kind in ("llm_call", "route") and s.attributes.get("ttft_s") is not None
    )
    to_ms = 1000.0
    return LatencyBreakdown(
        total_ms=round(total * to_ms, 3),
        llm_ms=round(llm * to_ms, 3),
        tool_ms=round(tool * to_ms, 3),
        framework_ms=round((total - llm - tool) * to_ms, 3),
        ttft_ms=tuple(round(t, 3) for t in ttft),
    )
