"""The agent's tools: memory_search, web_search (a stub), and a fault-injection wrapper.

`final_answer` is not a tool object: the agent loop handles it as the delivery
action. Tool failures are returned to the loop as a `ToolResult` with an
error (never raised into the agent), so the policy can retry and the outcome
classification can see what happened:

* `infra`     - timeouts, unreachable service, 5xx: environment failures;
* `malformed` - the tool answered, but not in its declared shape;
* `unknown_tool` - the agent asked for a tool that does not exist: the agent's own
  mistake, so it is never counted as an environment failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from agentmemeval.arms.base import MemoryArm
from agentmemeval.errors import InfraError, MemoryServiceError
from agentmemeval.llm.base import ToolSpec
from agentmemeval.text import normalize
from agentmemeval.tracing import Clock, FakeClock

ErrorKind = Literal["infra", "malformed", "unknown_tool"]


class ToolTimeoutError(InfraError):
    """The tool did not answer within its timeout."""


@dataclass(frozen=True)
class ToolResult:
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_kind: ErrorKind | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Tool(ABC):
    name: str
    spec: ToolSpec

    @abstractmethod
    def raw_call(self, arguments: dict[str, Any]) -> Any:
        """Do the work. May raise InfraError; returns the tool's raw payload."""

    def __call__(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            payload = self.raw_call(arguments)
        except InfraError as exc:
            return ToolResult(error=f"{type(exc).__name__}: {exc}", error_kind="infra")
        return validate_payload(payload)


def validate_payload(payload: Any) -> ToolResult:
    """Tools return a list of objects, each with a string `text`; anything else is malformed."""
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) and isinstance(item.get("text"), str) for item in payload
    ):
        return ToolResult(
            error=f"malformed tool output: {type(payload).__name__}", error_kind="malformed"
        )
    return ToolResult(results=payload)


_QUERY = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


class MemorySearchTool(Tool):
    name = "memory_search"
    spec = ToolSpec(name, "Search the team's group-chat history (long-term memory).", _QUERY)

    def __init__(
        self,
        arm: MemoryArm,
        *,
        top_k: int = 5,
        clock: Clock | None = None,
        latency_s: float = 0.0,
    ):
        self.arm = arm
        self.top_k = top_k
        self.clock = clock
        self.latency_s = latency_s

    def raw_call(self, arguments: dict[str, Any]) -> Any:
        if self.clock is not None:
            self.clock.sleep(self.latency_s)
        items = self.arm.search(str(arguments.get("query", "")), self.top_k)
        return [
            {
                "message_id": it.message_id,
                "date": it.date,
                "time": it.time,
                "group": it.group,
                "speaker": it.speaker,
                "text": it.text,
            }
            for it in items
        ]


# The stub "web": general knowledge only. Internal team questions get nothing.
WEB_KNOWLEDGE: dict[str, str] = {
    "boiling point of water": "Water boils at 100 degrees Celsius at sea level.",
    "pride and prejudice": "Pride and Prejudice is an 1813 novel by Jane Austen.",
    "http stand for": "HTTP stands for Hypertext Transfer Protocol.",
    "leap year": "A leap year has 366 days.",
}


class WebSearchTool(Tool):
    name = "web_search"
    spec = ToolSpec(name, "Search the public web for general knowledge.", _QUERY)

    def __init__(self, *, clock: Clock | None = None, latency_s: float = 0.0):
        self.clock = clock
        self.latency_s = latency_s

    def raw_call(self, arguments: dict[str, Any]) -> Any:
        if self.clock is not None:
            self.clock.sleep(self.latency_s)
        query = normalize(str(arguments.get("query", "")))
        return [
            {"title": topic, "text": snippet}
            for topic, snippet in WEB_KNOWLEDGE.items()
            if topic in query
        ]


FINAL_ANSWER_SPEC = ToolSpec(
    "final_answer",
    "Deliver the final answer to the user. Call exactly once, with the answer itself.",
    {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)

FaultMode = Literal["timeout", "empty", "exception", "malformed"]
FAULT_MODES: tuple[FaultMode, ...] = ("timeout", "empty", "exception", "malformed")


class FaultyTool(Tool):
    """Wraps a tool and injects a fault on its first `times` calls (None = every call)."""

    def __init__(
        self,
        inner: Tool,
        mode: FaultMode,
        *,
        times: int | None = None,
        clock: Clock | None = None,
        timeout_s: float = 10.0,
    ):
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}")
        self.inner = inner
        self.name = inner.name
        self.spec = inner.spec
        self.mode = mode
        self.times = times
        self.clock = clock
        self.timeout_s = timeout_s
        self.calls = 0

    def raw_call(self, arguments: dict[str, Any]) -> Any:
        self.calls += 1
        if self.times is not None and self.calls > self.times:
            return self.inner.raw_call(arguments)
        if self.mode == "timeout":
            if isinstance(self.clock, FakeClock):
                self.clock.sleep(self.timeout_s)
            raise ToolTimeoutError(f"{self.name} timed out after {self.timeout_s:.0f}s")
        if self.mode == "exception":
            raise MemoryServiceError(f"{self.name}: 500 internal server error")
        if self.mode == "malformed":
            return {"unexpected": "shape"}
        return []  # empty
