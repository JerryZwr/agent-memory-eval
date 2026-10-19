"""LLM abstraction shared by answering, judging and the agent.

Implementations must raise `LLMUnavailableError` (an `InfraError`) when the
endpoint fails, so callers can tell "the model could not be reached" apart
from "the model answered badly".

Tool calling: pass `tools`; a reply then carries either text or `tool_calls`.
Streaming: implementations that stream report `ttft_s`, the time from request
to first chunk measured on the caller-supplied clock (None if not streamed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    tool_call: ToolCall | None = None  # assistant message that requested a tool
    name: str | None = None  # tool name, on role="tool" result messages


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[ToolCall, ...] = ()
    ttft_s: float | None = None


class LLM(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier recorded in result metadata (`answer_model` / `judge_model`)."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """One chat completion."""
