"""LLM abstraction shared by answering, judging and (M4) the agent.

Implementations must raise `LLMUnavailableError` (an `InfraError`) when the
endpoint fails, so callers can tell "the model could not be reached" apart
from "the model answered badly".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier recorded in result metadata (`answer_model` / `judge_model`)."""

    @abstractmethod
    def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> LLMResponse:
        """One chat completion."""
