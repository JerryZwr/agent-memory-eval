"""Judge interface for open-ended answers.

A judge returns a verdict or raises `JudgeError`. It never returns WRONG
because a call failed: a timeout recorded as WRONG is indistinguishable from a
real wrong answer afterwards (METHODOLOGY 1.5). The raw reply and the reason
are kept for calibration (1.6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

Label = Literal["CORRECT", "WRONG"]


class JudgeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    qid: str
    qtype: str
    question: str
    gold: str
    answer: str


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Label
    reason: str
    raw: str


class Judge(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Recorded as `judge_model`."""

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """Recorded as `judge_prompt_version`. Different versions are reported side by side."""

    @abstractmethod
    def judge(self, item: JudgeInput) -> JudgeVerdict:
        """Return a verdict, or raise JudgeError. Never return WRONG on failure."""
