"""The answer-stage prompt.

`ANSWER_PROMPT_VERSION` is recorded in every result file. Changing the prompt
text changes what is being measured, so bump the version when you do.

`parse_answer_prompt` is the inverse of `render_answer_prompt`. It exists for
the scripted reader (an offline stand-in for an LLM) and is pinned by a
round-trip test so the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agentmemeval.arms.base import RetrievedItem
from agentmemeval.data.models import MC_LETTERS, Question
from agentmemeval.llm.base import ChatMessage

ANSWER_PROMPT_VERSION = "answer-v1"

SYSTEM_PROMPT = (
    "You answer questions about a team's group-chat history using only the memory "
    "records provided. If the records do not contain the answer, say that you cannot "
    "find it in the records. For a multiple-choice question, reply with the letter of "
    "the single best option."
)

H_RECORDS = "### Memory records"
H_QUESTION = "### Question"
H_OPTIONS = "### Options"
H_ANSWER = "### Answer"
NO_RECORDS = "(no records found)"
_SEP = " | "


@dataclass(frozen=True)
class PromptRecord:
    date: str
    time: str
    group: str
    speaker: str
    text: str


@dataclass(frozen=True)
class AnswerPromptView:
    question: str
    options: dict[str, str] | None
    records: tuple[PromptRecord, ...]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def render_answer_prompt(question: Question, items: Sequence[RetrievedItem]) -> list[ChatMessage]:
    lines = [H_RECORDS]
    if items:
        for n, it in enumerate(items, start=1):
            head = _SEP.join([it.time, _one_line(it.group), _one_line(it.speaker)])
            lines.append(f"[{n}] {head}{_SEP}{_one_line(it.text)}")
    else:
        lines.append(NO_RECORDS)
    lines += ["", H_QUESTION, _one_line(question.question)]
    if question.options is not None:
        lines += ["", H_OPTIONS]
        lines += [f"{k}. {_one_line(v)}" for k, v in question.options.items()]
    lines += ["", H_ANSWER]
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", "\n".join(lines)),
    ]


def _section(lines: list[str], header: str) -> list[str]:
    start = lines.index(header) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("### ")), len(lines))
    return [ln for ln in lines[start:end] if ln]


def parse_answer_prompt(messages: Sequence[ChatMessage]) -> AnswerPromptView:
    user = next(m.content for m in reversed(messages) if m.role == "user")
    lines = user.split("\n")
    records = []
    for ln in _section(lines, H_RECORDS):
        if ln == NO_RECORDS:
            continue
        _, _, body = ln.partition("] ")
        time, group, speaker, text = body.split(_SEP, 3)
        records.append(PromptRecord(time[:10], time, group, speaker, text))
    (question,) = _section(lines, H_QUESTION)
    options = None
    if H_OPTIONS in lines:
        options = {}
        for ln in _section(lines, H_OPTIONS):
            letter, _, text = ln.partition(". ")
            if letter not in MC_LETTERS:
                raise ValueError(f"unexpected option line {ln!r}")
            options[letter] = text
    return AnswerPromptView(question, options, tuple(records))
