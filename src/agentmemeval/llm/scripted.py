"""Deterministic LLM stand-ins for offline tests, CI and the demo.

`ScriptedLLM` wraps a *responder*: a pure function from chat messages to reply
text. Behaviour lives in responders so tests can plug in their own (M4 adds the
agent policies here).

`extractive_reader` is the default answering responder. It never sees the gold
answer: it reads the memory records in the prompt and answers from them, so
the gap between arms in the demo is produced by retrieval, not hard-coded.
Its limits are deliberate and realistic: it cannot do arithmetic over dates or
chain two facts that share no words with the question, which gives the
"retrieved but still wrong" cases the analysis has to attribute correctly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from agentmemeval.data.models import MC_LETTERS
from agentmemeval.llm.base import LLM, ChatMessage, LLMResponse
from agentmemeval.pipeline.prompts import AnswerPromptView, PromptRecord, parse_answer_prompt
from agentmemeval.text import tokenize

Responder = Callable[[Sequence[ChatMessage]], str]

READER_VERSION = "extractive-reader-v1"
REFUSAL = "I cannot find this in the records."
# Fraction of the question's content words a record must contain to be used.
MIN_COVERAGE = 0.34


class ScriptedLLM(LLM):
    def __init__(self, responder: Responder, *, model: str):
        self._responder = responder
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(text=self._responder(messages), model=self._model)


def _record_tokens(r: PromptRecord) -> set[str]:
    return set(tokenize(f"{r.speaker} {r.text}"))


def _guess(question: str) -> str:
    """What a model does with no usable context: pick an option. Stable across runs."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return MC_LETTERS[int(digest[:8], 16) % len(MC_LETTERS)]


def _answer_open(view: AnswerPromptView) -> str:
    q = set(tokenize(view.question))
    if not q or not view.records:
        return REFUSAL
    scored = sorted(
        ((len(q & _record_tokens(r)) / len(q), n, r) for n, r in enumerate(view.records)),
        key=lambda t: (-t[0], t[1]),
    )
    usable = [r for cov, _, r in scored if cov >= MIN_COVERAGE][:2]
    if not usable:
        return REFUSAL
    first, *rest = usable
    answer = f"On {first.date}, {first.speaker} said: {first.text}"
    for r in rest:
        answer += f" Also on {r.date}, {r.speaker} said: {r.text}"
    return answer


def _answer_mc(view: AnswerPromptView) -> str:
    assert view.options is not None
    if not view.records:
        return _guess(view.question)
    q = set(tokenize(view.question))
    scores: dict[str, tuple[float, str]] = {}  # letter -> (overlap, latest supporting date)
    for letter, text in view.options.items():
        opt = set(tokenize(text)) - q
        best = (0.0, "")
        for r in view.records:
            overlap = len(opt & _record_tokens(r)) / len(opt) if opt else 0.0
            best = max(best, (overlap, r.date))
        scores[letter] = best
    top = max(s for s, _ in scores.values())
    if top == 0:
        return _guess(view.question)
    tied = [k for k, (s, _) in scores.items() if s == top]
    if len(tied) > 1:
        # Prefer the option supported by the most recent record (newer facts win).
        latest = max(scores[k][1] for k in tied)
        tied = [k for k in tied if scores[k][1] == latest]
    if len(tied) == 1:
        return tied[0]
    return f"It is either {tied[0]} or {tied[1]}."


def extractive_reader(messages: Sequence[ChatMessage]) -> str:
    view = parse_answer_prompt(messages)
    return _answer_mc(view) if view.options is not None else _answer_open(view)


def make_reader_llm() -> ScriptedLLM:
    return ScriptedLLM(extractive_reader, model=f"scripted/{READER_VERSION}")
