"""Deterministic LLM stand-ins for offline tests, CI and the demo.

`ScriptedLLM` wraps a *responder*: a pure function from chat messages to a
reply (text, or a `ToolCall`). Behaviour lives in responders so tests can plug
in their own; the agent policies are in `agentmemeval.agent.policies`.

With a clock, it simulates a *streamed* call: the reply arrives in chunks, the
first after `ttft_s`, the last after `latency_s`, and TTFT is measured on that
clock exactly as a streaming client would measure it.

`extractive_reader` is the default answering responder. It never sees the gold
answer: it reads the memory records in the prompt and answers from them, so
the gap between arms in the demo is produced by retrieval, not hard-coded.
Its limits are deliberate and realistic: it cannot do arithmetic over dates or
chain two facts that share no words with the question, which gives the
"retrieved but still wrong" cases the analysis has to attribute correctly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence

from agentmemeval.data.models import MC_LETTERS
from agentmemeval.llm.base import LLM, ChatMessage, LLMResponse, ToolCall, ToolSpec
from agentmemeval.pipeline.prompts import AnswerPromptView, PromptRecord, parse_answer_prompt
from agentmemeval.text import tokenize
from agentmemeval.tracing import Clock

Responder = Callable[[Sequence[ChatMessage]], "str | ToolCall"]

READER_VERSION = "extractive-reader-v1"
REFUSAL = "I cannot find this in the records."
# Fraction of the question's content words a record must contain to be used.
MIN_COVERAGE = 0.34


class ScriptedLLM(LLM):
    def __init__(
        self,
        responder: Responder,
        *,
        model: str,
        clock: Clock | None = None,
        latency_s: float = 0.0,
        ttft_s: float = 0.0,
        chunks: int = 4,
    ):
        if latency_s < ttft_s:
            raise ValueError("latency_s must be >= ttft_s")
        self._responder = responder
        self._model = model
        self._clock = clock
        self._latency_s = latency_s
        self._ttft_s = ttft_s
        self._chunks = max(1, chunks)

    @property
    def model(self) -> str:
        return self._model

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str | ToolCall]:
        """Yield the reply in chunks, spending simulated time on the clock."""
        reply = self._responder(messages)
        clock = self._clock
        if clock is not None:
            clock.sleep(self._ttft_s)
        if isinstance(reply, ToolCall):
            yield reply
            pieces: list[str | ToolCall] = []
        else:
            size = max(1, -(-len(reply) // self._chunks))
            pieces = [reply[i : i + size] for i in range(0, len(reply), size)] or [""]
            yield pieces[0]
            pieces = pieces[1:]
        rest = (self._latency_s - self._ttft_s) / max(1, len(pieces))
        for piece in pieces:
            if clock is not None:
                clock.sleep(rest)
            yield piece
        if not pieces and clock is not None:
            clock.sleep(self._latency_s - self._ttft_s)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        start = self._clock.now() if self._clock else None
        ttft: float | None = None
        text, calls = [], []
        for chunk in self.stream(messages):
            if ttft is None and self._clock is not None and start is not None:
                ttft = self._clock.now() - start  # measured, not copied from the config
            if isinstance(chunk, ToolCall):
                calls.append(chunk)
            else:
                text.append(chunk)
        return LLMResponse(
            text="".join(text), model=self._model, tool_calls=tuple(calls), ttft_s=ttft
        )


def _record_tokens(r: PromptRecord) -> set[str]:
    return set(tokenize(f"{r.speaker} {r.text}"))


def guess(question: str) -> str:
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
        return guess(view.question)
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
        return guess(view.question)
    tied = [k for k, (s, _) in scores.items() if s == top]
    if len(tied) > 1:
        # Prefer the option supported by the most recent record (newer facts win).
        latest = max(scores[k][1] for k in tied)
        tied = [k for k in tied if scores[k][1] == latest]
    if len(tied) == 1:
        return tied[0]
    return f"It is either {tied[0]} or {tied[1]}."


def answer_from_view(view: AnswerPromptView) -> str:
    """The reader's answer for a question, its options and the records it was given."""
    return _answer_mc(view) if view.options is not None else _answer_open(view)


def extractive_reader(messages: Sequence[ChatMessage]) -> str:
    return answer_from_view(parse_answer_prompt(messages))


def make_reader_llm() -> ScriptedLLM:
    return ScriptedLLM(extractive_reader, model=f"scripted/{READER_VERSION}")
