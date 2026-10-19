"""Scripted agent policies: deterministic stand-ins for the agent's LLM (METHODOLOGY 2.1).

Each policy is a pure function of the conversation so far, like a real model:
it reads the routing decision and the tool observations from the messages.

| policy             | behaviour                                              | caught by        |
|--------------------|--------------------------------------------------------|------------------|
| good               | routes, searches memory, answers from what it found    | (reference)      |
| stubborn_searcher  | keeps rephrasing memory_search, never answers           | convergence      |
| skips_memory       | answers without searching (hallucination risk)          | trajectory, D    |
| narrator           | only says "let me check...", never calls final_answer   | delivery         |
| misrouted          | routes memory questions to web, no fallback             | routing, fallback|
| acknowledger       | delivers only an ack ("收到，稍后汇报")             | delivery         |

These are the known-defective inputs every acceptance check must fail on
(METHODOLOGY 3.1, P5).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agentmemeval.agent.core import AgentTask, parse_task
from agentmemeval.agent.tools import WEB_KNOWLEDGE
from agentmemeval.llm.base import ChatMessage, ToolCall
from agentmemeval.llm.scripted import REFUSAL, ScriptedLLM, answer_from_view, guess
from agentmemeval.pipeline.prompts import AnswerPromptView, PromptRecord
from agentmemeval.text import normalize
from agentmemeval.tracing import Clock

CHITCHAT_REPLY = "Hi! I can answer questions about the team's chat history or general facts."
ACK = "收到，稍后汇报。"
_CHITCHAT = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|how are you)\b",
    re.IGNORECASE,
)


def heuristic_route(question: str) -> str:
    """The routing a competent model would do: small talk, known general facts, else memory."""
    if _CHITCHAT.match(question):
        return "chitchat"
    q = normalize(question)
    if any(topic in q for topic in WEB_KNOWLEDGE):
        return "web"
    return "memory"


@dataclass
class Observed:
    call: ToolCall
    obs: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.obs.get("ok", False)) or bool(self.obs.get("delivered"))

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self.obs.get("results", []))


@dataclass
class Conversation:
    is_router: bool
    task: AgentTask
    route: str | None
    history: list[Observed]

    def calls_to(self, name: str) -> list[Observed]:
        return [h for h in self.history if h.call.name == name]


def read(messages: Sequence[ChatMessage]) -> Conversation:
    is_router = messages[0].content.startswith("ROUTER")
    task, route = parse_task(messages[1].content)
    history: list[Observed] = []
    pending: ToolCall | None = None
    for m in messages[2:]:
        if m.role == "assistant" and m.tool_call is not None:
            pending = m.tool_call
        elif m.role == "tool" and pending is not None:
            history.append(Observed(pending, json.loads(m.content)))
            pending = None
    return Conversation(is_router, task, route, history)


def final(text: str) -> ToolCall:
    return ToolCall("final_answer", {"text": text})


def search(name: str, query: str) -> ToolCall:
    return ToolCall(name, {"query": query})


def answer_from_memory(task: AgentTask, results: list[dict[str, Any]]) -> str:
    records = tuple(
        PromptRecord(r.get("date", ""), r.get("time", ""), r.get("group", ""),
                     r.get("speaker", ""), r["text"])
        for r in results
    )  # fmt: skip
    return answer_from_view(AnswerPromptView(task.question, task.options, records))


Policy = Callable[[Sequence[ChatMessage]], "str | ToolCall"]


class GoodPolicy:
    """Route, search, answer from what was found; retry a failed tool once; fall back to memory."""

    def __init__(self, *, force_route: str | None = None, max_retries: int = 1):
        self.force_route = force_route
        self.max_retries = max_retries

    def __call__(self, messages: Sequence[ChatMessage]) -> str | ToolCall:
        c = read(messages)
        if c.is_router:
            return self.force_route or heuristic_route(c.task.question)
        if c.route == "chitchat":
            return final(CHITCHAT_REPLY)
        if not c.history:
            first = "web_search" if c.route == "web" else "memory_search"
            return search(first, c.task.question)
        last = c.history[-1]
        if not last.ok:
            failures = 0
            for h in reversed(c.history):
                if h.call.name != last.call.name or h.ok:
                    break
                failures += 1
            if failures <= self.max_retries:
                return last.call  # retry once
            # give up honestly: no answer is better than a made-up one
            return f"The {last.call.name} tool keeps failing; I cannot answer reliably."
        if last.call.name == "web_search":
            if last.results:
                return final(f"According to the web: {last.results[0]['text']}")
            if not c.calls_to("memory_search"):
                return search("memory_search", c.task.question)  # fallback
            return final(REFUSAL)
        return final(answer_from_memory(c.task, last.results))


class AcknowledgeThenAnswer(GoodPolicy):
    """Acknowledges through final_answer first, then delivers the real answer (a legit pattern)."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str | ToolCall:
        c = read(messages)
        if not c.is_router and not c.history:
            return final(ACK)
        if not c.is_router and len(c.history) == 1 and c.history[0].call.name == "final_answer":
            return search("memory_search", c.task.question)
        return super().__call__(messages)


def stubborn_searcher(messages: Sequence[ChatMessage]) -> str | ToolCall:
    c = read(messages)
    if c.is_router:
        return heuristic_route(c.task.question)
    n = len(c.history)
    return search(
        "memory_search", c.task.question if n == 0 else f"{c.task.question} (rephrased {n})"
    )


def skips_memory(messages: Sequence[ChatMessage]) -> str | ToolCall:
    c = read(messages)
    if c.is_router:
        return heuristic_route(c.task.question)
    if c.route == "chitchat":
        return final(CHITCHAT_REPLY)
    if c.task.options:
        return final(guess(c.task.question))  # sometimes right by luck
    return final("Based on what I know, the team settled this on March 10.")


def narrator(messages: Sequence[ChatMessage]) -> str | ToolCall:
    c = read(messages)
    if c.is_router:
        return heuristic_route(c.task.question)
    return "Let me check the records first."


def misrouted(messages: Sequence[ChatMessage]) -> str | ToolCall:
    c = read(messages)
    if c.is_router:
        return "chitchat" if heuristic_route(c.task.question) == "chitchat" else "web"
    if c.route == "chitchat":
        return final(CHITCHAT_REPLY)
    if not c.history:
        return search("web_search", c.task.question)
    last = c.history[-1]
    if last.ok and last.results:
        return final(f"According to the web: {last.results[0]['text']}")
    return final("I could not find anything about this.")  # zero hits, and no fallback


def acknowledger(messages: Sequence[ChatMessage]) -> str | ToolCall:
    c = read(messages)
    if c.is_router:
        return heuristic_route(c.task.question)
    if not c.calls_to("final_answer"):
        return final(ACK)
    return ""  # never follows up


POLICIES: dict[str, Callable[[], Policy]] = {
    "good": GoodPolicy,
    "stubborn_searcher": lambda: stubborn_searcher,
    "skips_memory": lambda: skips_memory,
    "narrator": lambda: narrator,
    "misrouted": lambda: misrouted,
    "acknowledger": lambda: acknowledger,
}
BAD_POLICIES = tuple(p for p in POLICIES if p != "good")


def make_policy_llm(
    policy: str | Policy,
    *,
    clock: Clock | None = None,
    latency_s: float = 0.0,
    ttft_s: float = 0.0,
) -> ScriptedLLM:
    if isinstance(policy, str):
        try:
            responder, name = POLICIES[policy](), policy
        except KeyError:
            raise ValueError(f"unknown policy {policy!r}; known: {sorted(POLICIES)}") from None
    else:
        responder, name = policy, type(policy).__name__
    return ScriptedLLM(
        responder, model=f"scripted-agent/{name}", clock=clock, latency_s=latency_s, ttft_s=ttft_s
    )
