"""A deliberately small tool-calling agent (METHODOLOGY 2.1).

    question -> route (memory / web / chitchat)          one LLM call, first decision wins
             -> loop, at most `max_iterations` LLM calls:
                  the LLM picks a tool, or calls final_answer
             -> outcome

Outcome rules (METHODOLOGY 1.4, 2.1):

* delivered   - final_answer produced a real answer. An acknowledgement sent
                through final_answer ("got it, will report back") is not an
                answer: the loop keeps going so a real answer can follow, and
                if none does the run is undelivered;
* undelivered - budget exhausted, or the run ended in plain text (narration);
* env_failure - the LLM endpoint failed, or the run gave up right after a
                tool infrastructure fault;
* agent_error - the agent's own code (or its policy) raised.

The agent exists to be tested, not to be good: every feature maps to one of
the nine test dimensions in METHODOLOGY 2.3.
"""

from __future__ import annotations

import json
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agentmemeval.agent.tools import FINAL_ANSWER_SPEC, Tool, ToolResult
from agentmemeval.analysis.outcome import (
    Outcome,
    UndeliveredReason,
    classify_delivery,
    placeholder_kind,
)
from agentmemeval.errors import InfraError
from agentmemeval.llm.base import LLM, ChatMessage, ToolCall
from agentmemeval.tracing import Clock, RealClock, Trace, Tracer

ROUTES = ("memory", "web", "chitchat")
# The system's own declared routing rule. Expected labels in evaluation come from it,
# so routing correctness does not depend on the evaluator's taste (METHODOLOGY 2.3 #3).
ROUTING_RULE = (
    "Questions that need the team's chat history (decisions, people, dates or numbers "
    "from past conversations) MUST go to `memory`. General-knowledge questions go to "
    "`web`. Greetings and small talk go to `chitchat`."
)
ROUTER_SYSTEM = (
    "ROUTER. Classify the user's question as exactly one word: memory, web or chitchat. "
    + ROUTING_RULE
)
AGENT_SYSTEM = (
    "AGENT. Answer the user's question. Use memory_search for the team's chat history and "
    "web_search for general knowledge. Deliver the answer with final_answer. If the tools "
    "do not contain the answer, say you cannot find it; never guess."
)


@dataclass(frozen=True)
class Budget:
    max_iterations: int = 6  # agent LLM calls (the routing call is separate)
    max_tool_calls: int = 4
    max_same_tool_streak: int = 3
    wall_clock_s: float = 60.0


@dataclass(frozen=True)
class AgentTask:
    """What the agent is asked. `options` makes it multiple choice."""

    question: str
    options: dict[str, str] | None = None


def render_task(task: AgentTask, route: str | None = None) -> str:
    lines = [f"Route: {route}"] if route is not None else []
    lines += ["### Question", " ".join(task.question.split())]
    if task.options:
        lines += ["### Options", *(f"{k}. {v}" for k, v in task.options.items())]
    return "\n".join(lines)


def parse_task(content: str) -> tuple[AgentTask, str | None]:
    route = None
    lines = content.split("\n")
    if lines and lines[0].startswith("Route: "):
        route = lines.pop(0)[len("Route: ") :]
    q_at = lines.index("### Question")
    question = lines[q_at + 1]
    options = None
    if "### Options" in lines:
        options = {}
        for ln in lines[lines.index("### Options") + 1 :]:
            letter, _, text = ln.partition(". ")
            options[letter] = text
    return AgentTask(question, options), route


StepKind = Literal["tool", "final", "text"]


@dataclass(frozen=True)
class Step:
    kind: StepKind
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    n_results: int | None = None
    error: str | None = None
    error_kind: str | None = None
    text: str | None = None


@dataclass
class AgentResult:
    task: AgentTask
    route: str | None
    steps: list[Step]
    final_answer: str | None
    outcome: Outcome
    undelivered_reason: UndeliveredReason | None
    budget_hit: str | None
    n_llm_calls: int  # including the routing call
    trace: Trace
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def route_valid(self) -> bool:
        return self.route in ROUTES

    @property
    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def n_tool_calls_by_name(self) -> dict[str, int]:
        return dict(Counter(s.name for s in self.tool_steps if s.name))

    @property
    def max_same_tool_streak(self) -> int:
        best = streak = 0
        previous = None
        for s in self.tool_steps:
            streak = streak + 1 if s.name == previous else 1
            previous = s.name
            best = max(best, streak)
        return best

    @property
    def deliveries(self) -> list[str]:
        return [s.text or "" for s in self.steps if s.kind == "final"]


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: Mapping[str, Tool],
        *,
        budget: Budget | None = None,
        clock: Clock | None = None,
        framework_cost_s: float = 0.0,
    ):
        """`framework_cost_s` advances the clock per iteration to model context assembly.

        Only meaningful with a FakeClock (tests and the simulated batch eval).
        """
        self.llm = llm
        self.tools = dict(tools)
        self.budget = budget or Budget()
        self.clock = clock or RealClock()
        self.framework_cost_s = framework_cost_s

    def run(self, task: AgentTask) -> AgentResult:
        tracer = Tracer(self.clock)
        steps: list[Step] = []
        warnings: list[str] = []
        route: str | None = None
        final: str | None = None
        budget_hit: str | None = None
        n_llm = 0
        llm_failed = False
        agent_error: str | None = None
        specs = [t.spec for t in self.tools.values()] + [FINAL_ANSWER_SPEC]

        with tracer.span("run", "agent.run", question=task.question[:120]) as run_span:
            start = self.clock.now()
            try:
                with tracer.span("route", "router", model=self.llm.model) as s:
                    n_llm += 1
                    resp = self.llm.complete(
                        [
                            ChatMessage("system", ROUTER_SYSTEM),
                            ChatMessage("user", render_task(task)),
                        ]
                    )
                    route = resp.text.strip().lower()
                    s.attributes.update(route=route, ttft_s=resp.ttft_s)

                messages = [
                    ChatMessage("system", AGENT_SYSTEM),
                    ChatMessage("user", render_task(task, route)),
                ]
                for _ in range(self.budget.max_iterations):
                    if self.clock.now() - start > self.budget.wall_clock_s:
                        budget_hit = "wall_clock"
                        break
                    self.clock.sleep(self.framework_cost_s)  # context assembly, serialisation
                    with tracer.span("llm_call", "agent.decide", model=self.llm.model) as s:
                        n_llm += 1
                        resp = self.llm.complete(messages, tools=specs)
                        s.attributes.update(ttft_s=resp.ttft_s, n_tool_calls=len(resp.tool_calls))
                    if not resp.tool_calls:
                        steps.append(Step(kind="text", text=resp.text))
                        break
                    call = resp.tool_calls[0]
                    messages.append(ChatMessage("assistant", "", tool_call=call))
                    if call.name == FINAL_ANSWER_SPEC.name:
                        text = str(call.arguments.get("text", ""))
                        steps.append(Step(kind="final", name=call.name, text=text))
                        messages.append(ChatMessage("tool", json.dumps({"delivered": True}),
                                                    name=call.name))  # fmt: skip
                        if placeholder_kind(text) is None and text.strip():
                            final = text
                            break
                        continue  # an acknowledgement: the real answer may still follow
                    hit = self._check_tool_budget(steps, call)
                    if hit:
                        budget_hit = hit
                        break
                    result = self._call_tool(tracer, call)
                    if result.ok and steps and steps[-1].name == call.name and steps[-1].error:
                        warnings.append(
                            f"{call.name} failed ({steps[-1].error}) then succeeded on retry"
                        )  # a retry that succeeds must not be silent (METHODOLOGY 3.4)
                    steps.append(
                        Step(
                            kind="tool",
                            name=call.name,
                            arguments=dict(call.arguments),
                            ok=result.ok,
                            n_results=len(result.results) if result.ok else None,
                            error=result.error,
                            error_kind=result.error_kind,
                        )
                    )
                    messages.append(ChatMessage("tool", _observation(result), name=call.name))
                else:
                    budget_hit = "iterations"
            except InfraError as exc:
                llm_failed = True
                agent_error = None
                run_span.attributes["env_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # the agent's own code or policy crashed
                agent_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                run_span.attributes["agent_error"] = agent_error
            run_span.attributes.update(route=route, budget_hit=budget_hit)

        outcome, reason = _classify(steps, final, budget_hit, llm_failed, agent_error)
        return AgentResult(
            task=task,
            route=route,
            steps=steps,
            final_answer=final,
            outcome=outcome,
            undelivered_reason=reason,
            budget_hit=budget_hit,
            n_llm_calls=n_llm,
            trace=tracer.trace,
            error=agent_error or run_span.attributes.get("env_error"),
            warnings=warnings,
        )

    def _check_tool_budget(self, steps: list[Step], call: ToolCall) -> str | None:
        tool_steps = [s for s in steps if s.kind == "tool"]
        if len(tool_steps) >= self.budget.max_tool_calls:
            return "tool_calls"
        streak = 0
        for s in reversed(tool_steps):
            if s.name != call.name:
                break
            streak += 1
        if streak >= self.budget.max_same_tool_streak:
            return "same_tool_streak"
        return None

    def _call_tool(self, tracer: Tracer, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        query = str(call.arguments.get("query", ""))[:80]
        with tracer.span("tool_call", call.name, query=query) as s:
            if tool is None:
                result = ToolResult(error=f"unknown tool {call.name!r}", error_kind="unknown_tool")
            else:
                result = tool(dict(call.arguments))
            s.attributes.update(ok=result.ok, n_results=len(result.results), error=result.error)
        return result


def _observation(result: ToolResult) -> str:
    if result.ok:
        return json.dumps({"ok": True, "results": result.results}, ensure_ascii=False)
    return json.dumps({"ok": False, "error": result.error, "error_kind": result.error_kind})


def _classify(
    steps: list[Step],
    final: str | None,
    budget_hit: str | None,
    llm_failed: bool,
    agent_error: str | None,
) -> tuple[Outcome, UndeliveredReason | None]:
    if agent_error:
        return Outcome.AGENT_ERROR, None
    if llm_failed:
        return Outcome.ENV_FAILURE, None
    if final is not None:
        c = classify_delivery(final)
        return c.outcome, c.reason
    acknowledged = any(s.kind == "final" for s in steps)
    if budget_hit:
        return Outcome.UNDELIVERED, UndeliveredReason.BUDGET
    last_tool = next((s for s in reversed(steps) if s.kind == "tool"), None)
    if last_tool is not None and last_tool.error_kind in ("infra", "malformed"):
        # Gave up right after a tool fault: the environment, not the agent, prevented an answer.
        return Outcome.ENV_FAILURE, None
    if acknowledged:
        return Outcome.UNDELIVERED, UndeliveredReason.ACKNOWLEDGEMENT
    last_text = next((s.text for s in reversed(steps) if s.kind == "text"), None)
    kind = placeholder_kind(last_text or "") if last_text else None
    return Outcome.UNDELIVERED, kind or UndeliveredReason.TEXT_ONLY
