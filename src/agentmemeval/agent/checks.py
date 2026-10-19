"""Business-level acceptance checks for agent runs (METHODOLOGY 2.3).

One definition, used by both the test suite (tests/agent) and the batch
evaluation (`agentmemeval agent-eval`), so a check cannot mean one thing in a
test and another in a report. Every check is shown to fail on at least one
known-defective policy (P5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentmemeval.agent.core import AgentResult, Budget
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.judge.scripted import is_refusal
from agentmemeval.tracing import check_integrity

ROUTE_TOOL = {"memory": "memory_search", "web": "web_search", "chitchat": None}


@dataclass(frozen=True)
class Check:
    passed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.passed


def check_delivered(r: AgentResult) -> Check:
    """Delivered means a real answer came through final_answer: not 'the call returned'."""
    if r.outcome is Outcome.DELIVERED:
        return Check(True, "a real answer was delivered")
    reason = r.undelivered_reason.value if r.undelivered_reason else r.outcome.value
    return Check(False, f"not delivered: {r.outcome.value} ({reason})")


def check_converged(r: AgentResult, budget: Budget) -> Check:
    if r.budget_hit:
        return Check(False, f"budget hit: {r.budget_hit} after {len(r.tool_steps)} tool calls")
    if len(r.tool_steps) > budget.max_tool_calls:
        return Check(False, "more tool calls than the budget allows (budget not enforced)")
    return Check(True, f"stopped by itself after {len(r.tool_steps)} tool calls")


def check_converged_and_correct(r: AgentResult, budget: Budget, verdict: str | None) -> Check:
    """Speed / call-count criteria only count together with a correct answer (METHODOLOGY 3.1).

    Otherwise a "fix" that just lowers the limits would look like an improvement.
    """
    converged = check_converged(r, budget)
    if not converged:
        return converged
    if verdict != "CORRECT":
        return Check(False, f"converged but the answer is {verdict or 'not judged'}")
    return Check(True, "converged within budget with a correct answer")


def check_route(r: AgentResult, expected: str) -> Check:
    """First routing decision vs the label from the system's own rule, plus what it then did."""
    if r.route != expected:
        return Check(False, f"routed to {r.route!r}, the declared rule says {expected!r}")
    wanted = ROUTE_TOOL.get(expected)
    first_tool = r.tool_steps[0].name if r.tool_steps else None
    if wanted is not None and first_tool != wanted:
        return Check(False, f"routed to {expected!r} but first called {first_tool or 'no tool'}")
    return Check(True, f"routed to {expected!r} and acted on it")


FallbackStatus = Literal["not_needed", "fell_back", "no_fallback"]


def fallback_status(r: AgentResult) -> FallbackStatus:
    """After web_search came back empty, did the agent fall back to memory_search?"""
    steps = r.tool_steps
    for i, s in enumerate(steps):
        if s.name == "web_search" and s.ok and s.n_results == 0:
            later = steps[i + 1 :]
            return "fell_back" if any(t.name == "memory_search" for t in later) else "no_fallback"
    return "not_needed"


def check_fallback(r: AgentResult) -> Check:
    status = fallback_status(r)
    if status == "no_fallback":
        return Check(False, "web returned nothing and memory was never tried: "
                            "routing failure with hallucination risk")  # fmt: skip
    return Check(True, status)


def check_trajectory(r: AgentResult, kind: str) -> Check:
    """Memory questions: a successful memory_search must precede the answer.

    Judged on the trajectory, not the outcome: a lucky correct answer without
    retrieval still fails ("right result, wrong process").
    """
    if kind != "memory":
        return Check(True, "no trajectory constraint for this kind")
    if r.outcome is not Outcome.DELIVERED:
        return Check(True, "nothing delivered; covered by the delivery check")
    if any(s.name == "memory_search" and s.ok for s in r.tool_steps):
        return Check(True, "answer preceded by a successful memory_search")
    tools = [s.name for s in r.tool_steps] or ["none"]
    return Check(False, f"answered a memory question without memory_search (tools: {tools})")


def refused(r: AgentResult) -> bool:
    return r.outcome is Outcome.DELIVERED and is_refusal(r.final_answer or "")


def check_fault_handling(r: AgentResult, faulty_tool: str) -> Check:
    """Under an injected tool fault the run must end in env_failure, or recover via a retry.

    What it must never do is deliver a normal-looking answer the tool never supported.
    """
    if r.outcome is Outcome.ENV_FAILURE:
        return Check(True, "reported as an environment failure")
    if r.outcome is Outcome.AGENT_ERROR:
        return Check(False, f"the agent crashed on the fault: {r.error}")
    calls = [s for s in r.tool_steps if s.name == faulty_tool]
    if r.outcome is Outcome.DELIVERED:
        recovered = any(s.ok and s.n_results for s in calls)
        if recovered:
            return Check(True, "recovered: a later call succeeded")
        if is_refusal(r.final_answer or ""):
            return Check(True, "nothing usable came back and the agent said so")
        return Check(False, "delivered an answer that no successful tool call supports")
    return Check(True, f"not delivered ({r.undelivered_reason}); no fabricated answer")


def check_trace(r: AgentResult) -> Check:
    problems = check_integrity(
        r.trace, expected_llm_calls=r.n_llm_calls, expected_tool_calls=len(r.tool_steps)
    )
    return Check(not problems, "; ".join(problems) or "trace complete and consistent")


def routing_confusion(pairs: list[tuple[str, str | None]]) -> dict[str, dict[str, int]]:
    """{expected: {actual: count}} from (expected, first routing decision) pairs."""
    matrix: dict[str, dict[str, int]] = {}
    for expected, actual in pairs:
        row = matrix.setdefault(expected, {})
        key = actual if actual is not None else "none"
        row[key] = row.get(key, 0) + 1
    return matrix


def precision_recall(matrix: dict[str, dict[str, int]]) -> dict[str, tuple[float, float]]:
    """Per route: (precision, recall). A route never predicted has precision 0."""
    labels = set(matrix) | {a for row in matrix.values() for a in row}
    out = {}
    for label in sorted(labels):
        tp = matrix.get(label, {}).get(label, 0)
        predicted = sum(row.get(label, 0) for row in matrix.values())
        actual = sum(matrix.get(label, {}).values())
        out[label] = (tp / predicted if predicted else 0.0, tp / actual if actual else 0.0)
    return out
