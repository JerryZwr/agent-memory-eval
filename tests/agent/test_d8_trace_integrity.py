"""Dimension 8 · trace integrity: the observability data itself must be trustworthy."""

from __future__ import annotations

import copy

import allure
import pytest

from agentmemeval.agent.checks import check_trace
from agentmemeval.agent.policies import POLICIES
from agentmemeval.tracing import Trace, check_integrity

pytestmark = [pytest.mark.agent, allure.feature("D8 trace 完整性")]


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_every_policy_produces_a_complete_trace(run_agent, policy):
    r = run_agent(policy, "F_MH_Mini_002")
    check = check_trace(r)
    assert check, check.reason
    assert len(r.trace.spans) == 1 + r.n_llm_calls + len(r.tool_steps)  # run + steps


def _broken(trace: Trace, how: str) -> Trace:
    t = copy.deepcopy(trace)
    tool = next(s for s in t.spans if s.kind == "tool_call")
    if how == "missing_tool_span":
        t.spans.remove(tool)
    elif how == "unclosed_span":
        tool.end = None
    elif how == "outside_parent":
        tool.end = t.root.end + 1.0
    elif how == "orphan":
        tool.parent_id = 999
    return t


@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("missing_tool_span", "0 tool spans for 1 tool calls"),
        ("unclosed_span", "was never closed"),
        ("outside_parent", "outside its parent's time range"),
        ("orphan", "missing parent"),
    ],
)
def test_integrity_check_reports_a_deliberately_broken_trace(run_agent, how, message):
    r = run_agent("good", "F_SH_Mini_001")
    assert check_trace(r)  # the original is fine
    problems = check_integrity(
        _broken(r.trace, how), expected_llm_calls=r.n_llm_calls, expected_tool_calls=1
    )
    assert any(message in p for p in problems), problems


def test_trace_round_trips_through_json(run_agent):
    r = run_agent("good", "WEB_001")
    again = Trace.from_dict(r.trace.to_dict())
    assert again.to_dict() == r.trace.to_dict()
