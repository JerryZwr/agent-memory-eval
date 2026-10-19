"""Dimension 5 · tool fault injection: timeout / empty / exception / malformed, and LLM failure.

A memory tool fault must end in env_failure or a recovery by retry; it must
never produce a normal-looking answer the tool never supported.
"""

from __future__ import annotations

from collections.abc import Sequence

import allure
import pytest

from agentmemeval.agent.checks import check_fault_handling
from agentmemeval.agent.policies import GoodPolicy, final, read, search
from agentmemeval.agent.tools import FAULT_MODES, FaultyTool, MemorySearchTool, WebSearchTool
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.errors import LLMUnavailableError
from agentmemeval.llm.base import ChatMessage, ToolCall

pytestmark = [
    pytest.mark.agent,
    allure.feature("D5 工具故障注入"),
    allure.severity(allure.severity_level.CRITICAL),
]


def faulty_memory(indexed_arm, mode, times=None):
    def build(clock):
        inner = MemorySearchTool(indexed_arm, clock=clock, latency_s=0.05)
        return {
            "memory_search": FaultyTool(inner, mode, times=times, clock=clock),
            "web_search": WebSearchTool(clock=clock),
        }

    return build


@pytest.mark.parametrize("mode", FAULT_MODES)
def test_persistent_fault_never_yields_a_made_up_answer(run_agent, indexed_arm, mode):
    r = run_agent("good", "F_SH_Mini_001", tools=faulty_memory(indexed_arm, mode))
    check = check_fault_handling(r, "memory_search")
    assert check, check.reason
    if mode == "empty":
        assert r.outcome is Outcome.DELIVERED  # honest "cannot find"
        assert "cannot find" in r.final_answer
    else:
        assert r.outcome is Outcome.ENV_FAILURE  # never counted as a wrong answer
        assert r.outcome is not Outcome.AGENT_ERROR


@pytest.mark.parametrize("mode", ["timeout", "exception"])
def test_transient_fault_recovers_on_retry_and_says_so(run_agent, indexed_arm, judge_task, mode):
    r = run_agent("good", "F_SH_Mini_001", tools=faulty_memory(indexed_arm, mode, times=1))
    assert check_fault_handling(r, "memory_search")
    assert judge_task("F_SH_Mini_001", r) == "CORRECT"
    assert any("succeeded on retry" in w for w in r.warnings)  # a retry is never silent


class CarelessPolicy(GoodPolicy):
    """Ignores tool errors and answers anyway: the check must catch it."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str | ToolCall:
        c = read(messages)
        if not c.is_router and c.history and not c.history[-1].ok:
            return final("The primary database is PostgreSQL 16.")
        if not c.is_router and not c.history:
            return search("memory_search", c.task.question)
        return super().__call__(messages)


def test_an_answer_unsupported_by_any_tool_result_is_rejected(run_agent, indexed_arm):
    r = run_agent(CarelessPolicy(), "F_SH_Mini_001", tools=faulty_memory(indexed_arm, "timeout"))
    assert r.outcome is Outcome.DELIVERED  # looks perfectly normal...
    check = check_fault_handling(r, "memory_search")
    assert not check
    assert "no successful tool call supports" in check.reason


def test_llm_endpoint_failure_is_env_failure(run_agent):
    def down(messages):
        raise LLMUnavailableError("429 rate limited")

    r = run_agent(down, "F_SH_Mini_001")
    assert r.outcome is Outcome.ENV_FAILURE
    assert "429" in r.error


def test_a_bug_in_the_agent_is_agent_error_not_env_failure(run_agent):
    def buggy(messages):
        return {}["missing key"]

    r = run_agent(buggy, "F_SH_Mini_001")
    assert r.outcome is Outcome.AGENT_ERROR
    assert "KeyError" in r.error


def test_calling_a_nonexistent_tool_is_the_agents_fault_not_the_environments(run_agent):
    def confused(messages: Sequence[ChatMessage]) -> str | ToolCall:
        c = read(messages)
        if c.is_router:
            return "memory"
        if not c.history:
            return ToolCall("database_lookup", {"query": c.task.question})
        return "I could not look that up."

    r = run_agent(confused, "F_SH_Mini_001")
    assert r.tool_steps[0].error_kind == "unknown_tool"
    assert r.outcome is Outcome.UNDELIVERED  # not env_failure
