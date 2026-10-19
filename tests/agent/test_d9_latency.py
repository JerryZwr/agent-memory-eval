"""Dimension 9 · latency breakdown: llm / tool / framework, and TTFT for streamed calls.

Uses the FakeClock, so the expected numbers are exact: route (0.30 s) + two
agent decisions (2 x 0.30 s) = 0.90 s LLM; one memory_search = 0.05 s;
two iterations of context assembly = 2 x 0.01 s framework.
"""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.core import Agent, AgentTask
from agentmemeval.agent.policies import make_policy_llm
from agentmemeval.agent.tools import MemorySearchTool, WebSearchTool
from agentmemeval.tracing import RealClock, latency_breakdown

pytestmark = [pytest.mark.agent, allure.feature("D9 耗时拆解")]


def test_three_way_breakdown_is_exact_on_the_simulated_clock(run_agent):
    b = latency_breakdown(run_agent("good", "F_SH_Mini_001").trace)
    allure.attach(str(b.to_dict()), name="latency", attachment_type=allure.attachment_type.TEXT)
    assert b.llm_ms == pytest.approx(900)
    assert b.tool_ms == pytest.approx(50)
    assert b.framework_ms == pytest.approx(20)
    assert b.total_ms == pytest.approx(970)
    assert sum(b.shares().values()) == pytest.approx(1.0)


def test_ttft_is_measured_for_every_streamed_call(run_agent):
    r = run_agent("good", "F_SH_Mini_001")
    b = latency_breakdown(r.trace)
    assert len(b.ttft_ms) == r.n_llm_calls == 3
    assert all(t == pytest.approx(80) for t in b.ttft_ms)


def test_framework_overhead_is_its_own_segment(run_agent):
    """A two-way llm/tool split would hide this time entirely."""
    slow_framework = latency_breakdown(run_agent("good", "F_SH_Mini_001", framework_cost=0.5).trace)
    assert slow_framework.framework_ms == pytest.approx(1000)
    assert slow_framework.shares()["framework"] > slow_framework.shares()["tool"]


def test_breakdown_on_the_real_clock_is_consistent(indexed_arm, allure_report):
    clock = RealClock()
    tools = {"memory_search": MemorySearchTool(indexed_arm), "web_search": WebSearchTool()}
    r = Agent(make_policy_llm("good"), tools, clock=clock).run(
        AgentTask("What does HTTP stand for?")
    )
    allure_report(r, "good (real clock)", "WEB_003")
    b = latency_breakdown(r.trace)
    assert b.total_ms >= b.llm_ms + b.tool_ms - 1e-6
    assert b.framework_ms >= -1e-6
    assert b.ttft_ms == ()  # no clock given to the LLM: not streamed, no TTFT invented
