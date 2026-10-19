"""Fixtures for the agent-layer test dimensions (METHODOLOGY 2.3).

`run_agent(policy, task_id, ...)` builds the agent on the offline stack (bm25
over the synthetic corpus, the web stub, a FakeClock) and reports the run to
Allure: steps route -> each turn -> outcome, with the trace JSON attached.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import allure
import pytest

from agentmemeval.agent.core import Agent, AgentResult, Budget
from agentmemeval.agent.policies import Policy, make_policy_llm
from agentmemeval.agent.tasks import TaskSpec, all_tasks, judge_answer
from agentmemeval.agent.tools import MemorySearchTool, Tool, WebSearchTool
from agentmemeval.judge.scripted import ScriptedJudge
from agentmemeval.pipeline.config import Components, build_components, default_eval_config
from agentmemeval.tracing import FakeClock

# Simulated latencies (seconds) on the FakeClock.
LLM_LATENCY, LLM_TTFT = 0.30, 0.08
MEMORY_LATENCY, WEB_LATENCY = 0.05, 0.20
FRAMEWORK_COST = 0.01


@pytest.fixture(scope="session")
def components() -> Components:
    return build_components(default_eval_config())


@pytest.fixture(scope="session")
def tasks(components) -> dict[str, TaskSpec]:
    return {t.task_id: t for t in all_tasks(components)}


@pytest.fixture(scope="session")
def indexed_arm(components):
    arm = components.new_arm()
    arm.add(components.dataset)
    return arm


def default_tools(arm, clock) -> dict[str, Tool]:
    return {
        "memory_search": MemorySearchTool(arm, clock=clock, latency_s=MEMORY_LATENCY),
        "web_search": WebSearchTool(clock=clock, latency_s=WEB_LATENCY),
    }


def report_to_allure(result: AgentResult, policy_name: str, task_id: str) -> None:
    allure.dynamic.story(policy_name)
    allure.dynamic.parameter("task", task_id)
    with allure.step(f"路由：{result.route}"):
        pass
    for i, s in enumerate(result.steps, start=1):
        if s.kind == "tool":
            status = f"{s.n_results} 条结果" if s.ok else f"失败（{s.error}）"
            title = f"第 {i} 轮：调用 {s.name}，{status}"
        elif s.kind == "final":
            title = f"第 {i} 轮：final_answer «{(s.text or '')[:60]}»"
        else:
            title = f"第 {i} 轮：只输出文本 «{(s.text or '')[:60]}»"
        with allure.step(title):
            pass
    reason = f"（{result.undelivered_reason.value}）" if result.undelivered_reason else ""
    budget = f"，预算触发：{result.budget_hit}" if result.budget_hit else ""
    with allure.step(f"结局：{result.outcome.value}{reason}{budget}"):
        pass
    allure.attach(
        json.dumps(result.trace.to_dict(), ensure_ascii=False, indent=2),
        name="trace.json",
        attachment_type=allure.attachment_type.JSON,
    )
    if result.final_answer:
        allure.attach(
            result.final_answer, name="最终答案", attachment_type=allure.attachment_type.TEXT
        )


RunAgent = Callable[..., AgentResult]


@pytest.fixture
def run_agent(tasks, indexed_arm) -> RunAgent:
    def run(
        policy: str | Policy,
        task_id: str,
        *,
        tools: Callable[[FakeClock], dict[str, Tool]] | None = None,
        budget: Budget | None = None,
        clock: FakeClock | None = None,
        llm_latency: float = LLM_LATENCY,
        framework_cost: float = FRAMEWORK_COST,
    ) -> AgentResult:
        clock = clock or FakeClock()
        llm = make_policy_llm(policy, clock=clock, latency_s=llm_latency,
                              ttft_s=min(LLM_TTFT, llm_latency))  # fmt: skip
        agent = Agent(
            llm,
            tools(clock) if tools else default_tools(indexed_arm, clock),
            budget=budget,
            clock=clock,
            framework_cost_s=framework_cost,
        )
        result = agent.run(tasks[task_id].task)
        name = policy if isinstance(policy, str) else type(policy).__name__
        report_to_allure(result, name, task_id)
        return result

    return run


@pytest.fixture(scope="session")
def judge_task(tasks):
    judge = ScriptedJudge()

    def verdict(task_id: str, result: AgentResult) -> str | None:
        return judge_answer(tasks[task_id], result.outcome, result.final_answer, judge)

    return verdict


@pytest.fixture
def allure_report() -> Callable[[AgentResult, str, str], None]:
    """For tests that build an Agent themselves: same Allure steps and trace attachment."""
    return report_to_allure
