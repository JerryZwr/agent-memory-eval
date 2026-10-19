"""Dimension 2 · convergence and budgets: budgets really stop the loop; non-convergence shows."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import check_converged, check_converged_and_correct
from agentmemeval.agent.core import Budget
from agentmemeval.analysis.outcome import Outcome, UndeliveredReason

pytestmark = [pytest.mark.agent, allure.feature("D2 收敛与预算")]


def test_good_policy_converges_with_a_correct_answer(run_agent, judge_task):
    r = run_agent("good", "F_SH_Mini_002")
    assert check_converged_and_correct(r, Budget(), judge_task("F_SH_Mini_002", r))


def test_stubborn_searcher_is_stopped_by_the_same_tool_streak(run_agent):
    budget = Budget(max_same_tool_streak=3, max_tool_calls=10)
    r = run_agent("stubborn_searcher", "F_SH_Mini_001", budget=budget)
    assert not check_converged(r, budget)
    assert r.budget_hit == "same_tool_streak"
    assert r.max_same_tool_streak == 3
    assert (r.outcome, r.undelivered_reason) == (Outcome.UNDELIVERED, UndeliveredReason.BUDGET)


def test_max_tool_calls_is_enforced(run_agent):
    budget = Budget(max_same_tool_streak=99, max_tool_calls=2)
    r = run_agent("stubborn_searcher", "F_SH_Mini_001", budget=budget)
    assert r.budget_hit == "tool_calls"
    assert len(r.tool_steps) == 2


def test_wall_clock_budget_is_enforced(run_agent):
    budget = Budget(max_same_tool_streak=99, max_tool_calls=99, max_iterations=99, wall_clock_s=5)
    r = run_agent("stubborn_searcher", "F_SH_Mini_001", budget=budget, llm_latency=2.0)
    assert r.budget_hit == "wall_clock"
    assert r.trace.root.duration <= 5 + 2.0 + 1.0  # stops within one more step of the limit


def test_a_tighter_budget_is_not_a_fix(run_agent, judge_task):
    """Fake fix: lower the limits and the run gets 'faster', but the answer is still missing."""
    tight = Budget(max_tool_calls=1)
    r = run_agent("stubborn_searcher", "F_SH_Mini_001", budget=tight)
    assert len(r.tool_steps) <= 1  # fewer calls than before...
    check = check_converged_and_correct(r, tight, judge_task("F_SH_Mini_001", r))
    assert not check  # ...yet still rejected, because the answer is not correct
