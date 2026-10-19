"""Dimension 6 · trajectory assertions: catch "right result, wrong process"."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import check_trajectory

pytestmark = [pytest.mark.agent, allure.feature("D6 轨迹断言")]

MEMORY_TASKS = ["F_SH_Mini_001", "F_MH_Mini_002", "MA_C_Mini_002", "F_HL_Mini_002"]


@pytest.mark.parametrize("task_id", MEMORY_TASKS)
def test_good_policy_searches_memory_before_answering(run_agent, task_id):
    assert check_trajectory(run_agent("good", task_id), "memory")


def test_skipping_memory_fails_even_when_the_answer_is_right(run_agent, judge_task):
    r = run_agent("skips_memory", "MA_C_Mini_003")
    assert judge_task("MA_C_Mini_003", r) == "CORRECT"  # a lucky guess
    check = check_trajectory(r, "memory")
    assert not check
    assert "without memory_search" in check.reason


def test_web_search_finding_nothing_is_not_the_same_as_memory_finding_nothing(
    run_agent, judge_task
):
    """METHODOLOGY 1.8: it searched the web, 'found nothing', and was judged correct."""
    r = run_agent("misrouted", "F_HL_Mini_001")
    assert judge_task("F_HL_Mini_001", r) == "CORRECT"
    check = check_trajectory(r, "memory")
    assert not check
    assert "web_search" in check.reason


def test_general_knowledge_questions_have_no_memory_constraint(run_agent):
    assert check_trajectory(run_agent("good", "WEB_003"), "web")
