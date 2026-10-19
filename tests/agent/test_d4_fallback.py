"""Dimension 4 · fallback: when the primary path finds nothing, try memory before answering."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import check_fallback, fallback_status
from agentmemeval.agent.policies import GoodPolicy

pytestmark = [pytest.mark.agent, allure.feature("D4 回退")]


def test_good_policy_falls_back_to_memory_after_an_empty_web_search(run_agent, judge_task):
    r = run_agent(GoodPolicy(force_route="web"), "F_SH_Mini_001")
    assert [s.name for s in r.tool_steps] == ["web_search", "memory_search"]
    assert fallback_status(r) == "fell_back"
    assert check_fallback(r)
    assert judge_task("F_SH_Mini_001", r) == "CORRECT"


def test_misrouted_without_fallback_is_flagged_as_hallucination_risk(run_agent):
    r = run_agent("misrouted", "F_SH_Mini_001")
    check = check_fallback(r)
    assert fallback_status(r) == "no_fallback"
    assert not check
    assert "hallucination risk" in check.reason


def test_no_fallback_needed_when_the_primary_path_answers(run_agent):
    r = run_agent("good", "WEB_002")
    assert fallback_status(r) == "not_needed"
    assert check_fallback(r)
