"""Dimension 7 · anti-hallucination: on no-evidence (D-class) questions the agent should refuse."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import refused
from agentmemeval.analysis.scoring import Ratio

pytestmark = [
    pytest.mark.agent,
    allure.feature("D7 反幻觉"),
    allure.severity(allure.severity_level.CRITICAL),
]


def _refusal_rate(run_agent, tasks, policy) -> Ratio:
    d_class = [tid for tid, t in tasks.items() if t.window_class == "D"]
    assert d_class, "the task set must contain no-evidence questions"
    return Ratio(sum(refused(run_agent(policy, tid)) for tid in d_class), len(d_class))


def test_good_policy_refuses_most_no_evidence_questions(run_agent, tasks):
    rate = _refusal_rate(run_agent, tasks, "good")
    assert rate.value >= 0.5, str(rate)


def test_skips_memory_refuses_markedly_less(run_agent, tasks):
    good = _refusal_rate(run_agent, tasks, "good")
    skips = _refusal_rate(run_agent, tasks, "skips_memory")
    allure.attach(f"good: {good}\nskips_memory: {skips}", name="拒答率",
                  attachment_type=allure.attachment_type.TEXT)  # fmt: skip
    assert skips.value < good.value
    assert skips.num == 0


def test_a_false_premise_still_baits_retrieval_based_answering(run_agent):
    """Documented limitation: related-but-wrong evidence leads the reader to 'answer'."""
    r = run_agent("good", "F_HL_Mini_003")
    assert not refused(r)
