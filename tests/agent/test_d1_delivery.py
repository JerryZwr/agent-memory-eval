"""Dimension 1 · delivery: a normal return is not delivery; an acknowledgement is not an answer."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import check_delivered
from agentmemeval.agent.policies import ACK, AcknowledgeThenAnswer
from agentmemeval.analysis.outcome import Outcome, UndeliveredReason

pytestmark = [
    pytest.mark.agent,
    allure.feature("D1 交付"),
    allure.severity(allure.severity_level.CRITICAL),
]


def test_good_policy_delivers_a_real_answer(run_agent):
    r = run_agent("good", "F_SH_Mini_001")
    assert check_delivered(r)
    assert "PostgreSQL 16" in r.final_answer


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        ("narrator", UndeliveredReason.NARRATION),
        ("acknowledger", UndeliveredReason.ACKNOWLEDGEMENT),
    ],
)
def test_returning_normally_is_not_delivering(run_agent, policy, reason):
    r = run_agent(policy, "F_SH_Mini_001")  # run() returned without any exception...
    check = check_delivered(r)
    assert not check  # ...and still nothing was delivered
    assert r.outcome is Outcome.UNDELIVERED
    assert r.undelivered_reason is reason


def test_an_acknowledgement_through_final_answer_is_not_the_answer(run_agent):
    r = run_agent("acknowledger", "MA_C_Mini_002")
    assert r.deliveries == [ACK]  # the delivery channel *was* used
    assert r.final_answer is None
    assert not check_delivered(r)


def test_ack_followed_by_the_real_answer_is_delivered(run_agent):
    """The other direction: an ack is fine when the real answer follows."""
    r = run_agent(AcknowledgeThenAnswer(), "F_SH_Mini_001")
    assert r.deliveries[0] == ACK
    assert len(r.deliveries) == 2
    assert check_delivered(r)
    assert "PostgreSQL 16" in r.final_answer
