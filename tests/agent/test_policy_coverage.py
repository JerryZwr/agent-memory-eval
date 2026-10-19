"""Test the tests (METHODOLOGY P5, 3.1): every known-defective policy is caught.

Runs every policy over the whole task set through the batch evaluator (which
uses the same checks as the dimension tests). The reference policy must pass
every check; each defective policy must fail the check that targets its defect.
"""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.eval import run_agent_eval, summarize
from agentmemeval.agent.policies import BAD_POLICIES

pytestmark = [pytest.mark.agent, allure.feature("测试的测试：坏策略都被抓住")]

# policy -> the checks that must catch it
CAUGHT_BY = {
    "stubborn_searcher": {"converged", "delivered"},
    "skips_memory": {"trajectory"},
    "narrator": {"delivered"},
    "misrouted": {"route", "fallback", "trajectory"},
    "acknowledger": {"delivered"},
}


@pytest.fixture(scope="module")
def summaries(components):
    return {p: summarize(run_agent_eval(p, components)) for p in ("good", *BAD_POLICIES)}


def test_every_defective_policy_has_an_expected_catcher():
    assert set(CAUGHT_BY) == set(BAD_POLICIES)


def test_reference_policy_passes_every_check(summaries):
    assert summaries["good"].check_failures == dict.fromkeys(summaries["good"].check_failures, 0)


@pytest.mark.parametrize("policy", BAD_POLICIES)
def test_defective_policy_is_caught_by_its_check(summaries, policy):
    failures = summaries[policy].check_failures
    allure.attach(str(failures), name="check failures", attachment_type=allure.attachment_type.TEXT)
    for check in CAUGHT_BY[policy]:
        assert failures[check] > 0, f"{policy} slipped past the {check} check"
