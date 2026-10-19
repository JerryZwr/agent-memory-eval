"""Dimension 3 · routing: accuracy and confusion against the agent's own declared rule."""

from __future__ import annotations

import allure
import pytest

from agentmemeval.agent.checks import check_route, precision_recall, routing_confusion
from agentmemeval.agent.core import ROUTING_RULE

pytestmark = [pytest.mark.agent, allure.feature("D3 路由")]


def _route_all(run_agent, tasks, policy):
    return [(spec.expected_route, run_agent(policy, tid).route) for tid, spec in tasks.items()]


def test_expected_labels_come_from_the_declared_rule(tasks):
    assert "MUST go to `memory`" in ROUTING_RULE
    assert {t.expected_route for t in tasks.values() if t.kind == "memory"} == {"memory"}


def test_good_policy_routes_every_class_correctly(run_agent, tasks):
    matrix = routing_confusion(_route_all(run_agent, tasks, "good"))
    for route, (precision, recall) in precision_recall(matrix).items():
        assert (precision, recall) == (1.0, 1.0), route


def test_misrouted_sends_memory_questions_to_web(run_agent, tasks):
    matrix = routing_confusion(_route_all(run_agent, tasks, "misrouted"))
    memory_total = sum(matrix["memory"].values())
    assert matrix["memory"].get("web", 0) / memory_total == 1.0  # every memory question
    assert precision_recall(matrix)["memory"][1] == 0.0  # memory recall


def test_right_label_wrong_behaviour_is_caught(run_agent):
    """skips_memory labels the route correctly, then never touches memory."""
    r = run_agent("skips_memory", "F_SH_Mini_001")
    assert r.route == "memory"
    check = check_route(r, "memory")
    assert not check
    assert "first called no tool" in check.reason


def test_the_first_routing_decision_is_the_one_scored(run_agent):
    r = run_agent("good", "WEB_001")
    assert len(r.trace.of_kind("route")) == 1
    assert check_route(r, "web")
