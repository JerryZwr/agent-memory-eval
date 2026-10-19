"""Paired arm comparison and its wording rules (METHODOLOGY 1.7)."""

from __future__ import annotations

import pytest

from agentmemeval.analysis.compare import compare, format_p
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.pipeline.records import JudgeRecord

pytestmark = pytest.mark.unit
A, D = WindowClass.A, WindowClass.D


def _r(qid, passed, outcome=Outcome.DELIVERED, wc=A):
    delivered = outcome is Outcome.DELIVERED
    return JudgeRecord(
        qid=qid,
        qtype="F_SH",
        evidence_status="located",
        window_class=wc,
        is_mc=False,
        outcome=outcome,
        gold="g",
        answer="a",
        verdict=("CORRECT" if passed else "WRONG") if delivered else None,
    )


def _arms(results):
    """results: list of (control_passed, treatment_passed)."""
    ctrl = [_r(f"q{i}", c) for i, (c, _) in enumerate(results)]
    trt = [_r(f"q{i}", t) for i, (_, t) in enumerate(results)]
    return ctrl, trt


def test_counts_cells_and_uses_only_discordant_pairs():
    ctrl, trt = _arms([(False, True)] * 7 + [(True, True)] * 3 + [(False, False)] * 2)
    c = compare("null", ctrl, "bm25", trt, classes={A})
    assert (c.n_pairs, c.both_pass, c.only_treatment, c.only_control, c.neither) == (12, 3, 7, 0, 2)
    assert c.p_value == pytest.approx(2 * 0.5**7)
    assert c.difference == pytest.approx(7 / 12)
    assert c.significant
    assert c.verdict().startswith("bm25 高于 null")


def test_without_significance_it_never_says_better():
    ctrl, trt = _arms([(False, True)] * 5 + [(True, True)] * 10)
    c = compare("null", ctrl, "bm25", trt, classes={A})
    assert not c.significant
    assert c.underpowered  # 5 discordant pairs can never reach 0.05
    assert "未观察到显著差异" in c.verdict()
    assert "高于" not in c.verdict()


def test_env_failure_on_either_side_drops_the_pair_but_undelivered_counts_as_fail():
    ctrl = [_r("q1", False), _r("q2", False), _r("q3", True)]
    trt = [_r("q1", True), _r("q2", True, outcome=Outcome.ENV_FAILURE),
           _r("q3", False, outcome=Outcome.UNDELIVERED)]  # fmt: skip
    c = compare("null", ctrl, "bm25", trt, classes={A})
    assert c.n_pairs == 2
    assert (c.only_treatment, c.only_control) == (1, 1)


def test_only_requested_window_classes_and_shared_qids_are_paired():
    ctrl = [_r("a", False), _r("d", True, wc=D), _r("only_ctrl", True)]
    trt = [_r("a", True), _r("d", False, wc=D)]
    c = compare("null", ctrl, "bm25", trt, classes={A})
    assert c.n_pairs == 1
    assert compare("null", ctrl, "bm25", trt, classes={D}).only_control == 1


@pytest.mark.parametrize(
    ("p", "text"), [(0.03125, "p = 0.031"), (0.0004, "p < 0.001"), (1.0, "p = 1.000")]
)
def test_format_p(p, text):
    assert format_p(p) == text


def test_no_pairs():
    c = compare("null", [], "bm25", [], classes={A})
    assert c.n_pairs == 0
    assert c.verdict() == "没有可配对的题，无法比较"
