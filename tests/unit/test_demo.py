"""M1 acceptance: `agentmemeval demo` runs offline, fast, and separates bm25 from null."""

from __future__ import annotations

import time

import pytest

from agentmemeval.analysis.scoring import tally_by
from agentmemeval.cli import main
from agentmemeval.pipeline.runner import Action, run_pipeline

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    started = time.perf_counter()
    code = main(["demo", "--out", str(out)])
    return out, code, time.perf_counter() - started


def test_demo_succeeds_well_within_two_minutes(demo):
    _, code, seconds = demo
    assert code == 0
    assert seconds < 120


def test_bm25_clearly_beats_null_on_questions_with_evidence(demo):
    out, _, _ = demo
    by_arm = {}
    for arm in ("null", "bm25"):
        res = run_pipeline(out / arm)  # resumes: reads the stage files the demo wrote
        assert set(res.actions.values()) == {Action.REUSED}
        by_arm[arm] = tally_by(res.judge_records, lambda r: r.evidence_status.value)["located"]
    null, bm25 = by_arm["null"].accuracy, by_arm["bm25"].accuracy
    assert null.den == bm25.den
    assert bm25.num - null.num >= null.den // 3, (null, bm25)


def test_demo_prints_the_table_with_denominators(demo, capsys):
    out, _, _ = demo
    assert main(["run", "--run-dir", str(out / "bm25")]) == 0
    text = capsys.readouterr().out
    assert "corpus_id: mini_corpus@" in text
    assert "/13 = " in text
    assert "qualitative only" in text


def test_run_reports_refusals_without_a_traceback(tmp_path, capsys):
    assert main(["run", "--run-dir", str(tmp_path)]) == 1
    assert "StageError" in capsys.readouterr().err
