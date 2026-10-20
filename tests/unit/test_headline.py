"""README headline: generated from summary.json, never hand-copied (P1, P4)."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from agentmemeval.analysis.compare import compare
from agentmemeval.analysis.scoring import tally_by
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.cli import main
from agentmemeval.errors import IncomparableError, StageError
from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, RunConfig
from agentmemeval.pipeline.runner import load_run, run_pipeline
from agentmemeval.report.headline import (
    END,
    START,
    accuracy,
    headline,
    paired,
    render_markdown,
    render_svg,
    splice,
    stale,
)
from agentmemeval.report.publish import _run_summary

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "live-v1" / "summary.json"
README = ROOT / "README.md"
FIGURE = ROOT / "docs" / "img" / "live-v1-accuracy.svg"
FIXTURE = ROOT / "tests" / "fixtures" / "mini_corpus"


def test_committed_readme_and_figure_match_the_published_summary():
    """The guard that keeps the README's first screen honest: re-publish without
    regenerating, or edit a number by hand, and this fails."""
    assert stale(SUMMARY, README, FIGURE) == []


@pytest.fixture(scope="module")
def demo_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("headline")
    for arm in ("null", "bm25"):
        cfg = RunConfig(
            run_id=f"h-{arm}",
            dataset=DatasetSpec(kind="fixture", path=FIXTURE.as_posix()),
            arm=ArmSpec(name=arm),
        )
        run_pipeline(root / arm, cfg)
    return root


def _summary(root: Path) -> dict:
    arms = {}
    for arm in ("null", "bm25"):
        s = _run_summary(root / arm)
        s["probes"] = {"round_status": "measured"}
        arms[arm] = s
    return {"arms": arms, "v2": {}}


def test_numbers_equal_the_report_code_on_real_runs(demo_runs):
    """Summary rows are not JudgeRecords, so the counting is re-implemented;
    pin it to tally() and compare() on the same runs."""
    summary = _summary(demo_runs)
    runs = {arm: load_run(demo_runs / arm) for arm in ("null", "bm25")}
    for arm, run in runs.items():
        by_class = tally_by(run.judge_records, lambda r: r.window_class.value)
        for wc in ("A", "D"):
            want = by_class[wc].accuracy
            got = accuracy(summary["arms"][arm]["questions"], wc)
            assert (got.num, got.den) == (want.num, want.den), (arm, wc)
    want = compare(
        "null",
        runs["null"].judge_records,
        "bm25",
        runs["bm25"].judge_records,
        classes={WindowClass.A},
    )
    got = paired(
        "bm25",
        summary["arms"]["bm25"]["questions"],
        "null",
        summary["arms"]["null"]["questions"],
        "A",
    )
    assert (got.only_treatment, got.only_control) == (want.only_treatment, want.only_control)
    assert got.p_value == want.p_value


def _row(qid, outcome="delivered", verdict="CORRECT", wc="A"):
    return {"qid": qid, "window_class": wc, "outcome": outcome, "verdict": verdict}


def test_env_failure_drops_the_pair_and_undelivered_counts_as_a_miss():
    t = [_row("q1"), _row("q2"), _row("q3", outcome="env_failure", verdict=None)]
    c = [_row("q1", outcome="undelivered", verdict=None), _row("q2"), _row("q3", verdict="WRONG")]
    p = paired("t", t, "c", c, "A")
    assert (p.only_treatment, p.only_control) == (1, 0)


def test_a_delivered_answer_without_a_verdict_is_refused():
    with pytest.raises(StageError, match="no verdict"):
        accuracy([_row("q1", verdict=None)], "A")


@pytest.fixture(scope="module")
def live_summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda s: s["arms"]["mem0"]["meta"].update(corpus_id="other"), IncomparableError),
        (
            lambda s: s["v2"]["bm25"]["meta"].update(judge_prompt_version="llm-v3"),
            IncomparableError,
        ),
        (
            lambda s: s["v2"]["null"]["stage_records_sha256"].update(answer="0" * 64),
            IncomparableError,
        ),
        (lambda s: s["v2"].pop("mem0"), IncomparableError),
        (lambda s: s["arms"]["bm25"]["probes"].update(round_status="not_measured"), StageError),
    ],
    ids=["corpus", "mixed-judge-prompts", "v2-graded-other-answers", "arm-missing", "not-measured"],
)
def test_incomparable_or_unmeasured_inputs_are_refused(live_summary, mutate, error):
    s = copy.deepcopy(live_summary)
    mutate(s)
    with pytest.raises(error):
        headline(s)


def test_every_rate_in_the_table_carries_its_interval(live_summary):
    md = render_markdown(headline(live_summary), source="s", figure="f.svg")
    table = [line for line in md.splitlines() if line.startswith("| null")]
    assert table and all(cell.count("[") == 1 for cell in table[0].split(" | ")[1:])


def test_figure_encodes_version_by_shape_as_well_as_color(live_summary):
    h = headline(live_summary)
    svg = render_svg(h)
    cells = len(h.arms) * 2  # two panels
    assert svg.count('<circle class="s0') == cells + 1  # + legend
    assert svg.count('<rect class="s1') == cells + 1
    assert "评分员 v1" in svg and "评分员 v2" in svg
    assert "prefers-color-scheme: dark" in svg


def test_splice_refuses_missing_or_repeated_markers():
    assert splice(f"a\n{START}\nold\n{END}\nb", "NEW") == "a\nNEW\nb"
    with pytest.raises(StageError):
        splice("no markers", "NEW")
    with pytest.raises(StageError):
        splice(f"{START}{END}{START}{END}", "NEW")


def test_cli_check_fails_on_a_hand_edited_number(tmp_path, capsys):
    # same layout as the repo: the block embeds paths relative to the README
    copies = {p: tmp_path / p.relative_to(ROOT) for p in (README, FIGURE, SUMMARY)}
    for src, dst in copies.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    readme = copies[README]
    args = ["readme", "--summary", str(copies[SUMMARY])]
    args += ["--readme", str(readme), "--figure", str(copies[FIGURE])]
    assert main([*args, "--check"]) == 0
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("59/110", "69/110", 1), encoding="utf-8")
    assert main([*args, "--check"]) == 1
    assert "stale" in capsys.readouterr().out
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
