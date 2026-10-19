"""The HTML evaluation report: format rules, provenance and refusals (M2 acceptance)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from agentmemeval.analysis.compare import compare, format_p
from agentmemeval.analysis.scoring import tally_by
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.cli import main
from agentmemeval.errors import IncomparableError, StageError
from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, RunConfig
from agentmemeval.pipeline.runner import load_run, run_pipeline
from agentmemeval.pipeline.store import stage_path
from agentmemeval.report.build import find_run_dirs, write_report

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
RATE = re.compile(r'<span class="rate[^"]*">([^<]*)</span>')
RATE_FORMAT = re.compile(r"^\d+/\d+ = \d+% \[\d+%, \d+%\]$|^\d+/0 = n/a$")


def _cfg(arm, path=FIXTURE, first_days=None):
    return RunConfig(
        run_id=f"r-{arm}",
        dataset=DatasetSpec(kind="fixture", path=Path(path).as_posix(), first_days=first_days),
        arm=ArmSpec(name=arm),
    )


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    for arm in ("bm25", "null"):
        run_pipeline(root / arm, _cfg(arm))
    path = write_report(root)
    return root, path.read_text(encoding="utf-8")


def _visible_text(html: str) -> str:
    html = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
    html = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    return html


def test_report_is_written_next_to_the_runs(demo):
    root, _ = demo
    assert (root / "report.html").is_file()


def test_every_rate_has_a_denominator_and_a_95_interval(demo):
    _, html = demo
    rates = RATE.findall(html)
    assert len(rates) > 20
    bad = [r for r in rates if not RATE_FORMAT.match(r)]
    assert bad == []
    # and no percentage-of-count sneaks in outside the macro without an interval
    assert not re.search(r"\d+/\d+ = \d+%(?! \[)", _visible_text(html))


def test_small_samples_are_flagged_as_qualitative_only(demo):
    _, html = demo
    assert "仅供定性参考" in html
    assert 'class="rate small"' in html


def test_numbers_are_the_ones_in_the_run_files(demo):
    root, html = demo
    for arm in ("null", "bm25"):
        run = load_run(root / arm)
        a_class = tally_by(run.judge_records, lambda r: r.window_class.value)["A"]
        assert str(a_class.accuracy) in html
        for stage, sf in run.files.items():
            assert sf.records_sha256[:16] in html, stage


def test_comparison_against_null_uses_the_paired_test_and_its_wording(demo):
    root, html = demo
    null, bm25 = load_run(root / "null"), load_run(root / "bm25")
    c = compare("null", null.judge_records, "bm25", bm25.judge_records, classes={WindowClass.A})
    assert c.verdict() in html
    assert format_p(c.p_value) in html


def test_report_sections_exist(demo):
    _, html = demo
    for heading in ("结论与局限", "计分范围", "总表", "配对比较", "按题型拆分", "按证据范围拆分",
                    "检索 × 判分", "逐题明细", "运行元数据"):  # fmt: skip
        assert heading in html, heading
    assert html.count("<details>") == 16
    assert "<svg" in html


def test_limitations_are_stated_for_the_offline_setup(demo):
    _, html = demo
    assert "合成小语料" in html
    assert "没有经过人工校准" in html


def test_incomplete_run_is_refused(tmp_path):
    run_pipeline(tmp_path / "bm25", _cfg("bm25"))
    stage_path(tmp_path / "bm25", "judge").unlink()
    with pytest.raises(StageError, match="missing"):
        write_report(tmp_path)


def test_runs_on_different_corpora_are_not_put_in_one_report(tmp_path):
    run_pipeline(tmp_path / "a", _cfg("null"))
    run_pipeline(tmp_path / "b", _cfg("bm25", first_days=2))
    with pytest.raises(IncomparableError, match="corpus_id"):
        write_report(tmp_path)


def test_single_arm_report_says_there_is_no_control(tmp_path):
    run_pipeline(tmp_path / "bm25", _cfg("bm25"))
    html = write_report(tmp_path / "bm25").read_text(encoding="utf-8")
    assert "没有 null 对照组" in html


def test_dataset_text_is_html_escaped(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    qa = json.loads((corpus / "qa.json").read_text(encoding="utf-8"))
    qa[0]["Q"] = "Which <i>database</i> is primary? <script>alert(1)</script>"
    (corpus / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    run_pipeline(tmp_path / "run" / "bm25", _cfg("bm25", path=corpus))
    html = write_report(tmp_path / "run").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_find_run_dirs_refuses_a_directory_without_runs(tmp_path):
    with pytest.raises(StageError, match="no run directories"):
        find_run_dirs(tmp_path)


def test_cli_report_command(demo, tmp_path, capsys):
    root, _ = demo
    out = tmp_path / "custom.html"
    assert main(["report", "--run", str(root), "--out", str(out)]) == 0
    assert out.is_file()
    assert "Report:" in capsys.readouterr().out
