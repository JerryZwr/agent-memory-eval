"""Self-tests for pytest_agenteval, each running a real inner pytest session (pytester).

Covers METHODOLOGY 3.1: probe red -> skip; upstream red -> downstream not run
even if the hook claims otherwise; gate before fixtures; post-probe red ->
round not_measured; not_measured() helper; --live; --eval-qids; baseline diff
with all four outcomes; refusals as UsageError.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmemeval.analysis.baseline import Baseline, BaselineEntry
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.pipeline.config import build_components, default_eval_config
from agentmemeval.pipeline.records import FileMeta

pytestmark = pytest.mark.unit

FAKE_PROBES = """
from agentmemeval.probes import ProbeResult, ProbeStatus

CALLS = []
PRE = {pre}
POST = {post}

def pytest_agenteval_probes(config, keys, phase):
    CALLS.append(phase)
    table = PRE if phase == "pre" else POST
    return {{k: ProbeResult(k, ProbeStatus(table.get(k, "pass")), f"fake {{phase}}") for k in keys}}
"""


def fake_probes(pytester: pytest.Pytester, pre: dict | None = None, post: dict | None = None):
    pytester.makeconftest(FAKE_PROBES.format(pre=pre or {}, post=post or {}))


# --------------------------------------------------------------------------- probes and gating


def test_red_probe_skips_with_its_reason(pytester):
    fake_probes(pytester, pre={"llm": "fail"})
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("llm")
        def test_needs_llm():
            pass

        def test_needs_nothing():
            pass
        """
    )
    result = pytester.runpytest("-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*environment probe llm is fail: fake pre*"])


def test_downstream_of_a_red_probe_is_not_run_even_if_the_hook_lies(pytester):
    # The fake hook claims corpus passed although memory_service failed.
    fake_probes(pytester, pre={"memory_service": "fail", "corpus": "pass"})
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("corpus")
        def test_needs_corpus():
            pass
        """
    )
    result = pytester.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    # The first red key in dependency order is memory_service; corpus itself is not_run.
    result.stdout.fnmatch_lines(["*environment probe memory_service is fail*"])
    result.stdout.fnmatch_lines(["*pre-probes: memory_service=fail, corpus=not_run*"])


def test_gate_runs_before_fixtures_so_a_dead_service_skips_instead_of_erroring(pytester):
    fake_probes(pytester, pre={"memory_service": "fail"})
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def memory_client():
            raise ConnectionError("memory service unreachable")

        @pytest.mark.requires_env("memory_service")
        def test_uses_client(memory_client):
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(skipped=1, errors=0)


def test_probes_run_once_per_phase_and_only_when_needed(pytester):
    fake_probes(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("llm")
        @pytest.mark.parametrize("i", range(5))
        def test_many(i):
            pass

        def test_last(request):
            import conftest
            assert conftest.CALLS == ["pre"]
        """
    )
    result = pytester.runpytest("-p", "no:randomly")
    result.assert_outcomes(passed=6)


def test_no_probes_when_nothing_requires_them(pytester):
    pytester.makepyfile("def test_plain():\n    pass\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert "pre-probes" not in result.stdout.str()


def test_unknown_probe_name_is_a_usage_error(pytester):
    fake_probes(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("memroy_service")
        def test_typo():
            pass
        """
    )
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "unknown probe(s) ['memroy_service']" in result.stdout.str() + result.stderr.str()


def test_default_probes_are_green_on_the_offline_stack(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("llm", "memory_service", "corpus")
        def test_all():
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*pre-probes: llm=pass, memory_service=pass, corpus=pass*"])


# --------------------------------------------------------------------------- round status


def test_post_probe_red_marks_the_whole_round_not_measured(pytester):
    fake_probes(pytester, post={"memory_service": "fail"})
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_env("corpus")
        def test_measure():
            pass
        """
    )
    report = pytester.path / "round.json"
    result = pytester.runpytest(f"--eval-json-report={report}")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*ROUND NOT MEASURED*", "*post-probe memory_service fail*"])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["round_status"] == "not_measured"
    assert data["probes"]["pre"]["corpus"]["status"] == "pass"
    assert data["probes"]["post"]["corpus"]["status"] == "not_run"


def test_green_post_probes_leave_the_round_measured(pytester):
    fake_probes(pytester)
    pytester.makepyfile(
        "import pytest\n\n@pytest.mark.requires_env('llm')\ndef test_x():\n    pass\n"
    )
    report = pytester.path / "round.json"
    pytester.runpytest(f"--eval-json-report={report}").assert_outcomes(passed=1)
    assert json.loads(report.read_text(encoding="utf-8"))["round_status"] == "measured"


def test_not_measured_helper_skips_records_and_keeps_the_round(pytester):
    fake_probes(pytester)
    pytester.makepyfile(
        """
        import pytest
        from pytest_agenteval import not_measured

        @pytest.fixture
        def answer(request):
            # probe was green, but this fetch hit a timeout mid-run
            not_measured(request, "answer fetch timed out")

        @pytest.mark.requires_env("llm")
        def test_question(answer):
            pass

        @pytest.mark.requires_env("llm")
        def test_other():
            pass
        """
    )
    report = pytester.path / "round.json"
    result = pytester.runpytest("-rs", f"--eval-json-report={report}")
    result.assert_outcomes(passed=1, skipped=1, errors=0)
    result.stdout.fnmatch_lines(["*not_measured tests: 1*"])
    result.stdout.fnmatch_lines(["*SKIPPED*not_measured: answer fetch timed out*"])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["round_status"] == "measured"
    assert data["not_measured_tests"][0]["reason"] == "answer fetch timed out"


# --------------------------------------------------------------------------- --live


LIVE_FILE = """
import pytest

@pytest.mark.live
def test_paid_call():
    pass

def test_offline():
    pass
"""


def test_live_tests_are_skipped_with_a_reason_by_default(pytester):
    pytester.makepyfile(LIVE_FILE)
    result = pytester.runpytest("-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*live test: pass --live to run*"])


def test_live_flag_runs_them(pytester):
    pytester.makepyfile(LIVE_FILE)
    pytester.runpytest("--live").assert_outcomes(passed=2)


# --------------------------------------------------------------------------- --eval-qids


QID_FILE = """
def test_question(eval_qid):
    assert eval_qid
"""


def test_eval_qids_file_parametrizes_one_case_per_question(pytester):
    pytester.makepyfile(QID_FILE)
    qids = pytester.path / "qids.txt"
    qids.write_text("# smoke subset\nF_SH_Mini_001\n\nMA_C_Mini_002\n", encoding="utf-8")
    result = pytester.runpytest("-v", f"--eval-qids={qids}")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*test_question[[]F_SH_Mini_001[]] PASSED*",
                                 "*test_question[[]MA_C_Mini_002[]] PASSED*"])  # fmt: skip


def test_without_a_qids_file_every_scored_question_is_a_case(pytester):
    pytester.makepyfile(QID_FILE)
    asked = len(build_components(default_eval_config()).questions())
    pytester.runpytest().assert_outcomes(passed=asked)


def test_unknown_qid_is_a_usage_error(pytester):
    pytester.makepyfile(QID_FILE)
    qids = pytester.path / "qids.txt"
    qids.write_text("F_SH_Mini_001\nF_SH_Mini_999\n", encoding="utf-8")
    result = pytester.runpytest(f"--eval-qids={qids}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*F_SH_Mini_999*not questions this run asks*"])


# --------------------------------------------------------------------------- --eval-baseline


def _entry(qid, outcome="delivered", verdict="CORRECT"):
    return BaselineEntry(
        qid=qid,
        qtype=qid[:4],
        window_class=WindowClass.A,
        outcome=Outcome(outcome),
        verdict=verdict if outcome == "delivered" else None,
        retrieval_hit=True,
    )


def _write_baseline(path: Path, **meta_overrides) -> Path:
    meta = build_components(default_eval_config()).meta.model_dump()
    meta.update(meta_overrides)
    baseline = Baseline(
        schema_version=1,
        name="plugin-test",
        frozen_at="2026-10-19T00:00:00+00:00",
        source_run_dir="n/a",
        meta=FileMeta(**meta, created_at="2026-10-19T00:00:00+00:00", git_commit="test"),
        entries=[
            _entry("F_SH_Mini_001", verdict="WRONG"),  # -> fixed
            _entry("F_SH_Mini_002"),  # -> regressed
            _entry("F_SH_Mini_003"),  # -> unchanged
            _entry("F_SH_Mini_004"),  # -> not measured this round (env failure)
            _entry("F_TP_Mini_001"),  # -> not measured (not run at all)
        ],
    )
    path.write_text(baseline.model_dump_json(), encoding="utf-8")
    return path


BASELINE_TESTS = """
import pytest
from agentmemeval.analysis.baseline import BaselineEntry

RESULTS = {
    "F_SH_Mini_001": ("delivered", "CORRECT"),
    "F_SH_Mini_002": ("delivered", "WRONG"),
    "F_SH_Mini_003": ("delivered", "CORRECT"),
    "F_SH_Mini_004": ("env_failure", None),
}

@pytest.mark.parametrize("qid", sorted(RESULTS))
def test_q(qid, eval_record):
    outcome, verdict = RESULTS[qid]
    eval_record(BaselineEntry(qid=qid, qtype=qid[:4], window_class="A", outcome=outcome,
                              verdict=verdict, retrieval_hit=True))
"""


def test_baseline_diff_reports_all_four_outcomes(pytester):
    pytester.makepyfile(BASELINE_TESTS)
    baseline = _write_baseline(pytester.path / "baseline.json")
    result = pytester.runpytest(f"--eval-baseline={baseline}")
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(
        ["*baseline 'plugin-test': fixed=1, regressed=1, unchanged=1, not_measured=2*"]
    )


def test_incomparable_baseline_is_a_usage_error(pytester):
    pytester.makepyfile(BASELINE_TESTS)
    baseline = _write_baseline(pytester.path / "baseline.json", corpus_id="other@x:1..2")
    result = pytester.runpytest(f"--eval-baseline={baseline}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*baseline is not comparable*corpus_id*"])


def test_incomparable_baseline_can_be_forced_but_is_flagged(pytester):
    pytester.makepyfile(BASELINE_TESTS)
    baseline = _write_baseline(pytester.path / "baseline.json", judge_prompt_version="v9")
    result = pytester.runpytest(f"--eval-baseline={baseline}", "--eval-allow-incomparable")
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(["*INCOMPARABLE BASELINE: judge_prompt_version*"])


def test_missing_baseline_file_is_a_usage_error(pytester):
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest(f"--eval-baseline={pytester.path / 'nope.json'}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR


# --------------------------------------------------------------------------- allure files


def test_allure_environment_and_categories_are_written(pytester):
    pytester.makepyfile("def test_x():\n    pass\n")
    alluredir = pytester.path / "allure-results"
    # Subprocess: allure-pytest keeps global listener state, so an in-process inner
    # session with its own --alluredir collides with (and leaks into) an outer session
    # that is itself writing Allure results, as the Pages workflow's does.
    result = pytester.runpytest_subprocess(f"--alluredir={alluredir}")
    result.assert_outcomes(passed=1)
    env = (alluredir / "environment.properties").read_text(encoding="ascii")
    assert "eval_arm=bm25" in env
    assert "round_status=measured" in env
    categories = json.loads((alluredir / "categories.json").read_text(encoding="utf-8"))
    assert {"skipped", "failed", "broken"} <= {s for c in categories for s in c["matchedStatuses"]}
