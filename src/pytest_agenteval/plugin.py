"""pytest plugin for evaluation suites (METHODOLOGY 3.1).

| feature                     | mechanism                                                    |
|-----------------------------|--------------------------------------------------------------|
| requires_env(...) marker    | probes run once, cached; tryfirst runtest_setup skips        |
| post-session probes         | pytest_sessionfinish; red -> whole round not_measured        |
| --eval-baseline             | per-question diff in the terminal summary                    |
| --eval-qids                 | pytest_generate_tests: one case per question, id = qid       |
| --live                      | live tests skipped unless given                              |
| probes as a hook            | pytest_agenteval_probes (firstresult); trylast default here  |
| not_measured(request, why)  | mid-run data failure -> skipped as "not measured", not ERROR |
| bad probe name / baseline   | pytest.UsageError                                            |

Loaded automatically through the `pytest11` entry point when the package is
installed (or explicitly with `-p pytest_agenteval.plugin`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pytest

from agentmemeval.analysis.baseline import (
    Baseline,
    BaselineEntry,
    DiffStatus,
    comparability_problems,
    diff,
    load_baseline,
)
from agentmemeval.errors import AgentMemEvalError
from agentmemeval.pipeline.config import Components, RunConfig, build_components
from agentmemeval.pipeline.config import default_eval_config as _default_eval_config
from agentmemeval.pipeline.records import FileMeta
from agentmemeval.probes import (
    ProbeResult,
    ProbeStatus,
    closure,
    enforce_dependencies,
    run_probes,
    unknown_probes,
)
from agentmemeval.provenance import git_commit, utc_now
from pytest_agenteval import hookspecs
from pytest_agenteval.allure_files import write_allure_categories, write_allure_environment

NOT_MEASURED_PREFIX = "not_measured: "
PROBE_SKIP_PREFIX = "environment probe "
LIVE_SKIP_REASON = "live test: pass --live to run (needs .env, costs money)"


@dataclass
class AgentEvalState:
    config: pytest.Config
    live: bool
    eval_config: RunConfig
    qids_file: Path | None
    baseline: Baseline | None
    allow_incomparable: bool
    json_report: Path | None
    required: set[str] = field(default_factory=set)
    pre: dict[str, ProbeResult] | None = None
    post: dict[str, ProbeResult] | None = None
    not_measured: list[tuple[str, str]] = field(default_factory=list)
    results: dict[str, BaselineEntry] = field(default_factory=dict)
    results_meta: FileMeta | None = None
    incomparable: list[str] = field(default_factory=list)
    qids: list[str] | None = None  # validated --eval-qids, set at session start

    @cached_property
    def components(self) -> Components:
        return build_components(self.eval_config)

    def probe(self, phase: str) -> dict[str, ProbeResult]:
        keys = closure(self.required)
        raw = self.config.hook.pytest_agenteval_probes(config=self.config, keys=keys, phase=phase)
        return enforce_dependencies(keys, raw)

    def pre_probes(self) -> dict[str, ProbeResult]:
        if self.pre is None:
            self.pre = self.probe("pre")
        return self.pre

    @property
    def round_not_measured(self) -> list[str]:
        """Why the whole round is not measured (post-probe not green), or []."""
        if self.post is None:
            return []
        return [
            f"post-probe {r.key} {r.status.value}: {r.detail}"
            for r in self.post.values()
            if not r.ok
        ]


STATE = pytest.StashKey[AgentEvalState]()


def _state(config: pytest.Config) -> AgentEvalState:
    return config.stash[STATE]


# --------------------------------------------------------------------------- registration


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agenteval", "agent-memory evaluation")
    group.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests marked `live` (real LLM / memory-service calls; needs .env, costs money)",
    )
    group.addoption("--eval-config", metavar="PATH", help="run configuration (run.json format)")
    group.addoption("--eval-qids", metavar="FILE", help="question ids to evaluate, one per line")
    group.addoption("--eval-baseline", metavar="PATH", help="frozen baseline to diff against")
    group.addoption(
        "--eval-allow-incomparable",
        action="store_true",
        default=False,
        help="diff against a baseline with different corpus/judge settings (flagged loudly)",
    )
    group.addoption("--eval-json-report", metavar="PATH", help="write the round summary as JSON")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_env(*keys): skip unless these environment probes are green "
        "(llm, memory_service, corpus)",
    )
    config.addinivalue_line("markers", "live: real network calls; skipped unless --live")

    def opt_path(name: str) -> Path | None:
        value = config.getoption(name)
        return Path(value) if value else None

    cfg_path = opt_path("--eval-config")
    try:
        eval_config = (
            RunConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))
            if cfg_path
            else _default_eval_config()
        )
        baseline_path = opt_path("--eval-baseline")
        baseline = load_baseline(baseline_path) if baseline_path else None
    except (OSError, ValueError, AgentMemEvalError) as exc:
        raise pytest.UsageError(f"pytest_agenteval: {exc}") from None
    config.stash[STATE] = AgentEvalState(
        config=config,
        live=config.getoption("--live"),
        eval_config=eval_config,
        qids_file=opt_path("--eval-qids"),
        baseline=baseline,
        allow_incomparable=config.getoption("--eval-allow-incomparable"),
        json_report=opt_path("--eval-json-report"),
    )


@pytest.hookimpl(trylast=True)
def pytest_agenteval_probes(config: pytest.Config, keys: list[str], phase: str):
    """Default probes: the real checks, against the configured evaluation stack."""
    return run_probes(_state(config).components, keys)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Validate inputs before collection, so bad ones are usage errors, not collection errors."""
    state = _state(session.config)
    if state.qids_file is not None:
        state.qids = _read_qids(state)
    if state.baseline is None:
        return
    try:
        current = state.components.meta
    except AgentMemEvalError as exc:
        raise pytest.UsageError(f"pytest_agenteval: cannot load the eval config: {exc}") from None
    problems = comparability_problems(state.baseline.meta, current)
    if problems and not state.allow_incomparable:
        raise pytest.UsageError(
            "pytest_agenteval: baseline is not comparable with this run: " + "; ".join(problems)
        )
    state.incomparable = problems


# --------------------------------------------------------------------------- collection


def _requires_env(item: pytest.Item) -> list[str]:
    keys: list[str] = []
    for mark in item.iter_markers(name="requires_env"):
        keys.extend(mark.args)
    return keys


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    state = _state(config)
    skip_live = pytest.mark.skip(reason=LIVE_SKIP_REASON)
    for item in items:
        keys = _requires_env(item)
        bad = unknown_probes(keys)
        if bad:
            # A misspelt probe name must not be treated as "not run" and silently skipped.
            raise pytest.UsageError(
                f"{item.nodeid}: unknown probe(s) {bad} in requires_env; known: llm, "
                "memory_service, corpus"
            )
        state.required.update(keys)
        if "live" in item.keywords and not state.live:
            item.add_marker(skip_live)


def _read_qids(state: AgentEvalState) -> list[str]:
    known = {q.qid for q in state.components.questions()}
    if state.qids_file is None:
        return sorted(known)
    try:
        lines = state.qids_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise pytest.UsageError(f"--eval-qids: {exc}") from None
    qids = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    unknown = sorted(set(qids) - known)
    if unknown:
        raise pytest.UsageError(
            f"--eval-qids: {unknown[:5]} are not questions this run asks (wrong id, or outside "
            "the evidence window)"
        )
    if len(set(qids)) != len(qids):
        raise pytest.UsageError("--eval-qids: duplicate question ids")
    return qids


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "eval_qid" in metafunc.fixturenames:
        state = _state(metafunc.config)
        qids = state.qids if state.qids is not None else _read_qids(state)
        metafunc.parametrize("eval_qid", qids, ids=qids)


# --------------------------------------------------------------------------- running


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate before fixtures run: a fixture touching a dead service would ERROR, not skip."""
    keys = _requires_env(item)
    if not keys:
        return
    results = _state(item.config).pre_probes()
    for key in closure(keys):
        r = results[key]
        if not r.ok:
            pytest.skip(f"{PROBE_SKIP_PREFIX}{key} is {r.status.value}: {r.detail}")


def not_measured(request: pytest.FixtureRequest, reason: str) -> None:
    """Skip the current test as *not measured* and record it.

    For data fetched mid-run that turns out unusable for environmental
    reasons (a probe was green but the service failed later). The test is
    neither failed nor errored, and the rest of the round stays valid.
    """
    _state(request.config).not_measured.append((request.node.nodeid, reason))
    pytest.skip(f"{NOT_MEASURED_PREFIX}{reason}")


class EvalRecorder:
    """Collects per-question results for the baseline diff."""

    def __init__(self, state: AgentEvalState):
        self._state = state

    def set_meta(self, meta: FileMeta) -> None:
        self._state.results_meta = meta

    def __call__(self, entry: BaselineEntry) -> None:
        if entry.qid in self._state.results:
            raise ValueError(f"{entry.qid} recorded twice")
        self._state.results[entry.qid] = entry


@pytest.fixture(scope="session")
def eval_config(pytestconfig: pytest.Config) -> RunConfig:
    return _state(pytestconfig).eval_config


@pytest.fixture(scope="session")
def eval_components(pytestconfig: pytest.Config) -> Components:
    return _state(pytestconfig).components


@pytest.fixture(scope="session")
def eval_record(pytestconfig: pytest.Config) -> EvalRecorder:
    return EvalRecorder(_state(pytestconfig))


@pytest.fixture(scope="session")
def eval_baseline(pytestconfig: pytest.Config) -> Baseline | None:
    return _state(pytestconfig).baseline


# --------------------------------------------------------------------------- reporting


def _diff_counts(state: AgentEvalState) -> dict[str, int] | None:
    if state.baseline is None:
        return None
    meta = state.results_meta or state.baseline.meta
    result = diff(state.baseline, meta, list(state.results.values()), allow_incomparable=True)
    return {s.value: n for s, n in result.counts.items()}


def _summary(state: AgentEvalState) -> dict[str, Any]:
    def dump(results: dict[str, ProbeResult] | None) -> dict[str, Any] | None:
        if results is None:
            return None
        return {k: {"status": r.status.value, "detail": r.detail} for k, r in results.items()}

    reasons = state.round_not_measured
    return {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "round_status": "not_measured" if reasons else "measured",
        "round_not_measured_reasons": reasons,
        "probes": {"pre": dump(state.pre), "post": dump(state.post)},
        "not_measured_tests": [{"nodeid": n, "reason": r} for n, r in state.not_measured],
        "baseline": None
        if state.baseline is None
        else {
            "name": state.baseline.name,
            "diff": _diff_counts(state),
            "incomparable": state.incomparable,
        },
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if STATE not in config.stash:
        return
    state = _state(config)
    if state.pre is not None:
        # Faults can start mid-run; a pre-probe cannot see them. Re-check the same keys.
        state.post = state.probe("post")
    if state.json_report:
        state.json_report.parent.mkdir(parents=True, exist_ok=True)
        state.json_report.write_text(
            json.dumps(_summary(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    alluredir = getattr(config.option, "allure_report_dir", None)
    if alluredir and not getattr(config, "workerinput", None):
        meta = state.results_meta
        write_allure_environment(Path(alluredir), state.eval_config, meta, state.round_not_measured)
        write_allure_categories(Path(alluredir))


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if STATE not in config.stash:
        return
    state = _state(config)
    if state.pre is None and state.baseline is None and not state.not_measured:
        return
    tr = terminalreporter
    tr.section("agent-eval")
    for phase, results in (("pre", state.pre), ("post", state.post)):
        if results:
            line = ", ".join(f"{k}={r.status.value}" for k, r in results.items())
            tr.write_line(f"{phase}-probes: {line}")
    if state.not_measured:
        tr.write_line(f"not_measured tests: {len(state.not_measured)}")
        for nodeid, reason in state.not_measured:
            tr.write_line(f"  {nodeid}: {reason}")
    counts = _diff_counts(state)
    if counts is not None:
        assert state.baseline is not None
        text = ", ".join(f"{s.value}={counts[s.value]}" for s in DiffStatus)
        tr.write_line(f"baseline {state.baseline.name!r}: {text}")
        for problem in state.incomparable:
            tr.write_line(f"INCOMPARABLE BASELINE: {problem}", red=True, bold=True)
    reasons = state.round_not_measured
    if reasons:
        tr.write_line("ROUND NOT MEASURED: results of this session must not be used", red=True,
                      bold=True)  # fmt: skip
        for reason in reasons:
            tr.write_line(f"  {reason}", red=True)


__all__ = ["NOT_MEASURED_PREFIX", "ProbeResult", "ProbeStatus", "not_measured"]
