"""Agent batch-evaluation files and summaries (METHODOLOGY 2.4)."""

from __future__ import annotations

import json

import pytest

from agentmemeval.agent.eval import (
    find_agent_runs,
    load_agent_run,
    run_agent_eval,
    summarize,
    write_agent_run,
)
from agentmemeval.analysis.scoring import Ratio
from agentmemeval.errors import StageError
from agentmemeval.pipeline.config import build_components, default_eval_config

pytestmark = pytest.mark.unit

METRICS = {"outcome", "n_llm_calls", "n_tool_calls_by_name", "budget_hit",
           "max_same_tool_streak", "route", "latency_breakdown", "ttft_ms"}  # fmt: skip


@pytest.fixture(scope="module")
def components():
    return build_components(default_eval_config())


@pytest.fixture(scope="module")
def good(components):
    return run_agent_eval("good", components)


def test_records_carry_the_methodology_metrics(good):
    for record in good.records:
        assert set(record) >= METRICS
    assert good.meta.clock == "simulated"
    assert good.meta.simulated_latency is not None  # the report must say timings are simulated
    assert good.meta.corpus_id.startswith("mini_corpus@")


def test_write_load_round_trip_and_tamper_refusal(good, tmp_path):
    path = write_agent_run(tmp_path / "good.json", good)
    assert load_agent_run(path) == good
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["records"][0]["outcome"] = (
        "delivered" if raw["records"][0]["outcome"] != "delivered" else "undelivered"
    )
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StageError, match="edited by hand"):
        load_agent_run(path)


def test_summary_denominators(components):
    s = summarize(run_agent_eval("narrator", components))
    assert s.delivery == Ratio(0, s.n)
    assert s.accuracy.den == 0  # nothing delivered: accuracy undefined, not 0%
    assert s.end_to_end == Ratio(0, 20)  # 16 memory + 4 web tasks have a gold answer


def test_real_clock_mode_is_recorded(components):
    run = run_agent_eval("good", components, real_clock=True)
    assert run.meta.clock == "real"
    assert run.meta.simulated_latency is None
    assert all(r["ttft_ms"] is None for r in run.records)  # not streamed: no invented TTFT


def test_find_agent_runs_puts_the_reference_first(components, tmp_path):
    for policy in ("narrator", "good"):
        write_agent_run(tmp_path / f"{policy}.json", run_agent_eval(policy, components))
    assert [r.meta.policy for r in find_agent_runs(tmp_path)] == ["good", "narrator"]
    assert find_agent_runs(tmp_path / "missing") == []
