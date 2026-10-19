"""Environment probes (METHODOLOGY 1.11)."""

from __future__ import annotations

import pytest

from agentmemeval.arms.base import RetrievedItem
from agentmemeval.arms.bm25 import BM25Arm
from agentmemeval.errors import LLMUnavailableError, MemoryServiceError
from agentmemeval.llm.scripted import ScriptedLLM
from agentmemeval.pipeline.config import build_components, default_eval_config
from agentmemeval.probes import (
    ProbeResult,
    ProbeStatus,
    closure,
    enforce_dependencies,
    run_probes,
    unknown_probes,
)

pytestmark = pytest.mark.unit
PASS, FAIL, NOT_RUN = ProbeStatus.PASS, ProbeStatus.FAIL, ProbeStatus.NOT_RUN
ALL = ["llm", "memory_service", "corpus"]


def _status(results):
    return {k: r.status for k, r in results.items()}


class TestDependencyRules:
    def test_closure_adds_upstream_in_evaluation_order(self):
        assert closure(["corpus"]) == ["memory_service", "corpus"]
        assert closure(["corpus", "llm"]) == ["llm", "memory_service", "corpus"]

    def test_unknown_names(self):
        assert unknown_probes(["llm", "memroy_service"]) == ["memroy_service"]

    def test_upstream_red_overrides_a_claimed_pass(self):
        claimed = {
            "memory_service": ProbeResult("memory_service", FAIL, "connection refused"),
            "corpus": ProbeResult("corpus", PASS, "looks fine"),  # cannot be known
        }
        out = enforce_dependencies(["corpus"], claimed)
        assert _status(out) == {"memory_service": FAIL, "corpus": NOT_RUN}
        assert "upstream memory_service" in out["corpus"].detail

    def test_missing_results_are_not_run_not_pass(self):
        out = enforce_dependencies(["llm"], {})
        assert out["llm"].status is NOT_RUN
        assert enforce_dependencies(["llm"], None)["llm"].status is NOT_RUN


@pytest.fixture
def components():
    return build_components(default_eval_config())


class TestDefaultProbes:
    def test_offline_stack_is_green(self, components):
        assert _status(run_probes(components, ALL)) == dict.fromkeys(ALL, PASS)

    def test_llm_outage_is_red(self, components):
        def down(messages):
            raise LLMUnavailableError("503 from upstream")

        components.answer_llm = ScriptedLLM(down, model="scripted/down")
        out = run_probes(components, ["llm"])
        assert out["llm"].status is FAIL
        assert "503" in out["llm"].detail

    def test_memory_outage_marks_corpus_not_run_and_never_calls_it(self, components, monkeypatch):
        calls = []

        def broken_add(self, dataset):
            calls.append("add")
            raise MemoryServiceError("write rejected: read-only replica")

        monkeypatch.setattr(BM25Arm, "add", broken_add)
        out = run_probes(components, ["corpus"])
        assert _status(out) == {"memory_service": FAIL, "corpus": NOT_RUN}
        assert calls == ["add"]  # the corpus probe did not even try

    def test_non_empty_but_wrong_retrieval_fails_the_corpus_probe(self, components, monkeypatch):
        """Same criterion as the analysis: evidence retrieved, not 'something came back'."""

        def polluted(self, query, top_k):
            # e.g. the store returns the asker's own earlier question
            return [
                RetrievedItem(
                    message_id="2025-03-05/Group 2/6", date="2025-03-05",
                    time="2025-03-05 16:05:00", group="Group 2", speaker="Mara Lind",
                    text="Thanks all, great progress this week.", score=1.0,
                )
            ]  # fmt: skip

        monkeypatch.setattr(BM25Arm, "search", polluted)
        out = run_probes(components, ["corpus"])
        assert out["memory_service"].status is PASS
        assert out["corpus"].status is FAIL
        assert "retrieves evidence" in out["corpus"].detail

    def test_control_arm_probes_are_not_applicable_rather_than_red(self):
        """A red probe would skip the control arm itself; it stores nothing by design."""
        out = run_probes(build_components(default_eval_config(arm="null")), ["corpus"])
        assert out["corpus"].status is PASS
        assert "not applicable" in out["corpus"].detail

    def test_corpus_probe_picks_a_canary_and_rechecks_the_same_one(self, components):
        from agentmemeval.probes import canary_of

        arm = components.new_arm()
        arm.add(components.dataset)
        first = run_probes(components, ["corpus"], ingested_arm=arm)["corpus"]
        canary = canary_of(first)
        assert first.ok and canary is not None
        again = run_probes(components, ["corpus"], ingested_arm=arm, canary=canary)["corpus"]
        assert again.ok and canary_of(again) == canary

    def test_memory_probe_writes_to_a_scratch_namespace_only(self, components):
        arm = components.new_arm()
        arm.add(components.dataset)
        before = arm.search("canary word heliotrope", top_k=3)
        assert run_probes(components, ["memory_service"])["memory_service"].ok
        assert arm.search("canary word heliotrope", top_k=3) == before  # measured store untouched
