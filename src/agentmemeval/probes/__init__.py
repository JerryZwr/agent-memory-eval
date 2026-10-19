"""Environment probes (METHODOLOGY 1.11).

Three probes with a dependency between them:

    llm             - the answering LLM produces a non-empty reply
    memory_service  - the memory backend accepts writes and serves a search
    corpus          - (needs memory_service) a known question still retrieves its evidence

Rules this module enforces:

* upstream red -> downstream NOT_RUN, never FAIL: a probe that cannot tell
  "I am broken" from "my upstream is broken" is worse than an honest "not run".
  `enforce_dependencies` applies this even to results someone else produced
  (the pytest hook may be overridden, and an override may claim too much);
* "reachable" is not "usable": the memory probe writes (ingests), it does not
  just connect;
* the corpus probe judges success with `retrieval_hit`, the very function the
  analysis uses, so the probe cannot be green under the failure being measured.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.arms.base import MemoryArm
from agentmemeval.data.loaders import _build_dataset
from agentmemeval.data.models import Dataset, EvidenceStatus, Message, Question, message_id
from agentmemeval.errors import InfraError
from agentmemeval.llm.openai_compat import OpenAICompatLLM
from agentmemeval.pipeline.config import LIVE_ARMS, Components
from agentmemeval.pipeline.prompts import render_answer_prompt
from agentmemeval.pipeline.records import SearchRecord


class ProbeStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class ProbeResult:
    key: str
    status: ProbeStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ProbeStatus.PASS


# key -> keys it depends on. Order is a valid evaluation order.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "llm": (),
    "memory_service": (),
    "corpus": ("memory_service",),
}
KNOWN_PROBES = tuple(DEPENDENCIES)


def unknown_probes(keys: Iterable[str]) -> list[str]:
    return sorted(set(keys) - set(DEPENDENCIES))


def closure(keys: Iterable[str]) -> list[str]:
    """`keys` plus everything they depend on, in evaluation order."""
    wanted = set(keys)
    changed = True
    while changed:
        extra = {d for k in wanted for d in DEPENDENCIES.get(k, ())} - wanted
        changed = bool(extra)
        wanted |= extra
    return [k for k in KNOWN_PROBES if k in wanted]


def enforce_dependencies(
    keys: Iterable[str], results: Mapping[str, ProbeResult] | None
) -> dict[str, ProbeResult]:
    """Normalise probe results: missing -> NOT_RUN; anything below a red upstream -> NOT_RUN."""
    results = dict(results or {})
    out: dict[str, ProbeResult] = {}
    for key in closure(keys):
        red = [d for d in DEPENDENCIES[key] if not out[d].ok]
        if red:
            out[key] = ProbeResult(
                key, ProbeStatus.NOT_RUN, f"not run: upstream {', '.join(red)} is not green"
            )
        elif key in results:
            out[key] = results[key]
        else:
            out[key] = ProbeResult(key, ProbeStatus.NOT_RUN, "no probe implementation returned it")
    return out


def _guard(key: str, fn: Callable[[], str | None]) -> ProbeResult:
    try:
        return ProbeResult(key, ProbeStatus.PASS, str(fn()))
    except InfraError as exc:
        return ProbeResult(key, ProbeStatus.FAIL, f"{type(exc).__name__}: {exc}")
    except AssertionError as exc:
        return ProbeResult(key, ProbeStatus.FAIL, str(exc))


CANARY_CANDIDATES = 10
CANARY_PREFIX = "canary="


def _canary_dataset(components: Components) -> Dataset:
    """Two synthetic messages for the write probe; written to a scratch namespace only."""
    day = components.dataset.days[0]
    texts = ["Probe check: the canary word for this run is heliotrope.",
             "Probe check: heliotrope is only used by the environment probe."]  # fmt: skip
    msgs = [
        Message(id=message_id(day, "probe", i), date=day, group="probe", index=i,
                speaker="probe", time=f"{day} 00:00:0{i}", text=t)
        for i, t in enumerate(texts, start=1)
    ]  # fmt: skip
    return _build_dataset("probe", msgs, [])


def _hits(arm: MemoryArm, q: Question, top_k: int) -> bool | None:
    items = arm.search(q.question, top_k=top_k)
    record = SearchRecord(qid=q.qid, qtype=q.qtype, evidence_status=q.evidence_status,
                          window_class=WindowClass.A, query=q.question, items=items)  # fmt: skip
    # Same criterion as the analysis. Extracted memories have no source message, so for
    # them the date-level criterion is the one that applies (D-006).
    criterion = HitCriterion.MESSAGE
    if any(it.kind != "message" for it in items):
        criterion = HitCriterion.DATE
    return retrieval_hit(record, q, criterion)


def canary_of(result: ProbeResult) -> str | None:
    """The canary question a passing corpus probe used (to re-check it after the run)."""
    detail = result.detail
    return detail.split(CANARY_PREFIX, 1)[1].split()[0] if CANARY_PREFIX in detail else None


def run_probes(
    components: Components,
    keys: Iterable[str],
    *,
    ingested_arm: MemoryArm | None = None,
    canary: str | None = None,
) -> dict[str, ProbeResult]:
    """The default probes, run against the components a measurement would use.

    * llm: one tiny completion;
    * memory_service: a *write* into a scratch namespace (never the measured one,
      METHODOLOGY 1.12), then a search that must find it;
    * corpus: against `ingested_arm` (the run's own store) when given. Without a
      `canary`, the first in-window question that retrieves its evidence becomes
      the canary; with one, that same question must still retrieve it. Offline
      arms without an ingested store get a fresh one; live arms report NOT_RUN.
    """
    order = closure(keys)
    results: dict[str, ProbeResult] = {}
    top_k = components.config.arm.top_k

    def llm() -> str:
        q = components.dataset.questions[0]
        llm = components.answer_llm
        prompt = render_answer_prompt(q, [])
        # Bypass the response cache: a cached reply proves nothing about reachability.
        if isinstance(llm, OpenAICompatLLM):
            reply = llm.complete(prompt, use_cache=False, purpose="probe").text
        else:
            reply = llm.complete(prompt).text
        assert reply.strip(), "LLM returned an empty reply"
        return f"{components.answer_llm.model} replied"

    def memory_service() -> str:
        with tempfile.TemporaryDirectory(prefix="agentmemeval-probe-") as scratch:
            arm = components.new_probe_arm(Path(scratch))
            arm.add(_canary_dataset(components))  # a write, not just a connection check
            found = arm.search("What is the canary word for this run?", top_k=3)
            assert found, f"{arm.name} accepted a write but the search found nothing"
            arm.close()
            return f"{arm.name} wrote and read back a canary in a scratch namespace"

    def corpus() -> str | None:
        arm = ingested_arm
        if arm is None:
            if components.config.arm.name in LIVE_ARMS:
                return None  # needs the run's ingested store
            arm = components.new_arm()
            arm.add(components.dataset)
        if canary is not None:
            q = components.dataset.question_by_id[canary]
            assert _hits(arm, q, top_k), f"canary {canary} no longer retrieves its evidence"
            return f"{CANARY_PREFIX}{canary} still retrieves its evidence"
        classes = components.window_classes
        candidates = [q for q in components.dataset.questions
                      if q.evidence_status is EvidenceStatus.LOCATED
                      and classes[q.qid] is WindowClass.A][:CANARY_CANDIDATES]  # fmt: skip
        assert candidates, "no in-window question to probe the corpus with"
        for q in candidates:
            if _hits(arm, q, top_k):
                return f"{CANARY_PREFIX}{q.qid} retrieves its evidence"
        raise AssertionError(f"none of {len(candidates)} in-window questions retrieves evidence")

    probes = {"llm": llm, "memory_service": memory_service, "corpus": corpus}
    if not components.new_arm().stores_data:
        # The control arm stores nothing by design; failing these probes would skip the
        # control itself. They are not applicable, and say so.
        def not_applicable() -> str:
            return "not applicable: the control arm stores nothing by design"

        probes["memory_service"] = probes["corpus"] = not_applicable
    for key in order:
        if all(results[d].ok for d in DEPENDENCIES[key]):
            result = _guard(key, probes[key])
            if result.ok and result.detail == "None":
                result = ProbeResult(key, ProbeStatus.NOT_RUN, "needs the run's ingested store")
            results[key] = result
    return enforce_dependencies(order, results)
