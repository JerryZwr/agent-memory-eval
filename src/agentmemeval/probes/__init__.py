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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.data.models import EvidenceStatus, Question
from agentmemeval.errors import InfraError
from agentmemeval.pipeline.config import Components
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


def _probe_question(components: Components) -> Question:
    """A question whose evidence is inside the ingested window, used as a canary."""
    classes = components.window_classes
    for q in components.dataset.questions:
        if q.evidence_status is EvidenceStatus.LOCATED and classes[q.qid] is WindowClass.A:
            return q
    raise LookupError("no class-A question to probe the corpus with")


def _guard(key: str, fn: Callable[[], str]) -> ProbeResult:
    try:
        return ProbeResult(key, ProbeStatus.PASS, fn())
    except InfraError as exc:
        return ProbeResult(key, ProbeStatus.FAIL, f"{type(exc).__name__}: {exc}")
    except AssertionError as exc:
        return ProbeResult(key, ProbeStatus.FAIL, str(exc))


def run_probes(components: Components, keys: Iterable[str]) -> dict[str, ProbeResult]:
    """The default probes, run against the components a measurement would use."""
    order = closure(keys)
    results: dict[str, ProbeResult] = {}

    def llm() -> str:
        q = components.dataset.questions[0]
        reply = components.answer_llm.complete(render_answer_prompt(q, [])).text
        assert reply.strip(), "LLM returned an empty reply"
        return f"{components.answer_llm.model} replied"

    def memory_service() -> str:
        arm = components.new_arm()
        summary = arm.add(components.dataset)  # a write, not just a connection check
        arm.search("probe", top_k=1)
        return f"{arm.name} accepted {len(components.dataset.messages)} messages: {summary}"

    def corpus() -> str:
        q = _probe_question(components)
        arm = components.new_arm()
        arm.add(components.dataset)
        items = arm.search(q.question, top_k=components.config.arm.top_k)
        record = SearchRecord(
            qid=q.qid,
            qtype=q.qtype,
            evidence_status=q.evidence_status,
            window_class=WindowClass.A,
            query=q.question,
            items=items,
        )
        # Same success criterion as the analysis: evidence retrieved, not "non-empty".
        assert retrieval_hit(record, q, HitCriterion.MESSAGE), (
            f"known question {q.qid} no longer retrieves its evidence"
        )
        return f"{q.qid} retrieves its evidence"

    probes = {"llm": llm, "memory_service": memory_service, "corpus": corpus}
    for key in order:
        if all(results[d].ok for d in DEPENDENCIES[key]):
            results[key] = _guard(key, probes[key])
    return enforce_dependencies(order, results)
