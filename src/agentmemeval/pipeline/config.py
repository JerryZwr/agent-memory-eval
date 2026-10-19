"""Run configuration (`run.json`) and the components built from it.

A run directory holds exactly one configuration. Resuming a run re-reads it;
asking to resume with a different configuration is refused (P2).

Live components (a real answering LLM, an LLM judge, the mem0 arm) require a
`live` section with a usage ledger and hard caps (CLAUDE.md §2.2). All live
components of a process share one ledger per ledger path, so the caps bound
the whole session, not each component separately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentmemeval.analysis.windows import WindowClass, classify_all, scored_classes
from agentmemeval.arms import MemoryArm, build_arm
from agentmemeval.data.loaders import load_fixture, load_legacy_batch
from agentmemeval.data.models import Dataset, Question
from agentmemeval.judge.base import Judge
from agentmemeval.judge.scripted import ScriptedJudge
from agentmemeval.llm.base import LLM
from agentmemeval.llm.scripted import make_reader_llm
from agentmemeval.llm.usage import LiveConfigError, ResponseCache, UsageLedger
from agentmemeval.pipeline.prompts import ANSWER_PROMPT_VERSION
from agentmemeval.pipeline.records import RunMeta

# The synthetic corpus lives with the tests (CLAUDE.md §2.3); resolved from a source checkout.
DEFAULT_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "mini_corpus"
LIVE_ARMS = frozenset({"mem0"})


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSpec(_Spec):
    kind: Literal["fixture", "legacy_batch"]
    path: str
    batch: str = "004"
    evidence_topic: str = "01"
    first_days: int | None = Field(default=None, ge=1)


class ArmSpec(_Spec):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1)


class AnswerSpec(_Spec):
    kind: Literal["scripted_reader", "openai_compat"] = "scripted_reader"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = Field(default=512, ge=16)


class JudgeSpec(_Spec):
    kind: Literal["scripted", "llm"] = "scripted"
    version: Literal["v1", "v2"] = "v1"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"


class LiveSpec(_Spec):
    """Where paid calls are counted and capped. Shared by every run of a session."""

    ledger: str
    cache: str | None = None
    max_calls: int = Field(ge=1)
    max_cost_cny: float = Field(gt=0)
    extraction_model: str = "deepseek-flash"  # mem0's fact-extraction calls


class QuestionSpec(_Spec):
    """Which evidence-window classes are asked and scored (METHODOLOGY 1.3)."""

    include_b: bool = False  # boundary-spanning evidence: excluded unless opted in
    # A pilot subset: this many questions, in a seeded hash order (qids are sorted by type,
    # so "the first n" would be biased). None = every question of the scored classes.
    sample: int | None = Field(default=None, ge=1)
    seed: int = 20251018


class RunConfig(_Spec):
    run_id: str
    dataset: DatasetSpec
    arm: ArmSpec
    questions: QuestionSpec = QuestionSpec()
    answer: AnswerSpec = AnswerSpec()
    judge: JudgeSpec = JudgeSpec()
    live: LiveSpec | None = None

    @property
    def is_live(self) -> bool:
        return (
            self.answer.kind != "scripted_reader"
            or self.judge.kind != "scripted"
            or self.arm.name in LIVE_ARMS
        )


def default_eval_config(arm: str = "bm25", run_id: str = "eval") -> RunConfig:
    """The offline evaluation stack: synthetic corpus, scripted reader and judge."""
    return RunConfig(
        run_id=run_id,
        dataset=DatasetSpec(kind="fixture", path=DEFAULT_FIXTURE.as_posix()),
        arm=ArmSpec(name=arm),
    )


def load_dataset(spec: DatasetSpec) -> Dataset:
    path = Path(spec.path)
    if spec.kind == "fixture":
        dataset = load_fixture(path)
    else:
        dataset = load_legacy_batch(path, batch=spec.batch, evidence_topic=spec.evidence_topic)
    return dataset.first_days(spec.first_days) if spec.first_days else dataset


def arm_config_hash(spec: ArmSpec, arm: MemoryArm) -> str:
    """Hash of the *effective* arm configuration (defaults resolved) plus retrieval depth."""
    payload = {"name": spec.name, "params": arm.params(), "top_k": spec.top_k}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- live plumbing

_LEDGERS: dict[Path, UsageLedger] = {}
_CACHES: dict[Path, ResponseCache] = {}


@dataclass
class LiveContext:
    spec: LiveSpec
    ledger: UsageLedger
    cache: ResponseCache | None

    def llm(self, model: str, purpose: str, *, base_url: str, max_tokens: int,
            cached: bool = True, **kw: Any):  # fmt: skip
        from agentmemeval.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(model, ledger=self.ledger, cache=self.cache if cached else None,
                               purpose=purpose, base_url=base_url, max_tokens=max_tokens,
                               **kw)  # fmt: skip


def live_context(spec: LiveSpec) -> LiveContext:
    path = Path(spec.ledger).resolve()
    ledger = _LEDGERS.get(path)
    if ledger is None:
        ledger = _LEDGERS[path] = UsageLedger(
            path, max_calls=spec.max_calls, max_cost_cny=spec.max_cost_cny
        )
    elif (ledger.max_calls, ledger.max_cost_cny) != (spec.max_calls, spec.max_cost_cny):
        raise LiveConfigError(f"{path}: already open with different caps")
    cache = None
    if spec.cache:
        cpath = Path(spec.cache).resolve()
        cache = _CACHES.get(cpath) or _CACHES.setdefault(cpath, ResponseCache(cpath))
    return LiveContext(spec, ledger, cache)


def build_answer_llm(spec: AnswerSpec, live: LiveContext | None) -> LLM:
    if spec.kind == "scripted_reader":
        return make_reader_llm()
    assert live is not None
    return live.llm(spec.model, "answer", base_url=spec.base_url, max_tokens=spec.max_tokens)


def build_judge(spec: JudgeSpec, live: LiveContext | None) -> Judge:
    if spec.kind == "scripted":
        return ScriptedJudge()
    from agentmemeval.judge.llm_judge import LLMJudge

    assert live is not None
    llm = live.llm(spec.model, f"judge-{spec.version}", base_url=spec.base_url, max_tokens=256)
    return LLMJudge(llm, spec.version)


def _make_arm(config: RunConfig, live: LiveContext | None, *, probe: bool = False) -> MemoryArm:
    if config.arm.name == "mem0":
        assert live is not None
        params = {"namespace": config.run_id, **config.arm.params}
        extraction = live.llm(live.spec.extraction_model, "probe" if probe else "mem0-extract",
                              base_url="https://api.deepseek.com", max_tokens=2048,
                              cached=not probe)  # fmt: skip
        return build_arm("mem0", params, llm=extraction)
    return build_arm(config.arm.name, config.arm.params)


@dataclass
class Components:
    config: RunConfig
    dataset: Dataset
    answer_llm: LLM
    judge: Judge
    meta: RunMeta
    live: LiveContext | None = None

    def new_arm(self) -> MemoryArm:
        """A fresh arm instance. Each stage gets its own, so stages share only files."""
        return _make_arm(self.config, self.live)

    def new_probe_arm(self, scratch: Path) -> MemoryArm:
        """An arm in a scratch namespace and store, so probes never write to the measured one."""
        if self.config.arm.name in LIVE_ARMS:
            params = {**self.config.arm.params, "store_dir": scratch.as_posix(),
                      "namespace": f"{self.config.run_id}-probe"}  # fmt: skip
            spec = self.config.arm.model_copy(update={"params": params})
            return _make_arm(self.config.model_copy(update={"arm": spec}), self.live, probe=True)
        return self.new_arm()

    @cached_property
    def window_classes(self) -> dict[str, WindowClass]:
        return classify_all(self.dataset)

    def questions(self) -> tuple[Question, ...]:
        """The questions this run asks: those whose window class is scored (optionally sampled)."""
        spec = self.config.questions
        keep = scored_classes(spec.include_b)
        asked = [q for q in self.dataset.questions if self.window_classes[q.qid] in keep]
        if spec.sample is not None:

            def order(q: Question) -> str:
                return hashlib.sha256(f"{spec.seed}:{q.qid}".encode()).hexdigest()

            chosen = {q.qid for q in sorted(asked, key=order)[: spec.sample]}
            asked = [q for q in asked if q.qid in chosen]
        return tuple(asked)


def build_components(
    config: RunConfig,
    *,
    answer_llm: LLM | None = None,
    judge: Judge | None = None,
) -> Components:
    """Build everything a run needs. `answer_llm` / `judge` overrides are for tests."""
    if config.is_live and config.live is None:
        raise LiveConfigError(
            "this configuration makes paid calls but has no `live` section (ledger and caps)"
        )
    live = live_context(config.live) if config.live else None
    dataset = load_dataset(config.dataset)
    arm = _make_arm(config, live)
    llm = answer_llm or build_answer_llm(config.answer, live)
    jdg = judge or build_judge(config.judge, live)
    meta = RunMeta(
        run_id=config.run_id,
        corpus_id=dataset.corpus_id,
        arm=config.arm.name,
        arm_config_hash=arm_config_hash(config.arm, arm),
        answer_model=llm.model,
        answer_prompt_version=ANSWER_PROMPT_VERSION,
        judge_model=jdg.model,
        judge_prompt_version=jdg.prompt_version,
        scored_classes=",".join(sorted(scored_classes(config.questions.include_b))),
        question_set="",
    )
    comps = Components(
        config=config, dataset=dataset, answer_llm=llm, judge=jdg, meta=meta, live=live
    )
    qids = json.dumps(sorted(q.qid for q in comps.questions()))
    comps.meta = meta.model_copy(
        update={"question_set": hashlib.sha256(qids.encode("utf-8")).hexdigest()[:12]}
    )
    return comps
