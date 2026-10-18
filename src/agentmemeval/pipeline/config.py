"""Run configuration (`run.json`) and the components built from it.

A run directory holds exactly one configuration. Resuming a run re-reads it;
asking to resume with a different configuration is refused (P2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentmemeval.arms import MemoryArm, build_arm
from agentmemeval.data.loaders import load_fixture, load_legacy_batch
from agentmemeval.data.models import Dataset, Question
from agentmemeval.judge.base import Judge
from agentmemeval.judge.scripted import ScriptedJudge
from agentmemeval.llm.base import LLM
from agentmemeval.llm.scripted import make_reader_llm
from agentmemeval.pipeline.prompts import ANSWER_PROMPT_VERSION
from agentmemeval.pipeline.records import RunMeta


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
    kind: Literal["scripted_reader"] = "scripted_reader"


class JudgeSpec(_Spec):
    kind: Literal["scripted"] = "scripted"


class RunConfig(_Spec):
    run_id: str
    dataset: DatasetSpec
    arm: ArmSpec
    answer: AnswerSpec = AnswerSpec()
    judge: JudgeSpec = JudgeSpec()


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


def build_answer_llm(spec: AnswerSpec) -> LLM:
    if spec.kind == "scripted_reader":
        return make_reader_llm()
    raise ValueError(f"unknown answer kind {spec.kind!r}")  # pragma: no cover


def build_judge(spec: JudgeSpec) -> Judge:
    if spec.kind == "scripted":
        return ScriptedJudge()
    raise ValueError(f"unknown judge kind {spec.kind!r}")  # pragma: no cover


@dataclass
class Components:
    config: RunConfig
    dataset: Dataset
    answer_llm: LLM
    judge: Judge
    meta: RunMeta

    def new_arm(self) -> MemoryArm:
        """A fresh arm instance. Each stage gets its own, so stages share only files."""
        return build_arm(self.config.arm.name, self.config.arm.params)

    def questions(self) -> tuple[Question, ...]:
        # M1 asks every question. M2 replaces this with evidence-window filtering.
        return self.dataset.questions


def build_components(
    config: RunConfig,
    *,
    answer_llm: LLM | None = None,
    judge: Judge | None = None,
) -> Components:
    """Build everything a run needs. `answer_llm` / `judge` overrides are for tests."""
    dataset = load_dataset(config.dataset)
    arm = build_arm(config.arm.name, config.arm.params)
    llm = answer_llm or build_answer_llm(config.answer)
    jdg = judge or build_judge(config.judge)
    meta = RunMeta(
        run_id=config.run_id,
        corpus_id=dataset.corpus_id,
        arm=config.arm.name,
        arm_config_hash=arm_config_hash(config.arm, arm),
        answer_model=llm.model,
        answer_prompt_version=ANSWER_PROMPT_VERSION,
        judge_model=jdg.model,
        judge_prompt_version=jdg.prompt_version,
    )
    return Components(config=config, dataset=dataset, answer_llm=llm, judge=jdg, meta=meta)
