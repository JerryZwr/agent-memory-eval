"""Batch agent evaluation (METHODOLOGY 2.4): run a policy over the task set, record metrics.

Per task: outcome, n_llm_calls, n_tool_calls_by_name, budget_hit,
max_same_tool_streak, route, latency_breakdown, ttft_ms, the verdict, and the
result of every acceptance check (the same functions the tests use), plus the
trace. Files carry metadata and a records hash like the memory-layer stage
files, and are refused on load if edited.

Timings come from a simulated clock unless `real_clock` is set; the metadata
records which, and the report says so next to every timing.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agentmemeval.agent import checks
from agentmemeval.agent.core import Agent, AgentResult, Budget
from agentmemeval.agent.policies import POLICIES, make_policy_llm
from agentmemeval.agent.tasks import TaskSpec, all_tasks, judge_answer
from agentmemeval.agent.tools import MemorySearchTool, WebSearchTool
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.scoring import Ratio
from agentmemeval.errors import StageError
from agentmemeval.judge.base import Judge
from agentmemeval.judge.scripted import ScriptedJudge
from agentmemeval.llm.base import LLM
from agentmemeval.pipeline.config import Components
from agentmemeval.pipeline.records import records_sha256
from agentmemeval.pipeline.store import write_json_atomic
from agentmemeval.provenance import git_commit, utc_now
from agentmemeval.tracing import Clock, FakeClock, RealClock, latency_breakdown

AGENT_SCHEMA = 1


@dataclass(frozen=True)
class SimulatedLatency:
    """Virtual-clock costs (seconds). Illustrative, not measured."""

    llm_s: float = 0.35
    ttft_s: float = 0.12
    memory_s: float = 0.06
    web_s: float = 0.25
    framework_s: float = 0.015


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentMeta(_Model):
    run_id: str
    corpus_id: str
    policy: str
    agent_model: str
    arm: str
    arm_config_hash: str
    judge_model: str
    judge_prompt_version: str
    budget: dict[str, Any]
    clock: str  # "simulated" or "real"
    simulated_latency: dict[str, float] | None
    created_at: str
    git_commit: str


class CheckResult(_Model):
    passed: bool
    reason: str


class AgentTaskRecord(_Model):
    task_id: str
    kind: str
    qtype: str | None
    window_class: str | None
    expected_route: str
    route: str | None
    outcome: Outcome
    undelivered_reason: str | None
    budget_hit: str | None
    final_answer: str | None
    verdict: str | None
    n_llm_calls: int
    n_tool_calls_by_name: dict[str, int]
    max_same_tool_streak: int
    latency_breakdown: dict[str, Any]
    ttft_ms: float | None
    fallback: str
    checks: dict[str, CheckResult]
    warnings: list[str]
    trace: dict[str, Any]


class AgentRunFile(_Model):
    schema_version: int
    meta: AgentMeta
    records_sha256: str
    records: list[dict[str, Any]]

    def parsed(self) -> list[AgentTaskRecord]:
        return [AgentTaskRecord.model_validate(r) for r in self.records]


def _record(spec: TaskSpec, r: AgentResult, budget: Budget, verdict: str | None) -> AgentTaskRecord:
    b = latency_breakdown(r.trace)
    results = {
        "delivered": checks.check_delivered(r),
        "converged": checks.check_converged(r, budget),
        "route": checks.check_route(r, spec.expected_route),
        "fallback": checks.check_fallback(r),
        "trajectory": checks.check_trajectory(r, spec.kind),
        "trace": checks.check_trace(r),
    }
    return AgentTaskRecord(
        task_id=spec.task_id,
        kind=spec.kind,
        qtype=spec.qtype,
        window_class=spec.window_class,
        expected_route=spec.expected_route,
        route=r.route,
        outcome=r.outcome,
        undelivered_reason=r.undelivered_reason.value if r.undelivered_reason else None,
        budget_hit=r.budget_hit,
        final_answer=r.final_answer,
        verdict=verdict,
        n_llm_calls=r.n_llm_calls,
        n_tool_calls_by_name=r.n_tool_calls_by_name,
        max_same_tool_streak=r.max_same_tool_streak,
        latency_breakdown=b.to_dict(),
        ttft_ms=round(mean(b.ttft_ms), 3) if b.ttft_ms else None,
        fallback=checks.fallback_status(r),
        checks={k: CheckResult(passed=c.passed, reason=c.reason) for k, c in results.items()},
        warnings=list(r.warnings),
        trace=r.trace.to_dict(),
    )


def run_agent_eval(
    policy: str,
    components: Components,
    *,
    tasks: list[TaskSpec] | None = None,
    budget: Budget | None = None,
    latency: SimulatedLatency | None = None,
    real_clock: bool = False,
    judge: Judge | None = None,
    run_id: str = "agent-eval",
    make_llm: Callable[[Clock], LLM] | None = None,
) -> AgentRunFile:
    """`make_llm` replaces the scripted policy with a real model (live; implies real_clock).

    `policy` is then just the label recorded in the metadata.
    """
    budget = budget or Budget()
    latency = latency or SimulatedLatency()
    judge = judge or ScriptedJudge()
    tasks = tasks if tasks is not None else all_tasks(components)
    real_clock = real_clock or make_llm is not None
    arm = components.new_arm()
    arm.add(components.dataset)
    records = []
    model = None
    for spec in tasks:
        clock = RealClock() if real_clock else FakeClock()
        sim = not real_clock
        if make_llm is not None:
            llm = make_llm(clock)
        else:
            llm = make_policy_llm(policy, clock=clock if sim else None,
                                  latency_s=latency.llm_s if sim else 0.0,
                                  ttft_s=latency.ttft_s if sim else 0.0)  # fmt: skip
        model = llm.model
        tools = {
            "memory_search": MemorySearchTool(arm, top_k=components.config.arm.top_k, clock=clock,
                                              latency_s=latency.memory_s if sim else 0.0),
            "web_search": WebSearchTool(clock=clock, latency_s=latency.web_s if sim else 0.0),
        }  # fmt: skip
        agent = Agent(llm, tools, budget=budget, clock=clock,
                      framework_cost_s=latency.framework_s if sim else 0.0)  # fmt: skip
        result = agent.run(spec.task)
        verdict = judge_answer(spec, result.outcome, result.final_answer, judge)
        records.append(_record(spec, result, budget, verdict))
    meta = components.meta
    dumped = [r.model_dump(mode="json") for r in records]
    return AgentRunFile(
        schema_version=AGENT_SCHEMA,
        meta=AgentMeta(
            run_id=run_id,
            corpus_id=meta.corpus_id,
            policy=policy,
            agent_model=model or f"scripted-agent/{policy}",
            arm=meta.arm,
            arm_config_hash=meta.arm_config_hash,
            judge_model=judge.model,
            judge_prompt_version=judge.prompt_version,
            budget=asdict(budget),
            clock="real" if real_clock else "simulated",
            simulated_latency=None if real_clock else asdict(latency),
            created_at=utc_now(),
            git_commit=git_commit(),
        ),
        records_sha256=records_sha256(dumped),
        records=dumped,
    )


def write_agent_run(path: Path, run: AgentRunFile) -> Path:
    write_json_atomic(path, run.model_dump(mode="json"))
    return path


def load_agent_run(path: Path) -> AgentRunFile:
    try:
        run = AgentRunFile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StageError(f"{path} not found") from None
    except ValidationError as exc:
        raise StageError(f"{path} is not an agent run file: {exc}") from None
    if run.schema_version != AGENT_SCHEMA:
        raise StageError(f"{path}: agent schema {run.schema_version} is not supported")
    if records_sha256(run.records) != run.records_sha256:
        raise StageError(f"{path}: records do not match records_sha256 (edited by hand?)")
    return run


def find_agent_runs(directory: Path) -> list[AgentRunFile]:
    """Agent run files in `directory`, the `good` reference first."""
    if not directory.is_dir():
        return []
    runs = [load_agent_run(p) for p in sorted(directory.glob("*.json"))]
    order = list(POLICIES)
    return sorted(runs, key=lambda r: order.index(r.meta.policy) if r.meta.policy in order else 99)


# --------------------------------------------------------------------------- summary


@dataclass(frozen=True)
class AgentSummary:
    policy: str
    n: int
    delivery: Ratio
    budget_hit: Ratio
    accuracy: Ratio  # correct / delivered, tasks with a gold answer
    end_to_end: Ratio  # correct / (delivered + undelivered), tasks with a gold answer
    env_failure: int
    agent_error: int
    tool_calls: dict[str, int]
    tool_calls_per_task: dict[str, float]
    confusion: dict[str, dict[str, int]]
    precision_recall: dict[str, tuple[float, float]]
    latency_ms: dict[str, float]  # mean per task
    latency_shares: dict[str, float]  # of the summed total
    mean_ttft_ms: float | None
    check_failures: dict[str, int]


def summarize(run: AgentRunFile) -> AgentSummary:
    recs = run.parsed()
    n = len(recs)
    delivered = sum(r.outcome is Outcome.DELIVERED for r in recs)
    gold = [r for r in recs if r.verdict is not None or (r.kind != "chitchat")]
    scored_delivered = [r for r in gold if r.outcome is Outcome.DELIVERED and r.verdict is not None]
    scored_undelivered = [r for r in gold if r.outcome is Outcome.UNDELIVERED]
    correct = sum(r.verdict == "CORRECT" for r in scored_delivered)
    tools: Counter[str] = Counter()
    for r in recs:
        tools.update(r.n_tool_calls_by_name)
    confusion = checks.routing_confusion([(r.expected_route, r.route) for r in recs])
    keys = ("total_ms", "llm_ms", "tool_ms", "framework_ms")
    sums = {k: sum(r.latency_breakdown[k] for r in recs) for k in keys}
    ttfts = [r.ttft_ms for r in recs if r.ttft_ms is not None]
    failures: Counter[str] = Counter()
    for r in recs:
        failures.update(k for k, c in r.checks.items() if not c.passed)
    total = sums["total_ms"]
    return AgentSummary(
        policy=run.meta.policy,
        n=n,
        delivery=Ratio(delivered, n),
        budget_hit=Ratio(sum(r.budget_hit is not None for r in recs), n),
        accuracy=Ratio(correct, len(scored_delivered)),
        end_to_end=Ratio(correct, len(scored_delivered) + len(scored_undelivered)),
        env_failure=sum(r.outcome is Outcome.ENV_FAILURE for r in recs),
        agent_error=sum(r.outcome is Outcome.AGENT_ERROR for r in recs),
        tool_calls=dict(sorted(tools.items())),
        tool_calls_per_task={k: v / n for k, v in sorted(tools.items())},
        confusion=confusion,
        precision_recall=checks.precision_recall(confusion),
        latency_ms={k: round(v / n, 1) for k, v in sums.items()} if n else {},
        latency_shares={
            k.removesuffix("_ms"): (sums[k] / total if total else 0.0)
            for k in ("llm_ms", "tool_ms", "framework_ms")
        },
        mean_ttft_ms=round(mean(ttfts), 1) if ttfts else None,
        check_failures={k: failures.get(k, 0) for k in
                        ("delivered", "converged", "route", "fallback", "trajectory", "trace")},
    )  # fmt: skip


def run_all(components: Components, out_dir: Path, policies: list[str] | None = None) -> list[Path]:
    paths = []
    for policy in policies or list(POLICIES):
        run = run_agent_eval(policy, components, run_id=f"{out_dir.name}-{policy}")
        paths.append(write_agent_run(out_dir / f"{policy}.json", run))
    return paths


def summary_lines(summaries: list[AgentSummary]) -> list[str]:
    header = f"{'policy':<18} {'delivered':<26} {'budget hit':<24} {'accuracy':<24} checks failed"
    lines = [header, "-" * len(header)]
    for s in summaries:
        failed = ", ".join(f"{k}={v}" for k, v in s.check_failures.items() if v) or "none"
        lines.append(f"{s.policy:<18} {s.delivery!s:<26} {s.budget_hit!s:<24} "
                     f"{s.accuracy!s:<24} {failed}")  # fmt: skip
    return lines


def dump_summary_json(summary: AgentSummary) -> str:
    return json.dumps(asdict(summary), default=str, ensure_ascii=False, indent=2)
