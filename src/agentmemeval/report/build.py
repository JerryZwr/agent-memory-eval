"""Build the static HTML evaluation report from finished run directories.

Every number in the report is computed here from verified stage files
(`load_run` refuses incomplete or stale runs); the template only formats.
Rates always go through `Ratio`, so each one carries its denominator and
Wilson 95% interval, and cells with n < 30 are flagged as qualitative only.
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup

from agentmemeval import __version__
from agentmemeval.agent.eval import AgentRunFile, find_agent_runs, summarize
from agentmemeval.analysis.breakdown import (
    CRITERION_LABELS,
    HitCriterion,
    TwoByTwo,
    by_qtype,
    by_scope,
    retrieval_hit,
    two_by_two,
)
from agentmemeval.analysis.compare import PairedComparison, compare, format_p
from agentmemeval.analysis.scoring import Tally, tally, tally_by
from agentmemeval.analysis.stats import SMALL_SAMPLE
from agentmemeval.analysis.windows import DESCRIPTIONS, WindowClass, class_counts, scored_classes
from agentmemeval.console import check_comparable
from agentmemeval.data.models import QTYPE_LABELS, Question
from agentmemeval.errors import IncomparableError, StageError
from agentmemeval.pipeline.records import STAGES, JudgeRecord, SearchRecord
from agentmemeval.pipeline.runner import RUN_CONFIG, LoadedRun, load_run
from agentmemeval.provenance import git_commit, utc_now

CONTROL_ARM = "null"
REPORT_NAME = "report.html"
CLASS_LABELS = {
    WindowClass.A: "A 类：证据在窗口内",
    WindowClass.B: "B 类：证据跨窗口边界",
    WindowClass.D: "D 类：无证据（应拒答）",
}


def find_run_dirs(path: Path) -> list[Path]:
    """`path` is one run directory, or a directory whose children are run directories."""
    if (path / RUN_CONFIG).is_file():
        return [path]
    runs = sorted(p for p in path.iterdir() if (p / RUN_CONFIG).is_file()) if path.is_dir() else []
    if not runs:
        raise StageError(f"{path}: no run directories (no {RUN_CONFIG}) found")
    return runs


def _arm_order(run: LoadedRun) -> tuple[int, str]:
    return (0 if run.meta.arm == CONTROL_ARM else 1, run.meta.arm)


@dataclass
class ArmView:
    name: str
    run: LoadedRun
    by_class: list[tuple[str, Tally]]
    by_qtype: dict[str, Tally]
    by_scope: dict[str, Tally]
    two_by_two: list[TwoByTwo]


@dataclass
class QuestionView:
    question: Question
    window_class: WindowClass
    per_arm: list[dict[str, Any]] = field(default_factory=list)


def _arm_view(run: LoadedRun, classes: list[WindowClass]) -> ArmView:
    judged = run.judge_records
    questions = run.components.dataset.question_by_id
    grouped = tally_by(judged, lambda r: r.window_class.value)
    return ArmView(
        name=run.meta.arm,
        run=run,
        by_class=[
            (CLASS_LABELS.get(c, c.value), grouped[c.value]) for c in classes if c.value in grouped
        ],
        by_qtype=by_qtype(judged),
        by_scope=by_scope(questions, judged),
        two_by_two=[two_by_two(questions, run.search_records, judged, c) for c in HitCriterion],
    )


def _question_views(runs: list[LoadedRun]) -> list[QuestionView]:
    dataset = runs[0].components.dataset
    judged = [{r.qid: r for r in run.judge_records} for run in runs]
    searched = [{s.qid: s for s in run.search_records} for run in runs]
    qids = sorted(set().union(*judged))
    views = []
    for qid in qids:
        q = dataset.question_by_id[qid]
        first: JudgeRecord = next(j[qid] for j in judged if qid in j)
        view = QuestionView(question=q, window_class=first.window_class)
        evidence = set(q.evidence_ids)
        for run, j, s in zip(runs, judged, searched, strict=True):
            rec: JudgeRecord | None = j.get(qid)
            srec: SearchRecord | None = s.get(qid)
            view.per_arm.append(
                {
                    "arm": run.meta.arm,
                    "judge": rec,
                    "items": [
                        (it, it.message_id in evidence) for it in (srec.items if srec else [])
                    ],
                    "hit": retrieval_hit(srec, q, HitCriterion.MESSAGE) if srec else None,
                    "search_error": srec.error if srec else None,
                }
            )
        views.append(view)
    return views


def _bar_chart(arms: list[ArmView], wc: WindowClass) -> Markup:
    """Accuracy with Wilson whiskers per arm, as inline SVG (no external libraries)."""
    rows = [
        (a.name, t.accuracy) for a in arms for label, t in a.by_class if label == CLASS_LABELS[wc]
    ]
    if not rows:
        return Markup("")
    left, width, bar_h, gap = 70, 420, 22, 14
    height = len(rows) * (bar_h + gap) + 30
    parts = [
        f'<svg class="chart" viewBox="0 0 {left + width + 60} {height}" role="img" '
        f'aria-label="answer accuracy with 95% intervals">'
    ]
    for tick in (0, 25, 50, 75, 100):
        x = left + width * tick / 100
        parts.append(f'<line class="grid" x1="{x}" y1="0" x2="{x}" y2="{height - 22}"/>')
        parts.append(
            f'<text class="tick" x="{x}" y="{height - 6}" text-anchor="middle">{tick}%</text>'
        )
    for i, (name, ratio) in enumerate(rows):
        y = i * (bar_h + gap) + 6
        text_y = y + bar_h * 0.7
        parts.append(
            f'<text class="label" x="{left - 8}" y="{text_y}" text-anchor="end">{name}</text>'
        )
        v, ci = ratio.value, ratio.interval
        if v is None or ci is None:
            continue
        parts.append(
            f'<rect class="bar {"control" if name == CONTROL_ARM else ""}" x="{left}" y="{y}" '
            f'width="{width * v:.1f}" height="{bar_h}" rx="3"/>'
        )
        lo, hi, mid = left + width * ci.low, left + width * ci.high, y + bar_h / 2
        parts.append(f'<line class="whisker" x1="{lo:.1f}" y1="{mid}" x2="{hi:.1f}" y2="{mid}"/>')
        for x in (lo, hi):
            parts.append(
                f'<line class="whisker" x1="{x:.1f}" y1="{mid - 6}" x2="{x:.1f}" y2="{mid + 6}"/>'
            )
        parts.append(f'<text class="value" x="{hi + 6:.1f}" y="{text_y}">{v:.0%}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def _limitations(runs: list[LoadedRun], arms: list[ArmView], include_b: bool) -> list[str]:
    meta = runs[0].meta
    spec = runs[0].components.config.dataset
    out = []
    if spec.kind == "fixture":
        out.append(
            "语料是自己编写的合成小语料，只用于演示流水线和分析方法，结论不能外推到真实数据。"
        )
    if meta.answer_model.startswith("scripted/"):
        out.append(
            "答题方是确定性的抽取式阅读器（ScriptedLLM），不代表任何真实 LLM 的能力；"
            "它做不了日期运算和缺少共同词的多跳推理，这是刻意保留的局限。"
        )
    if meta.judge_model.startswith("scripted"):
        out.append(
            "开放题评分员是规则匹配的离线脚本，没有经过人工校准（METHODOLOGY 1.6），"
            "判分只反映规则本身。"
        )
    if any(t.n < SMALL_SAMPLE for a in arms for _, t in a.by_class):
        out.append(
            f"多个格子样本量小于 {SMALL_SAMPLE}，区间很宽，仅供定性参考；"
            "按维度拆分后的小格子不做横向比较。"
        )
    out.append("D 类（无证据题）的分数不能跨语料规模比较：库越小，“正确地说不知道”越容易。")
    if not include_b:
        out.append("B 类（证据跨窗口边界）默认不提问、不计分；需要时在运行配置里显式打开。")
    out.append(
        "“检索命中”按证据判定，给出两种口径：命中证据消息（严格）和命中证据所在的日期（宽松）；"
        "不以“返回非空”判定命中。"
    )
    return out


AGENT_DIR = "agent"
CHECK_LABELS = {
    "delivered": "交付",
    "converged": "收敛",
    "route": "路由",
    "fallback": "回退",
    "trajectory": "轨迹",
    "trace": "trace 完整性",
}


def _agent_context(agent_runs: list[AgentRunFile], corpus_id: str) -> dict[str, Any] | None:
    if not agent_runs:
        return None
    for run in agent_runs:
        if run.meta.corpus_id != corpus_id:
            raise IncomparableError(
                f"agent run {run.meta.policy} used corpus {run.meta.corpus_id}, "
                f"the memory runs used {corpus_id}"
            )
    kinds: dict[str, int] = {}
    for rec in agent_runs[0].parsed():
        kinds[rec.kind] = kinds.get(rec.kind, 0) + 1
    summaries = [summarize(r) for r in agent_runs]
    first = agent_runs[0].meta
    return {
        "summaries": summaries,
        "tools": sorted({t for s in summaries for t in s.tool_calls}),
        "routes": ["memory", "web", "chitchat"],
        "check_labels": CHECK_LABELS,
        "budget": first.budget,
        "clock": first.clock,
        "latency": first.simulated_latency,
        "task_kinds": kinds,
        "agent_model_prefix": first.agent_model.split("/")[0],
    }


def build_context(
    runs: list[LoadedRun], agent_runs: list[AgentRunFile] | None = None
) -> dict[str, Any]:
    if not runs:
        raise StageError("no runs to report on")
    runs = sorted(runs, key=_arm_order)
    check_comparable([r for r in runs])  # same corpus, answerer and judge, or refuse
    arms_names = [r.meta.arm for r in runs]
    if len(set(arms_names)) != len(arms_names):
        raise StageError(f"two runs have the same arm: {arms_names}")

    cfg = runs[0].components.config
    include_b = cfg.questions.include_b
    scored = scored_classes(include_b)
    shown = [c for c in (WindowClass.A, WindowClass.B, WindowClass.D) if c in scored]
    arms = [_arm_view(r, shown) for r in runs]
    dataset = runs[0].components.dataset
    counts = class_counts(dataset)

    comparisons: list[tuple[str, PairedComparison]] = []
    control = next((r for r in runs if r.meta.arm == CONTROL_ARM), None)
    if control is not None:
        for r in runs:
            if r is control:
                continue
            for wc in shown:
                comparisons.append(
                    (
                        CLASS_LABELS[wc],
                        compare(CONTROL_ARM, control.judge_records, r.meta.arm, r.judge_records,
                                classes={wc}),
                    )
                )  # fmt: skip

    qtypes = sorted({qt for a in arms for qt in a.by_qtype})
    scopes = [s for s in ("single-group", "cross-group", "no-evidence", "unknown")
              if any(s in a.by_scope for a in arms)]  # fmt: skip
    small = any(t.n < SMALL_SAMPLE for a in arms for _, t in a.by_class)
    return {
        "title": "agent-memory-eval 评测报告",
        "generated_at": utc_now(),
        "generator": f"agentmemeval {__version__} @ {git_commit()}",
        "meta": runs[0].meta,
        "dataset": dataset,
        "dataset_spec": cfg.dataset,
        "class_rows": [(c, counts[c], c in scored, DESCRIPTIONS[c]) for c in WindowClass],
        "arms": arms,
        "chart": _bar_chart(arms, WindowClass.A),
        "comparisons": comparisons,
        "qtypes": [(qt, QTYPE_LABELS.get(qt, qt)) for qt in qtypes],
        "scopes": scopes,
        "criteria": [(c, CRITERION_LABELS[c]) for c in HitCriterion],
        "questions": _question_views(runs),
        "limitations": _limitations(runs, arms, include_b),
        "small_sample": small,
        "small_threshold": SMALL_SAMPLE,
        "stages": STAGES,
        "overall": [(a.name, tally(a.run.judge_records)) for a in arms],
        "agent": _agent_context(agent_runs or [], runs[0].meta.corpus_id),
    }


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("agentmemeval.report", "templates"),
        autoescape=True,  # the template is *.html.j2, so extension-based autoescape won't fire
        undefined=StrictUndefined,  # a typo in the template must fail, not render blank
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "n/a" if v is None else f"{v:.0%}"
    env.filters["signed_pct"] = lambda v: "n/a" if v is None else f"{v:+.0%}"
    env.filters["p_value"] = format_p
    return env


def render(context: dict[str, Any]) -> str:
    return _environment().get_template("report.html.j2").render(**context)


def write_report(path: Path, out: Path | None = None, *, open_browser: bool = False) -> Path:
    run_dirs = find_run_dirs(path)
    runs = [load_run(d) for d in run_dirs]
    agent_runs = find_agent_runs(path / AGENT_DIR)
    target = out or (path / REPORT_NAME if len(run_dirs) > 1 else run_dirs[0] / REPORT_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(build_context(runs, agent_runs)), encoding="utf-8", newline="\n")
    if open_browser:
        webbrowser.open(target.resolve().as_uri())
    return target
