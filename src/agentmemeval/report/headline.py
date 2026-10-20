"""README headline: the results table and figure, generated from a committed summary.json.

The README's first screen shows numbers, and a hand-copied number is exactly
the "plausible wrong number" P1 warns about: it goes stale the moment a run is
re-published. So the table between the README markers and the SVG figure are
rendered here from `results/<session>/summary.json` (the public bundle written
by `agentmemeval publish`), and `agentmemeval readme --check` fails, in CI via
tests/unit/test_headline.py, when either one no longer matches.

Counting follows the report exactly: accuracy = correct / delivered, wrapped in
`Ratio` (Wilson 95%); paired comparisons count delivered-and-CORRECT as a pass
and drop a pair when either side is env_failure (analysis/compare.py).
The summary rows are not JudgeRecords (no dataset text is published), so a
test cross-checks these numbers against `tally` / `compare` on real runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from agentmemeval.analysis.compare import format_p
from agentmemeval.analysis.scoring import Ratio
from agentmemeval.analysis.stats import mcnemar_exact
from agentmemeval.errors import IncomparableError, StageError

START = "<!-- headline:start -->"
END = "<!-- headline:end -->"
CONTROL_ARM = "null"
ARM_ORDER = ("null", "bm25", "mem0", "everos")
CLASSES = (("A", "A 类：证据在窗口内"), ("D", "D 类：无证据题，应拒答"))
COMPARED_CLASS = "A"
# Everything but the judge prompt must match across all runs in one table.
SHARED_META = ("corpus_id", "question_set", "answer_model", "answer_prompt_version", "judge_model")

Rows = list[dict[str, Any]]


@dataclass(frozen=True)
class Paired:
    treatment: str
    control: str
    only_treatment: int
    only_control: int

    @property
    def p_value(self) -> float:
        return mcnemar_exact(self.only_treatment, self.only_control)

    def __str__(self) -> str:
        return f"{self.only_treatment} / {self.only_control}，{format_p(self.p_value)}"


@dataclass(frozen=True)
class Headline:
    corpus_id: str
    answer_model: str
    judge_model: str
    versions: tuple[str, ...]  # judge prompt versions, e.g. ("llm-v1", "llm-v2")
    arms: tuple[str, ...]
    accuracy: dict[tuple[str, str, str], Ratio]  # (version, arm, class) -> ratio
    class_sizes: dict[str, int]
    paired: dict[tuple[str, str, str], Paired]  # (version, treatment, control), class A


def accuracy(rows: Rows, window_class: str) -> Ratio:
    delivered = [r for r in rows if r["window_class"] == window_class]
    delivered = [r for r in delivered if r["outcome"] == "delivered"]
    if any(r["verdict"] is None for r in delivered):
        raise StageError("a delivered answer has no verdict; refusing to score")
    return Ratio(sum(r["verdict"] == "CORRECT" for r in delivered), len(delivered))


def paired(treatment: str, t_rows: Rows, control: str, c_rows: Rows, window_class: str) -> Paired:
    def passes(rows: Rows) -> dict[str, bool | None]:
        return {
            r["qid"]: None
            if r["outcome"] == "env_failure"
            else (r["outcome"] == "delivered" and r["verdict"] == "CORRECT")
            for r in rows
            if r["window_class"] == window_class
        }

    t, c = passes(t_rows), passes(c_rows)
    pairs = [(c[q], t[q]) for q in sorted(t.keys() & c.keys())]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    return Paired(
        treatment,
        control,
        only_treatment=sum(1 for x, y in pairs if y and not x),
        only_control=sum(1 for x, y in pairs if x and not y),
    )


def _ordered(arms: Sequence[str]) -> tuple[str, ...]:
    rank = {a: i for i, a in enumerate(ARM_ORDER)}
    return tuple(sorted(arms, key=lambda a: (rank.get(a, len(rank)), a)))


def _sessions(summary: dict[str, Any]) -> list[dict[str, dict[str, Any]]]:
    sessions = [summary["arms"]]
    if summary.get("v2"):
        sessions.append(summary["v2"])
    return sessions


def _check(summary: dict[str, Any]) -> None:
    """Refuse a table whose cells were not measured under the same conditions (P2)."""
    sessions = _sessions(summary)
    if set(sessions[-1]) != set(sessions[0]):
        raise IncomparableError("judge versions cover different arms; refusing to tabulate")
    first = sessions[0][next(iter(sessions[0]))]["meta"]
    for runs in sessions:
        versions = {r["meta"]["judge_prompt_version"] for r in runs.values()}
        if len(versions) != 1:
            raise IncomparableError(f"one table, several judge prompts: {sorted(versions)}")
        for arm, run in runs.items():
            diff = [f for f in SHARED_META if run["meta"][f] != first[f]]
            if diff:
                raise IncomparableError(f"{arm} differs in {diff}; refusing to tabulate")
    # Probes ran in the round that produced the answers; a rejudge copies those
    # answers (checked below) and has no probes of its own.
    for arm, run in sessions[0].items():
        status = run["probes"].get("round_status")
        if status != "measured":
            raise StageError(f"{arm}: round status is {status!r}, not measured")
    # The judge prompt must be the only variable between versions (METHODOLOGY 1.5).
    for arm in sessions[0]:
        answers = {s[arm]["stage_records_sha256"]["answer"] for s in sessions}
        configs = {s[arm]["meta"]["arm_config_hash"] for s in sessions}
        if len(answers) != 1 or len(configs) != 1:
            raise IncomparableError(f"{arm}: judge versions graded different answers")


def headline(summary: dict[str, Any]) -> Headline:
    _check(summary)
    sessions = _sessions(summary)
    arms = _ordered(list(sessions[0]))
    first = sessions[0][arms[0]]
    versions = tuple(s[arms[0]]["meta"]["judge_prompt_version"] for s in sessions)
    acc: dict[tuple[str, str, str], Ratio] = {}
    pairs: dict[tuple[str, str, str], Paired] = {}
    for version, runs in zip(versions, sessions, strict=True):
        for arm in arms:
            for wc, _ in CLASSES:
                acc[(version, arm, wc)] = accuracy(runs[arm]["questions"], wc)
        others = [a for a in arms if a != CONTROL_ARM]
        matchups = [(a, CONTROL_ARM) for a in others if CONTROL_ARM in arms]
        matchups += [(b, a) for i, a in enumerate(others) for b in others[i + 1 :]]
        for t, c in matchups:
            rows_t, rows_c = runs[t]["questions"], runs[c]["questions"]
            pairs[(version, t, c)] = paired(t, rows_t, c, rows_c, COMPARED_CLASS)
    sizes = {wc: sum(r["window_class"] == wc for r in first["questions"]) for wc, _ in CLASSES}
    return Headline(
        corpus_id=first["meta"]["corpus_id"],
        answer_model=first["meta"]["answer_model"],
        judge_model=first["meta"]["judge_model"],
        versions=versions,
        arms=arms,
        accuracy=acc,
        class_sizes=sizes,
        paired=pairs,
    )


def _short(version: str) -> str:
    return version.removeprefix("llm-")


def _arm_label(arm: str) -> str:
    return f"{arm}（对照组）" if arm == CONTROL_ARM else arm


def render_markdown(h: Headline, *, source: str, figure: str) -> str:
    cols = [(wc, label, v) for wc, label in CLASSES for v in h.versions]
    head = " | ".join(f"{wc} 类 · 评分员 {_short(v)}" for wc, _, v in cols)
    lines = [
        START,
        f"<!-- 由 `agentmemeval readme` 从 {source} 生成，不要手改 -->",
        "",
        f"![各 arm 的答案正确率与 95% 区间]({figure})",
        "",
        f"**答案正确率**（答对 / 已交付，95% Wilson 区间）。语料 `{h.corpus_id}`，"
        f"答题 `{h.answer_model}`，评分员 `{h.judge_model}`；"
        f"A 类 {h.class_sizes['A']} 题，D 类 {h.class_sizes['D']} 题。",
        "",
        f"| arm | {head} |",
        "|---|" + "---|" * len(cols),
    ]
    for arm in h.arms:
        cells = " | ".join(str(h.accuracy[(v, arm, wc)]) for wc, _, v in cols)
        lines.append(f"| {_arm_label(arm)} | {cells} |")
    lines += [
        "",
        f"**配对比较**（{COMPARED_CLASS} 类，McNemar 精确检验；"
        "单元格 = 只有前者答对的题数 / 只有后者答对的题数，p 值）",
        "",
        "| 比较 | " + " | ".join(f"评分员 {_short(v)}" for v in h.versions) + " |",
        "|---|" + "---|" * len(h.versions),
    ]
    matchups = list(dict.fromkeys((t, c) for _, t, c in h.paired))
    for t, c in matchups:
        cells = " | ".join(str(h.paired[(v, t, c)]) for v in h.versions)
        lines.append(f"| {t} vs {c} | {cells} |")
    lines.append(END)
    return "\n".join(lines)


# --- figure -----------------------------------------------------------------
# Dot + Wilson whisker per (arm, judge version), one panel per window class.
# Version identity is color AND marker shape (circle / square), and the legend
# names both, so it never rests on color alone. Colors are the first two slots
# of the dataviz reference palette (validated as an adjacent pair, both modes).

_STYLE = """
.bg { fill: #fcfcfb; }
.t { font: 12px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; fill: #52514e; }
.t.strong { fill: #0b0b0b; font-weight: 600; }
.grid { stroke: #e4e3df; stroke-width: 1; }
.axis { stroke: #b9b8b2; stroke-width: 1; }
.s0 { fill: #2a78d6; stroke: #2a78d6; }
.s1 { fill: #eb6834; stroke: #eb6834; }
.w { fill: none; stroke-width: 2; }
.ring { stroke: #fcfcfb; stroke-width: 2; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; }
  .t { fill: #c3c2b7; }
  .t.strong { fill: #f0efec; }
  .grid { stroke: #33332f; }
  .axis { stroke: #5c5b55; }
  .s0 { fill: #3987e5; stroke: #3987e5; }
  .s1 { fill: #d95926; stroke: #d95926; }
  .ring { stroke: #1a1a19; }
}
"""


def _marker(i: int, x: float, y: float) -> str:
    cls = f"s{i} ring"
    if i == 0:
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
    return f'<rect class="{cls}" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1"/>'


def render_svg(h: Headline) -> str:
    if len(h.versions) > 2:
        raise ValueError("the figure encodes at most two judge versions")
    label_w, panel_w, plot_w, gap = 92, 330, 270, 24
    top, row_h, offset = 64, 46, 8
    width = label_w + len(CLASSES) * panel_w + (len(CLASSES) - 1) * gap
    plot_h = len(h.arms) * row_h
    height = top + plot_h + 34
    desc = "; ".join(
        f"{wc} {arm} {_short(v)}: {h.accuracy[(v, arm, wc)]}"
        for wc, _ in CLASSES
        for arm in h.arms
        for v in h.versions
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">',
        '<title id="title">答案正确率与 95% Wilson 区间，按证据窗口类别和评分员口径</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        f"<style>{_STYLE}</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="6"/>',
    ]
    # legend, top left
    lx = label_w
    for i, v in enumerate(h.versions):
        out.append(_marker(i, lx + 6, 18))
        out.append(f'<text class="t" x="{lx + 16}" y="22">评分员 {_short(v)}</text>')
        lx += 104
    out.append(f'<text class="t" x="{lx + 8}" y="22">点 = 正确率，横线 = 95% Wilson 区间</text>')
    for p, (wc, title) in enumerate(CLASSES):
        x0 = label_w + p * (panel_w + gap)
        out.append(
            f'<text class="t strong" x="{x0}" y="{top - 14}">{escape(title)}'
            f"（n = {h.class_sizes[wc]}）</text>"
        )
        for tick in (0, 25, 50, 75, 100):
            x = x0 + plot_w * tick / 100
            cls = "axis" if tick == 0 else "grid"
            out.append(
                f'<line class="{cls}" x1="{x}" y1="{top - 4}" x2="{x}" y2="{top + plot_h}"/>'
            )
            out.append(
                f'<text class="t" x="{x}" y="{top + plot_h + 18}" text-anchor="middle">'
                f"{tick}%</text>"
            )
        for r, arm in enumerate(h.arms):
            yc = top + r * row_h + row_h / 2
            if p == 0:
                out.append(
                    f'<text class="t strong" x="{label_w - 14}" y="{yc + 4}" '
                    f'text-anchor="end">{escape(arm)}</text>'
                )
            for i, v in enumerate(h.versions):
                ratio = h.accuracy[(v, arm, wc)]
                val, ci = ratio.value, ratio.interval
                if val is None or ci is None:
                    continue
                y = yc + (i - (len(h.versions) - 1) / 2) * 2 * offset
                lo, hi = x0 + plot_w * ci.low, x0 + plot_w * ci.high
                out.append(
                    f'<line class="s{i} w" x1="{lo:.1f}" y1="{y:.1f}" x2="{hi:.1f}" y2="{y:.1f}"/>'
                )
                out.append(_marker(i, x0 + plot_w * val, y))
                out.append(f'<text class="t" x="{hi + 6:.1f}" y="{y + 4:.1f}">{val:.0%}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- README block -------------------------------------------------------------


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def splice(readme: str, block: str) -> str:
    """Replace the text between the markers; refuse when they are missing or repeated."""
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise StageError(f"README must contain exactly one {START} … {END} block")
    head, rest = readme.split(START)
    _, tail = rest.split(END)
    return head + block + tail


@dataclass(frozen=True)
class Outputs:
    readme: str
    figure: str


def build(summary_path: Path, readme_path: Path, figure_path: Path) -> Outputs:
    h = headline(load_summary(summary_path))
    root = readme_path.parent.resolve()

    def rel(p: Path) -> str:
        return p.resolve().relative_to(root).as_posix()

    block = render_markdown(h, source=f"`{rel(summary_path)}`", figure=rel(figure_path))
    return Outputs(
        readme=splice(readme_path.read_text(encoding="utf-8"), block), figure=render_svg(h)
    )


def stale(summary_path: Path, readme_path: Path, figure_path: Path) -> list[Path]:
    """Files whose committed content no longer matches the summary."""
    want = build(summary_path, readme_path, figure_path)
    have_figure = figure_path.read_text(encoding="utf-8") if figure_path.exists() else None
    out = []
    if readme_path.read_text(encoding="utf-8") != want.readme:
        out.append(readme_path)
    if have_figure != want.figure:
        out.append(figure_path)
    return out


def write(summary_path: Path, readme_path: Path, figure_path: Path) -> list[Path]:
    want = build(summary_path, readme_path, figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(want.readme, encoding="utf-8", newline="\n")
    figure_path.write_text(want.figure, encoding="utf-8", newline="\n")
    return [readme_path, figure_path]
