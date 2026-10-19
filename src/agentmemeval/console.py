"""Plain-text result tables for the terminal (the HTML report is M2)."""

from __future__ import annotations

from collections.abc import Sequence

from agentmemeval.analysis.scoring import Tally, tally_by
from agentmemeval.errors import IncomparableError
from agentmemeval.pipeline.runner import RunResult

SCOPE_LABELS = {
    "located": "evidence",
    "no_evidence": "no-evidence",
    "unrecoverable": "unrecoverable (E)",
}
SMALL_SAMPLE = 30


def check_comparable(results: Sequence[RunResult]) -> None:
    """Arms may only sit in one table if everything except the arm is identical."""
    fields = ("corpus_id", "question_set", "answer_model", "answer_prompt_version", "judge_model")
    fields += ("judge_prompt_version",)
    first = results[0].components.meta
    for r in results[1:]:
        meta = r.components.meta
        diff = [f for f in fields if getattr(meta, f) != getattr(first, f)]
        if diff:
            raise IncomparableError(f"runs differ in {diff}; refusing to put them in one table")


def _row(cells: Sequence[object], widths: Sequence[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()


def format_results(results: Sequence[RunResult]) -> str:
    check_comparable(results)
    meta = results[0].components.meta
    header = ["arm", "scope", "n", "delivered", "undelivered", "env_failure", "ambiguous"]
    header += ["accuracy", "end-to-end"]
    rows: list[list[object]] = []
    smallest = None
    for r in results:
        groups: dict[str, Tally] = tally_by(r.judge_records, lambda j: j.evidence_status.value)
        for scope, t in groups.items():
            smallest = t.n if smallest is None else min(smallest, t.n)
            rows.append(
                [
                    r.components.meta.arm,
                    SCOPE_LABELS.get(scope, scope),
                    t.n,
                    t.delivered,
                    t.undelivered,
                    t.env_failure,
                    t.ambiguous_extraction,
                    t.accuracy,
                    t.end_to_end,
                ]
            )
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]
    lines = [
        f"corpus_id: {meta.corpus_id}",
        f"answer:    {meta.answer_model} (prompt {meta.answer_prompt_version})",
        f"judge:     {meta.judge_model} ({meta.judge_prompt_version})",
        "",
        _row(header, widths),
        _row(["-" * w for w in widths], widths),
        *(_row(row, widths) for row in rows),
        "",
        "accuracy = correct / delivered; end-to-end = correct / (delivered + undelivered);",
        "env_failure is in neither denominator (METHODOLOGY 1.4).",
    ]
    if smallest is not None and smallest < SMALL_SAMPLE:
        lines.append(
            f"Some rows have n < {SMALL_SAMPLE}: qualitative only (METHODOLOGY 1.7). "
            "The HTML report adds paired tests: agentmemeval report --run <dir>."
        )
    return "\n".join(lines)
