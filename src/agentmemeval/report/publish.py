"""Export a live session as a public bundle that can be committed and served by Pages.

Live results cannot be regenerated in CI (no key, no raw data), so publication
goes through committed files only (docs/decisions.md D-005):

    results/<session>/
      report.html    the report in public mode: no question, gold, answer or
                     retrieved text, no local paths (D-002 / D-010)
      summary.json   per arm: metadata, per-question outcome / verdict / hit,
                     probe status, records hashes of every stage file; plus the
                     session usage ledger totals. No dataset text.

Every number in the public report can be traced to summary.json, and every
summary.json entry to a local stage file by its records_sha256.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.llm.usage import UsageLedger
from agentmemeval.pipeline.records import STAGES
from agentmemeval.pipeline.runner import load_run, read_probes
from agentmemeval.pipeline.store import write_json_atomic
from agentmemeval.provenance import git_commit, utc_now
from agentmemeval.report.build import USAGE_FILE, V2_DIR, find_run_dirs, write_report


def _run_summary(run_dir: Path) -> dict[str, Any]:
    run = load_run(run_dir)
    questions = run.components.dataset.question_by_id
    search = {s.qid: s for s in run.search_records}
    rows = []
    for j in run.judge_records:
        q = questions[j.qid]
        rows.append(
            {
                "qid": j.qid,
                "qtype": j.qtype,
                "window_class": j.window_class.value,
                "outcome": j.outcome.value,
                "verdict": j.verdict,
                "method": j.method,
                "ambiguous_extraction": j.ambiguous_extraction,
                "hit_message": retrieval_hit(search[j.qid], q, HitCriterion.MESSAGE),
                "hit_date": retrieval_hit(search[j.qid], q, HitCriterion.DATE),
            }
        )
    probes = read_probes(run_dir) or {}
    probes.pop("usage", None)  # session-level, reported once below
    return {
        "meta": run.meta.model_dump(mode="json"),
        "stage_records_sha256": {s: run.files[s].records_sha256 for s in STAGES},
        "probes": probes,
        "questions": rows,
    }


def publish(session: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    report = write_report(session, out / "report.html", public=True)
    arms = {d.name: _run_summary(d) for d in find_run_dirs(session)}
    v2_root = session / V2_DIR
    v2 = {d.name: _run_summary(d) for d in find_run_dirs(v2_root)} if v2_root.is_dir() else {}
    usage = None
    if (session / USAGE_FILE).exists():
        usage = UsageLedger(session / USAGE_FILE, max_calls=10**9, max_cost_cny=10.0**9).summary()
        usage.pop("ledger", None)
    summary = {
        "session": session.name,
        "exported_at": utc_now(),
        "git_commit": git_commit(),
        "arms": arms,
        "v2": v2,
        "usage": usage,
        "note": "No dataset text is included; see docs/decisions.md D-002 and D-010.",
    }
    path = out / "summary.json"
    write_json_atomic(path, summary)
    return [report, path]
