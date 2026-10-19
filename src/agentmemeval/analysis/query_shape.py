"""Query-shape sensitivity (METHODOLOGY 1.9): does retrieval depend on how the question is phrased?

For every question with located evidence in a finished run, the run's own
ingested store is searched three times:

1. `stem`         the question text only (what the pipeline used);
2. `stem_options` the question plus every option text (multiple choice; else = stem);
3. `short`        an *automatic* short query: the first content words of the stem.
                  METHODOLOGY asks for a manually rewritten query; that would need
                  the author to write one per question, so this is a stated
                  substitute (docs/decisions.md D-010), labelled as automatic.

Reported per shape: evidence-hit rate (same `retrieval_hit` as everywhere,
message criterion where it applies, else date) and empty-result rate. Search
only: nothing is written and no LLM is called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.analysis.scoring import Ratio
from agentmemeval.data.models import EvidenceStatus, Question
from agentmemeval.pipeline.records import IngestRecord, SearchRecord
from agentmemeval.pipeline.runner import LoadedRun
from agentmemeval.pipeline.store import parse_records, write_json_atomic
from agentmemeval.provenance import git_commit, utc_now
from agentmemeval.text import tokenize

SHAPES = ("stem", "stem_options", "short")
SHORT_QUERY_WORDS = 8
QUERY_SHAPE_FILE = "query_shape.json"


def short_query(question: str, n: int = SHORT_QUERY_WORDS) -> str:
    words: list[str] = []
    for tok in tokenize(question):
        if tok not in words:
            words.append(tok)
    return " ".join(words[:n])


def queries(q: Question) -> dict[str, str]:
    options = " ".join(q.options.values()) if q.options else ""
    return {
        "stem": q.question,
        "stem_options": f"{q.question} {options}".strip(),
        "short": short_query(q.question),
    }


@dataclass(frozen=True)
class ShapeSummary:
    shape: str
    n: int
    hits: Ratio
    empty: Ratio
    criterion: str


def run_query_shape(run: LoadedRun) -> dict[str, Any]:
    comps = run.components
    (ingest,) = parse_records(run.files["ingest"], IngestRecord)
    questions = [comps.dataset.question_by_id[j.qid] for j in run.judge_records]
    located = [q for q in questions if q.evidence_status is EvidenceStatus.LOCATED]
    arm = comps.new_arm()
    rows = []
    try:
        arm.attach(comps.dataset, ingest.summary)
        for q in located:
            row: dict[str, Any] = {"qid": q.qid, "qtype": q.qtype, "is_mc": q.is_mc}
            for shape, text in queries(q).items():
                items = arm.search(text, top_k=comps.config.arm.top_k)
                rec = SearchRecord(qid=q.qid, qtype=q.qtype, evidence_status=q.evidence_status,
                                   window_class="A", query=text, items=items)  # fmt: skip
                hit = retrieval_hit(rec, q, HitCriterion.MESSAGE)
                criterion = "message"
                if hit is None:
                    hit, criterion = retrieval_hit(rec, q, HitCriterion.DATE), "date"
                row[shape] = {"hit": bool(hit), "n_items": len(items), "criterion": criterion,
                              "query_chars": len(text)}  # fmt: skip
            rows.append(row)
    finally:
        arm.close()
    summaries = []
    for shape in SHAPES:
        n = len(rows)
        summaries.append(
            asdict(
                ShapeSummary(
                    shape=shape,
                    n=n,
                    hits=Ratio(sum(r[shape]["hit"] for r in rows), n),
                    empty=Ratio(sum(r[shape]["n_items"] == 0 for r in rows), n),
                    criterion=rows[0][shape]["criterion"] if rows else "n/a",
                )
            )
        )
    return {
        "arm": run.meta.arm,
        "corpus_id": run.meta.corpus_id,
        "arm_config_hash": run.meta.arm_config_hash,
        "question_set": run.meta.question_set,
        "short_query": f"automatic: first {SHORT_QUERY_WORDS} content words (not manual)",
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "summary": summaries,
        "rows": rows,
    }


def write_query_shape(run: LoadedRun) -> Path:
    path = run.run_dir / QUERY_SHAPE_FILE
    write_json_atomic(path, run_query_shape(run))
    return path


def summary_ratios(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-create Ratio objects from a saved result (for the report)."""
    out = []
    for s in result["summary"]:
        out.append({**s, "hits": Ratio(**s["hits"]), "empty": Ratio(**s["empty"])})
    return out
