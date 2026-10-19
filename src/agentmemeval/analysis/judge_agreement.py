"""Agreement between two judge prompt versions on the *same* answers (METHODOLOGY 1.5, 1.6).

This is judge-vs-judge, not judge-vs-human: it shows where the v2 rules change
verdicts, not which version is right. Human calibration is not done in this
project (docs/decisions.md D-009), and no human agreement figure is reported.

Refused unless both runs judged byte-identical answers (same answer-stage
records hash), so the prompt is the only variable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.scoring import Ratio
from agentmemeval.analysis.stats import Interval, cohen_kappa_with_interval
from agentmemeval.data.models import NO_EVIDENCE_QTYPE
from agentmemeval.errors import IncomparableError
from agentmemeval.judge.scripted import is_refusal
from agentmemeval.pipeline.runner import LoadedRun
from agentmemeval.text import normalize


@dataclass(frozen=True)
class Flip:
    qid: str
    qtype: str
    first: str
    second: str
    first_reason: str | None
    second_reason: str | None


@dataclass(frozen=True)
class Agreement:
    first_version: str
    second_version: str
    n: int
    agreement: Ratio
    kappa: float | None
    kappa_interval: Interval | None
    confusion: dict[str, dict[str, int]]  # first -> second -> count
    by_qtype: dict[str, Ratio]
    flips: list[Flip]


def judge_agreement(first: LoadedRun, second: LoadedRun) -> Agreement:
    if first.files["answer"].records_sha256 != second.files["answer"].records_sha256:
        raise IncomparableError(
            "the two runs judged different answers; the prompt is not the only variable"
        )
    a = {
        r.qid: r
        for r in first.judge_records
        if r.method == "judge" and r.outcome is Outcome.DELIVERED
    }
    b = {
        r.qid: r
        for r in second.judge_records
        if r.method == "judge" and r.outcome is Outcome.DELIVERED
    }
    qids = sorted(a.keys() & b.keys())
    labels_a = [a[q].verdict for q in qids]
    labels_b = [b[q].verdict for q in qids]
    kappa, interval = cohen_kappa_with_interval(labels_a, labels_b) if qids else (None, None)
    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for x, y in zip(labels_a, labels_b, strict=True):
        confusion[x][y] = confusion[x].get(y, 0) + 1
    per_type_total: Counter[str] = Counter()
    per_type_same: Counter[str] = Counter()
    flips = []
    for q in qids:
        qtype = a[q].qtype
        per_type_total[qtype] += 1
        if a[q].verdict == b[q].verdict:
            per_type_same[qtype] += 1
        else:
            flips.append(Flip(q, qtype, a[q].verdict, b[q].verdict, a[q].judge_reason,
                              b[q].judge_reason))  # fmt: skip
    return Agreement(
        first_version=first.meta.judge_prompt_version,
        second_version=second.meta.judge_prompt_version,
        n=len(qids),
        agreement=Ratio(sum(x == y for x, y in zip(labels_a, labels_b, strict=True)), len(qids)),
        kappa=kappa,
        kappa_interval=interval,
        confusion={k: dict(v) for k, v in confusion.items()},
        by_qtype={t: Ratio(per_type_same[t], n) for t, n in sorted(per_type_total.items())},
        flips=flips,
    )


@dataclass(frozen=True)
class RuleCheck:
    """Verdicts that contradict the judge's own written rules (automatic; not human labels)."""

    version: str
    refusal_on_answerable_judged_correct: list[str]  # violates rule 1 (key information missing)
    refusal_on_no_evidence_judged_wrong: list[str]  # violates v2 rule 4; a known v1 weakness
    checked: int


def rule_check(run: LoadedRun) -> RuleCheck:
    """Mechanical consistency check of LLM-judged verdicts against the prompt's rules.

    * answerable question, the answer is a refusal and does not contain the gold
      answer, verdict CORRECT: contradicts rule 1 in every version;
    * no-evidence question (F_HL), the answer is a pure refusal, verdict WRONG:
      contradicts v2's rule 4 (v1 has no such rule; reported as its known weakness).
    """
    bad_correct, bad_wrong = [], []
    checked = 0
    for r in run.judge_records:
        if r.method != "judge" or r.outcome is not Outcome.DELIVERED or r.answer is None:
            continue
        checked += 1
        refused = is_refusal(r.answer)
        has_gold = normalize(r.gold).strip(" .") in normalize(r.answer)
        if r.qtype != NO_EVIDENCE_QTYPE and refused and not has_gold and r.verdict == "CORRECT":
            bad_correct.append(r.qid)
        if r.qtype == NO_EVIDENCE_QTYPE and refused and r.verdict == "WRONG":
            bad_wrong.append(r.qid)
    return RuleCheck(run.meta.judge_prompt_version, bad_correct, bad_wrong, checked)


def as_dict(agreement: Agreement) -> dict[str, Any]:
    return {
        "first": agreement.first_version,
        "second": agreement.second_version,
        "n": agreement.n,
        "agreement": str(agreement.agreement),
        "kappa": agreement.kappa,
        "flips": len(agreement.flips),
    }
