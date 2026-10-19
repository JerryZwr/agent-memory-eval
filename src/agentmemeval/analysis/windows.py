"""Evidence-window classes and question selection (METHODOLOGY 1.3).

Only part of a corpus is usually ingested, but questions cover the whole
period, so asking every question would be wrong:

| class | evidence vs. the ingested days    | asked / scored by default |
|-------|-----------------------------------|---------------------------|
| A     | all inside                        | yes                       |
| B     | spans the boundary                | no (opt in)               |
| C     | all outside                       | no                        |
| D     | no evidence by design (F_HL)      | yes                       |
| E     | location unrecoverable            | no                        |
| F     | dataset marks scope unknown       | no                        |

B is the dangerous one: retrieval works and returns real memories, but the
material is incomplete, which looks exactly like a ranking defect. It is only
included when asked for explicitly.

F exists for completeness; neither current loader produces it (EverMemBench
batch 004 has no "scope unknown" marker).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

from agentmemeval.data.models import Dataset, EvidenceStatus, Question


class WindowClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


DESCRIPTIONS = {
    WindowClass.A: "证据全部在已灌入的天数内",
    WindowClass.B: "证据跨越窗口边界（材料残缺）",
    WindowClass.C: "证据全部在窗口外",
    WindowClass.D: "无证据题（正确答案是“没有发生 / 不知道”）",
    WindowClass.E: "证据位置无法恢复",
    WindowClass.F: "数据集标为范围未知",
}
DEFAULT_SCORED = frozenset({WindowClass.A, WindowClass.D})


def classify(question: Question, window_days: Iterable[str]) -> WindowClass:
    if question.evidence_status is EvidenceStatus.NO_EVIDENCE:
        return WindowClass.D
    if question.evidence_status is EvidenceStatus.UNRECOVERABLE:
        return WindowClass.E
    days = set(window_days)
    inside = [d in days for d in question.evidence_dates]
    if all(inside):
        return WindowClass.A
    if any(inside):
        return WindowClass.B
    return WindowClass.C


def classify_all(dataset: Dataset) -> dict[str, WindowClass]:
    """Class of every question relative to the days actually ingested in `dataset`."""
    days = dataset.days
    return {q.qid: classify(q, days) for q in dataset.questions}


def scored_classes(include_b: bool = False) -> frozenset[WindowClass]:
    return DEFAULT_SCORED | {WindowClass.B} if include_b else DEFAULT_SCORED


def select_questions(dataset: Dataset, *, include_b: bool = False) -> tuple[Question, ...]:
    keep = scored_classes(include_b)
    classes = classify_all(dataset)
    return tuple(q for q in dataset.questions if classes[q.qid] in keep)


def class_counts(dataset: Dataset) -> dict[WindowClass, int]:
    counts = Counter(classify_all(dataset).values())
    return {c: counts.get(c, 0) for c in WindowClass}
