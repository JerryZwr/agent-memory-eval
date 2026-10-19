"""Paired comparison of two arms on the same questions (METHODOLOGY 1.7).

Questions are paired by qid. Only pairs measured on both sides count (an
env_failure on either side drops the pair). "Pass" is delivered and CORRECT,
the end-to-end sense, so an undelivered answer counts as a failure.

The verdict wording follows 1.7: without a significant paired test we say
"no significant difference observed", never "better".
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from agentmemeval.analysis.outcome import Outcome
from agentmemeval.analysis.stats import (
    Interval,
    mcnemar_exact,
    min_discordant_for_significance,
    paired_accuracy_difference,
)
from agentmemeval.analysis.windows import WindowClass
from agentmemeval.pipeline.records import JudgeRecord

ALPHA = 0.05


def format_p(p: float) -> str:
    """`p = 0.031`, or `p < 0.001` rather than a misleading `p = 0.000`."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _passed(r: JudgeRecord) -> bool:
    return r.outcome is Outcome.DELIVERED and r.verdict == "CORRECT"


@dataclass(frozen=True)
class PairedComparison:
    control: str
    treatment: str
    n_pairs: int
    both_pass: int
    only_treatment: int  # b: treatment right, control wrong
    only_control: int  # c: control right, treatment wrong
    neither: int
    p_value: float
    difference: float | None  # treatment accuracy - control accuracy on the pairs
    difference_interval: Interval | None

    @property
    def discordant(self) -> int:
        return self.only_treatment + self.only_control

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def underpowered(self) -> bool:
        """Too few discordant pairs for any outcome to reach significance."""
        return self.discordant < min_discordant_for_significance(ALPHA)

    def verdict(self) -> str:
        if self.n_pairs == 0:
            return "没有可配对的题，无法比较"
        if not self.significant:
            text = f"未观察到显著差异（配对 McNemar 精确检验 {format_p(self.p_value)}）"
            if self.underpowered:
                need = min_discordant_for_significance(ALPHA)
                text += (
                    f"；结论不一致的配对只有 {self.discordant} 个，"
                    f"少于 {need} 个时任何结果都达不到 p < {ALPHA}"
                )
            return text
        direction = "高于" if self.only_treatment > self.only_control else "低于"
        return (
            f"{self.treatment} {direction} {self.control}"
            f"（配对 McNemar 精确检验 {format_p(self.p_value)}）"
        )


def compare(
    control_name: str,
    control: Sequence[JudgeRecord],
    treatment_name: str,
    treatment: Sequence[JudgeRecord],
    *,
    classes: Collection[WindowClass],
) -> PairedComparison:
    a = {r.qid: r for r in control if r.window_class in classes}
    b = {r.qid: r for r in treatment if r.window_class in classes}
    pairs: list[tuple[bool, bool]] = []
    for qid in sorted(a.keys() & b.keys()):
        ra, rb = a[qid], b[qid]
        if Outcome.ENV_FAILURE in (ra.outcome, rb.outcome):
            continue
        pairs.append((_passed(ra), _passed(rb)))
    only_t = sum(1 for x, y in pairs if y and not x)
    only_c = sum(1 for x, y in pairs if x and not y)
    diff, interval = paired_accuracy_difference(pairs)
    return PairedComparison(
        control=control_name,
        treatment=treatment_name,
        n_pairs=len(pairs),
        both_pass=sum(1 for x, y in pairs if x and y),
        only_treatment=only_t,
        only_control=only_c,
        neither=sum(1 for x, y in pairs if not x and not y),
        p_value=mcnemar_exact(only_t, only_c),
        difference=diff,
        difference_interval=interval,
    )
