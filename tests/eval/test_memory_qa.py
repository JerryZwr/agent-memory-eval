"""Memory layer, one case per question (PLAN M3 task 4).

These are measurements, not a CI gate: a case fails when the system under test
answers wrong. Run with

    uv run pytest tests/eval --alluredir=reports/allure-results [--eval-qids FILE]
                             [--eval-config run.json] [--eval-baseline baselines/x.json]

Outcomes map onto the three conclusions of METHODOLOGY P3:

* wrong / undelivered answer  -> failed, message tagged [PRODUCT]  (product defect)
* env_failure during the run  -> skipped via not_measured()        (environment)
* an exception in this code   -> broken                            (test defect)
"""

from __future__ import annotations

import json

import allure
import pytest

from agentmemeval.analysis.baseline import entries_from_run
from agentmemeval.analysis.breakdown import HitCriterion, retrieval_hit
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.data.models import NO_EVIDENCE_QTYPE, QTYPE_LABELS
from agentmemeval.pipeline.runner import LoadedRun
from pytest_agenteval import not_measured
from pytest_agenteval.allure_files import PRODUCT_DEFECT_PREFIX

pytestmark = [
    pytest.mark.eval,
    pytest.mark.requires_env("llm", "memory_service", "corpus"),
]


def _by_qid(records):
    return {r.qid: r for r in records}


@allure.feature("逐题问答")
def test_memory_qa(eval_qid: str, eval_run: LoadedRun, eval_record, request) -> None:
    run = eval_run
    q = run.components.dataset.question_by_id[eval_qid]
    search = _by_qid(run.search_records)[eval_qid]
    answer = _by_qid(run.answer_records)[eval_qid]
    judged = _by_qid(run.judge_records)[eval_qid]

    eval_record.set_meta(run.meta)
    eval_record(_by_qid(entries_from_run(run))[eval_qid])

    label = QTYPE_LABELS.get(q.qtype, q.qtype)
    allure.dynamic.title(f"{q.qid} · {label} · {run.meta.arm}")
    allure.dynamic.story(label)
    allure.dynamic.severity(
        allure.severity_level.CRITICAL
        if q.qtype == NO_EVIDENCE_QTYPE
        else allure.severity_level.NORMAL
    )
    allure.dynamic.description(
        f"**题目**：{q.question}\n\n**标准答案**：{q.gold}\n\n"
        f"证据窗口类别 {judged.window_class.value}；语料 `{run.meta.corpus_id}`；"
        f"评分员 `{run.meta.judge_model}` ({run.meta.judge_prompt_version})"
    )
    allure.dynamic.parameter("arm", run.meta.arm)

    hit = retrieval_hit(search, q, HitCriterion.MESSAGE)
    hit_text = {True: "命中证据", False: "未命中证据", None: "本题无证据位置"}[hit]
    with allure.step(f"检索：返回 {len(search.items)} 条，{hit_text}"):
        evidence = set(q.evidence_ids)
        allure.attach(
            json.dumps(
                [
                    {**it.model_dump(), "is_evidence": it.message_id in evidence}
                    for it in search.items
                ],
                ensure_ascii=False,
                indent=2,
            ),
            name="检索返回",
            attachment_type=allure.attachment_type.JSON,
        )
    with allure.step(f"作答：结局 {answer.outcome.value}"):
        allure.attach(answer.response or "（无答案）", name="最终答案",
                      attachment_type=allure.attachment_type.TEXT)  # fmt: skip
    if answer.error is not None:
        allure.attach(answer.error.model_dump_json(indent=2), name="环境错误",
                      attachment_type=allure.attachment_type.JSON)  # fmt: skip
    with allure.step(f"判分：{judged.verdict or '未判分'}"):
        if judged.method == "mc_extraction":
            detail = (
                f"选择题确定性提取：{judged.extracted_choice or '无'}"
                f"（{judged.extraction_method}；ambiguous_extraction={judged.ambiguous_extraction}）"
            )
        else:
            detail = f"评分员理由：{judged.judge_reason}\n原始回复：{judged.judge_raw}"
        allure.attach(detail, name="判分依据", attachment_type=allure.attachment_type.TEXT)

    if judged.outcome is Outcome.ENV_FAILURE:
        not_measured(
            request, f"{eval_qid}: env_failure ({answer.error.type if answer.error else '?'})"
        )
    if judged.outcome is Outcome.UNDELIVERED:
        pytest.fail(f"{PRODUCT_DEFECT_PREFIX}{eval_qid}: undelivered ({answer.undelivered_reason})")
    assert judged.verdict == "CORRECT", (
        f"{PRODUCT_DEFECT_PREFIX}{eval_qid} judged {judged.verdict}; gold={q.gold!r}; "
        f"retrieval {hit_text}; answer={(judged.answer or '')[:160]!r}"
    )
