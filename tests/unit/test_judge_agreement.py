"""v1-vs-v2 judge agreement: same answers only, and flips are reported per question."""

from __future__ import annotations

import pytest

from agentmemeval.analysis.judge_agreement import judge_agreement
from agentmemeval.errors import IncomparableError
from agentmemeval.judge.base import Judge, JudgeVerdict
from agentmemeval.judge.scripted import ScriptedJudge
from agentmemeval.pipeline.config import default_eval_config
from agentmemeval.pipeline.runner import load_run, run_pipeline
from agentmemeval.pipeline.store import stage_path

pytestmark = pytest.mark.unit


class StricterJudge(Judge):
    """Like the scripted judge, but rejects every answer longer than 120 characters."""

    model = "scripted-judge"
    prompt_version = "scripted-strict"

    def judge(self, item):
        v = ScriptedJudge().judge(item)
        if v.label == "CORRECT" and len(item.answer) > 120:
            return JudgeVerdict(label="WRONG", reason="too long", raw="WRONG: too long")
        return v


def test_flips_and_kappa_on_the_same_answers(tmp_path, monkeypatch):
    run_pipeline(tmp_path / "a", default_eval_config())
    (tmp_path / "b").mkdir()
    for stage in ("ingest", "search", "answer"):
        stage_path(tmp_path / "b", stage).write_bytes(
            stage_path(tmp_path / "a", stage).read_bytes()
        )
    run_pipeline(tmp_path / "b", default_eval_config(), judge=StricterJudge())
    first = load_run(tmp_path / "a")

    # load_run rebuilds the default judge; give the second run its own judge for loading
    monkeypatch.setattr(
        "agentmemeval.pipeline.config.build_judge", lambda spec, live: StricterJudge()
    )
    second = load_run(tmp_path / "b")
    agreement = judge_agreement(first, second)
    assert agreement.n == 11  # delivered open questions judged by the judge
    assert agreement.flips and all(
        f.first == "CORRECT" and f.second == "WRONG" for f in agreement.flips
    )
    assert agreement.agreement.num == agreement.n - len(agreement.flips)
    assert agreement.confusion["CORRECT"]["WRONG"] == len(agreement.flips)
    assert agreement.kappa is not None and agreement.kappa < 1.0


def test_different_answers_are_refused(tmp_path):
    run_pipeline(tmp_path / "a", default_eval_config())
    run_pipeline(tmp_path / "b", default_eval_config(arm="null"))
    with pytest.raises(IncomparableError, match="different answers"):
        judge_agreement(load_run(tmp_path / "a"), load_run(tmp_path / "b"))


def test_rule_check_flags_verdicts_that_contradict_the_rules(tmp_path, monkeypatch):
    from agentmemeval.analysis.judge_agreement import rule_check
    from agentmemeval.judge.base import JudgeVerdict

    class Contrarian(Judge):
        """Inverts the scripted judge: refusals on answerable questions become CORRECT, etc."""

        model = "scripted-judge"
        prompt_version = "scripted-contrarian"

        def judge(self, item):
            v = ScriptedJudge().judge(item)
            flipped = "WRONG" if v.label == "CORRECT" else "CORRECT"
            return JudgeVerdict(label=flipped, reason="inverted", raw=flipped)

    monkeypatch.setattr("agentmemeval.pipeline.config.build_judge", lambda spec, live: Contrarian())
    run_pipeline(tmp_path / "x", default_eval_config(arm="null"))  # null: open answers all refuse
    check = rule_check(load_run(tmp_path / "x"))
    assert check.checked == 11  # 8 answerable + 3 no-evidence open questions
    assert len(check.refusal_on_answerable_judged_correct) == 8
    assert len(check.refusal_on_no_evidence_judged_wrong) == 3


def test_rule_check_is_clean_for_the_scripted_judge(tmp_path):
    from agentmemeval.analysis.judge_agreement import rule_check

    run_pipeline(tmp_path / "x", default_eval_config(arm="null"))
    check = rule_check(load_run(tmp_path / "x"))
    assert check.refusal_on_answerable_judged_correct == []
    assert check.refusal_on_no_evidence_judged_wrong == []
