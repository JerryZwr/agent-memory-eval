"""The whole live path offline: real pipeline, ledger, cache, probes, mem0 and re-judging.

Only the HTTP endpoint (a fake that answers like the models would) and mem0's
embedder (a hashed bag of words) are fake. Proves, before any money is spent:
probes run before / after ingest / after the run; caps stop the run; resuming
costs nothing; v2 re-judges the *same* answers; a red pre-probe measures nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mem0")
openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

from agentmemeval.arms.mem0 import Mem0Arm  # noqa: E402
from agentmemeval.live import rejudge  # noqa: E402
from agentmemeval.llm.base import ChatMessage  # noqa: E402
from agentmemeval.llm.openai_compat import OpenAICompatLLM  # noqa: E402
from agentmemeval.llm.scripted import extractive_reader  # noqa: E402
from agentmemeval.llm.usage import BudgetExceededError, UsageLedger  # noqa: E402
from agentmemeval.pipeline.config import (  # noqa: E402
    DEFAULT_FIXTURE,
    AnswerSpec,
    ArmSpec,
    DatasetSpec,
    JudgeSpec,
    LiveSpec,
    RunConfig,
)
from agentmemeval.pipeline.runner import Action, read_probes, run_pipeline  # noqa: E402
from agentmemeval.pipeline.store import stage_path  # noqa: E402
from agentmemeval.report.build import write_report  # noqa: E402
from agentmemeval.text import normalize  # noqa: E402

pytestmark = pytest.mark.unit
DIMS = 64
REQ = httpx2.Request("POST", "https://api.example/chat/completions")


def _reply(text: str):
    usage = NS(prompt_tokens=200, completion_tokens=20, model_extra={"prompt_cache_hit_tokens": 0},
               prompt_tokens_details=None)  # fmt: skip
    return NS(choices=[NS(message=NS(content=text, tool_calls=None), finish_reason="stop")],
              model="fake", usage=usage)  # fmt: skip


class FakeEndpoint:
    """Answers like the three kinds of callers expect: judge, mem0 extraction, answering."""

    def __init__(self):
        self.requests = []
        self.down = False
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **req):
        self.requests.append(req)
        if self.down:
            raise openai.APIConnectionError(request=REQ)
        system = req["messages"][0]["content"]
        user = req["messages"][-1]["content"]
        if system.startswith("You grade"):
            gold = re.search(r"GOLD: (.*)\nANSWER:", user, re.S).group(1)
            answer = re.search(r"ANSWER: (.*)\nReturn", user, re.S).group(1)
            ok = normalize(gold).strip(" .") in normalize(answer)
            return _reply(json.dumps({"verdict": "CORRECT" if ok else "WRONG", "reason": "fake"}))
        if req.get("response_format"):  # mem0 extraction
            lines = re.findall(r"\] ([^:\n]+): ([^\n\"]+)", user)
            return _reply(json.dumps({"memory": [{"text": f"{w} said {t}"} for w, t in lines]}))
        msgs = [ChatMessage(m["role"], m["content"]) for m in req["messages"]]
        return _reply(extractive_reader(msgs))


class HashEmbedder:
    def _vec(self, text):
        v = [0.0] * DIMS
        for tok in re.findall(r"[a-z0-9]+", str(text).lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIMS] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, text, memory_action=None):
        return self._vec(text)

    def embed_batch(self, texts, memory_action="add"):
        return [self._vec(t) for t in texts]


def _factory(config):
    from mem0 import Memory

    config = json.loads(json.dumps(config))
    config["embedder"] = {"provider": "openai", "config": {"api_key": "offline-test"}}
    config["vector_store"]["config"]["embedding_model_dims"] = DIMS
    memory = Memory.from_config(config)
    memory.embedding_model = HashEmbedder()
    return memory


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(OpenAICompatLLM, "client", property(lambda self: fake))
    monkeypatch.setattr(Mem0Arm, "default_memory_factory", staticmethod(_factory))
    monkeypatch.setattr("agentmemeval.llm.openai_compat.time.sleep", lambda s: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test")
    return fake


def _config(session: Path, arm: str, *, max_calls=400, version="v1") -> RunConfig:
    params = {"store_dir": (session / arm / "store").as_posix()} if arm == "mem0" else {}
    return RunConfig(
        run_id=f"{session.name}-{arm}",
        dataset=DatasetSpec(kind="fixture", path=DEFAULT_FIXTURE.as_posix()),
        arm=ArmSpec(name=arm, params=params),
        answer=AnswerSpec(kind="openai_compat"),
        judge=JudgeSpec(kind="llm", version=version),
        live=LiveSpec(
            ledger=(session / "usage.jsonl").as_posix(),
            cache=(session / "cache.jsonl").as_posix(),
            max_calls=max_calls,
            max_cost_cny=5.0,
        ),
    )


def _ledger(session: Path) -> UsageLedger:
    return UsageLedger(session / "usage.jsonl", max_calls=10**6, max_cost_cny=10**6)


@pytest.mark.parametrize("arm", ["null", "bm25", "mem0"])
def test_live_run_probes_and_records_usage(endpoint, tmp_path, arm):
    res = run_pipeline(tmp_path / arm, _config(tmp_path, arm))
    assert set(res.actions.values()) == {Action.RAN}
    assert res.not_measured_reasons == []
    probes = read_probes(tmp_path / arm)
    assert probes["round_status"] == "measured"
    assert {k for k in ("pre", "after_ingest", "post") if k in probes} == {
        "pre",
        "after_ingest",
        "post",
    }
    if arm != "null":
        assert "canary=" in probes["after_ingest"]["corpus"]["detail"]
    purposes = _ledger(tmp_path).totals.by_purpose
    assert purposes["answer"]["calls"] == 16
    assert purposes["judge-v1"]["calls"] > 0  # open questions only; MC is deterministic
    if arm == "mem0":
        assert purposes["mem0-extract"]["calls"] == 6  # one per (date, group)
    assert probes["usage"]["calls"] == _ledger(tmp_path).totals.calls


def test_resuming_a_finished_run_pays_only_for_probes(endpoint, tmp_path):
    run_pipeline(tmp_path / "bm25", _config(tmp_path, "bm25"))
    before = _ledger(tmp_path).totals
    res = run_pipeline(tmp_path / "bm25", _config(tmp_path, "bm25"))
    after = _ledger(tmp_path).totals
    assert set(res.actions.values()) == {Action.REUSED}
    new = {k: after.by_purpose[k]["calls"] - before.by_purpose.get(k, {"calls": 0})["calls"]
           for k in after.by_purpose}  # fmt: skip
    assert new.get("answer", 0) == 0  # every answer came from the stage file
    assert new["probe"] == 2  # the uncached reachability probe, pre and post
    assert new.get("judge-v1", 0) == 0


def test_v2_rejudges_the_same_answers(endpoint, tmp_path):
    src = tmp_path / "bm25"
    run_pipeline(src, _config(tmp_path, "bm25"))
    target = rejudge(src, tmp_path / "v2" / "bm25", version="v2")
    res = run_pipeline(target, probe=False)
    assert res.actions == {"ingest": Action.REUSED, "search": Action.REUSED,
                           "answer": Action.REUSED, "judge": Action.RAN}  # fmt: skip
    assert stage_path(target, "answer").read_bytes() == stage_path(src, "answer").read_bytes()
    assert res.components.meta.judge_prompt_version == "llm-v2"
    judge_prompts = [r["messages"][0]["content"] for r in endpoint.requests
                     if r["messages"][0]["content"].startswith("You grade")]  # fmt: skip
    assert any("4. If the gold answer says" in p for p in judge_prompts)  # the v2 rules were sent


def test_call_cap_stops_the_run_and_writes_nothing_partial(endpoint, tmp_path):
    with pytest.raises(BudgetExceededError, match="call cap"):
        run_pipeline(tmp_path / "bm25", _config(tmp_path, "bm25", max_calls=8))
    assert stage_path(tmp_path / "bm25", "search").exists()
    assert not stage_path(tmp_path / "bm25", "answer").exists()  # no half-written answers


def test_red_pre_probe_measures_nothing(endpoint, tmp_path):
    from agentmemeval.errors import StageError

    endpoint.down = True
    with pytest.warns(UserWarning), pytest.raises(StageError, match="pre-probe not green"):
        run_pipeline(tmp_path / "bm25", _config(tmp_path, "bm25"))
    assert not stage_path(tmp_path / "bm25", "ingest").exists()
    assert read_probes(tmp_path / "bm25")["pre"]["llm"]["status"] == "fail"


def test_red_post_probe_marks_the_round_not_measured(endpoint, tmp_path, monkeypatch):
    import agentmemeval.pipeline.runner as runner

    real_post = runner.LiveProbes.post

    def post_with_outage(self, ingest_file):
        endpoint.down = True  # the endpoint dies after the last answer was judged
        return real_post(self, ingest_file)

    monkeypatch.setattr(runner.LiveProbes, "post", post_with_outage)
    with pytest.warns(UserWarning):
        res = run_pipeline(tmp_path / "bm25", _config(tmp_path, "bm25"))
    assert res.not_measured_reasons and "post-probe llm fail" in res.not_measured_reasons[0]
    assert read_probes(tmp_path / "bm25")["round_status"] == "not_measured"


def test_report_over_a_live_session(endpoint, tmp_path):
    for arm in ("null", "bm25", "mem0"):
        run_pipeline(tmp_path / arm, _config(tmp_path, arm))
    html = write_report(tmp_path).read_text(encoding="utf-8")
    assert "deepseek-v4-pro+thinking=off" in html
    assert "不适用" in html  # mem0 returns memories: the message-level hit criterion n/a


def test_live_report_sections_and_public_mode(endpoint, tmp_path):
    from agentmemeval.analysis.query_shape import write_query_shape
    from agentmemeval.pipeline.runner import load_run

    for arm in ("null", "bm25"):
        run_pipeline(tmp_path / arm, _config(tmp_path, arm))
    write_query_shape(load_run(tmp_path / "bm25"))
    target = rejudge(tmp_path / "bm25", tmp_path / "v2" / "bm25", version="v2")
    run_pipeline(target, probe=False)

    full = write_report(tmp_path).read_text(encoding="utf-8")
    for heading in ("运行状态与费用", "查询形态敏感性实验", "评分员口径对比", "未经人工校准"):
        assert heading in full, heading
    question = "Which database is the primary database for the Heron scheduling service?"
    assert question in full

    public = write_report(tmp_path, tmp_path / "public.html", public=True).read_text(
        encoding="utf-8"
    )
    assert question not in public  # no dataset text in the published version
    assert "PostgreSQL" not in public  # no answers or retrieved dialogue either
    assert "usage.jsonl" in public and str(tmp_path) not in public  # no local paths


def test_publish_bundle_has_no_dataset_text_or_local_paths(endpoint, tmp_path):
    from agentmemeval.report.publish import publish

    session = tmp_path / "session"
    for arm in ("null", "bm25"):
        run_pipeline(session / arm, _config(session, arm))
    report, summary = publish(session, tmp_path / "public")
    text = report.read_text(encoding="utf-8") + summary.read_text(encoding="utf-8")
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert set(data["arms"]) == {"null", "bm25"}
    row = data["arms"]["bm25"]["questions"][0]
    assert set(row) >= {"qid", "outcome", "verdict", "hit_message", "hit_date"}
    assert data["arms"]["bm25"]["stage_records_sha256"]["judge"]
    for leak in ("Which database is the primary database", "PostgreSQL", str(tmp_path),
                 tmp_path.as_posix()):  # fmt: skip
        assert leak not in text, leak
