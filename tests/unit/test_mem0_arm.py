"""The mem0 arm on mem0's real code path, with a fake LLM and a fake embedder (offline).

mem0 is built from its normal config; only its LLM (our adapter over a scripted
model) and its embedder (a hashed bag of words) are swapped, so add/search,
the local Qdrant store and the metadata round trip are all exercised without a
key, a network or Ollama.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import pytest

pytest.importorskip("mem0")

from agentmemeval.arms.mem0 import Mem0Arm
from agentmemeval.errors import LLMUnavailableError, MemoryServiceError, StageError
from agentmemeval.llm.base import LLMResponse

pytestmark = pytest.mark.unit
DIMS = 64


class HashEmbedder:
    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIMS
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIMS] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, text, memory_action=None):
        return self._vec(text if isinstance(text, str) else json.dumps(text))

    def embed_batch(self, texts, memory_action="add"):
        return [self._vec(t) for t in texts]


class ExtractingLLM:
    """Stands in for the extraction model: one memory per chat line it is shown."""

    model = "fake/extractor"

    def __init__(self, fail=False, reply=None):
        self.fail = fail
        self.reply = reply
        self.calls = 0

    def complete(self, messages, *, temperature=0.0, tools=None, response_format=None):
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("503 from extraction endpoint")
        if self.reply is not None:
            return LLMResponse(text=self.reply, model=self.model)
        user = messages[-1].content
        lines = re.findall(r"\] ([^:\n]+): ([^\n\"]+)", user)
        memories = [{"text": f"{who.strip()} said {what.strip()}"} for who, what in lines]
        return LLMResponse(text=json.dumps({"memory": memories}), model=self.model)


def _factory(config):
    from mem0 import Memory

    config = json.loads(json.dumps(config))
    config["embedder"] = {"provider": "openai", "config": {"api_key": "offline-test"}}
    config["vector_store"]["config"]["embedding_model_dims"] = DIMS
    memory = Memory.from_config(config)
    memory.embedding_model = HashEmbedder()
    return memory


@pytest.fixture
def make_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test")

    def make(llm=None, namespace="run-1"):
        return Mem0Arm(store_dir=str(tmp_path / "store"), namespace=namespace,
                       llm=llm or ExtractingLLM(), memory_factory=_factory)  # fmt: skip

    return make


def test_ingest_one_add_per_date_and_group_and_search_returns_memories(make_arm, mini):
    llm = ExtractingLLM()
    arm = make_arm(llm)
    summary = arm.add(mini)
    assert summary["n_batches"] == 6  # 3 days x 2 groups
    assert llm.calls == 6  # exactly one extraction call per batch
    assert summary["n_memories"] > 0
    assert summary["n_batches_without_memories"] == 0
    hits = arm.search("Which database is the primary database for Heron?", top_k=3)
    assert hits and all(h.kind == "memory" for h in hits)
    assert hits[0].date in mini.days  # the date travelled through mem0 metadata
    assert hits[0].message_id.startswith("mem0:")


def test_reattach_reads_the_same_store(make_arm, mini):
    arm = make_arm()
    summary = arm.add(mini)
    query = "nightly backup job time"
    first = arm.search(query, 3)
    arm._memory.vector_store.client.close()  # release the local Qdrant lock, as a new process would
    fresh = make_arm()
    fresh.attach(mini, summary)
    assert [h.text for h in fresh.search(query, 3)] == [h.text for h in first]


def test_attach_refuses_another_namespace_or_corpus(make_arm, mini):
    summary = make_arm().add(mini)
    with pytest.raises(StageError, match="namespace"):
        make_arm(namespace="run-2").attach(mini, summary)
    with pytest.raises(StageError, match="corpus differs"):
        make_arm().attach(mini.first_days(2), summary)


def test_extraction_outage_surfaces_as_infra_error_not_a_silent_empty_store(make_arm, mini):
    with pytest.raises(LLMUnavailableError, match="503"):
        make_arm(ExtractingLLM(fail=True)).add(mini)


def test_replies_without_memories_for_every_batch_are_refused(make_arm, mini):
    with pytest.raises(MemoryServiceError, match="stored no memories"):
        make_arm(ExtractingLLM(reply='{"facts": []}')).add(mini)


def test_params_describe_behaviour_not_locations(make_arm):
    params = make_arm().params()
    assert params["implementation"] == "mem0ai-oss"
    assert params["extraction_llm"] == "fake/extractor"
    assert "store_dir" not in params and "namespace" not in params
