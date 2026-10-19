"""OpenAICompatLLM, the usage ledger and the response cache, against a fake client (offline).

No key, no network, no cost: the fake client stands in for the HTTP API and can
raise the SDK's real exception types.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

from agentmemeval.errors import LLMUnavailableError  # noqa: E402
from agentmemeval.llm.base import ChatMessage, ToolCall, ToolSpec  # noqa: E402
from agentmemeval.llm.openai_compat import OpenAICompatLLM, _to_openai_messages  # noqa: E402
from agentmemeval.llm.usage import (  # noqa: E402
    BudgetExceededError,
    LiveConfigError,
    ResponseCache,
    UsageLedger,
    cost_cny,
)
from agentmemeval.tracing import FakeClock  # noqa: E402

pytestmark = pytest.mark.unit
REQ = httpx2.Request("POST", "https://api.example/chat/completions")


def _completion(text="ok", tool_calls=None, prompt=100, hit=40, out=10):
    extra = {"prompt_cache_hit_tokens": hit}
    usage = NS(prompt_tokens=prompt, completion_tokens=out, model_extra=extra,
               prompt_tokens_details=None)  # fmt: skip
    msg = NS(content=text, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason="stop")], model="deepseek-flash", usage=usage)


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **req):
        self.requests.append(req)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def ledger(tmp_path):
    return UsageLedger(tmp_path / "usage.jsonl", max_calls=10, max_cost_cny=1.0)


def _llm(client, ledger, tmp_path=None, **kw):
    cache = ResponseCache(tmp_path / "cache.jsonl") if tmp_path else None
    return OpenAICompatLLM("deepseek-flash", ledger=ledger, cache=cache, purpose="answer",
                           client=client, **kw)  # fmt: skip


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("agentmemeval.llm.openai_compat.time.sleep", lambda s: None)


def test_cost_uses_the_published_price_and_cache_hits():
    # 60 miss * 2 + 40 hit * 0.04 + 10 out * 8, per million tokens
    assert cost_cny("deepseek-flash", 100, 40, 10) == pytest.approx((120 + 1.6 + 80) / 1e6)
    with pytest.raises(LiveConfigError):
        cost_cny("unpriced-model", 1, 0, 1)


def test_request_disables_thinking_and_is_recorded(ledger):
    client = FakeClient(_completion("hello"))
    llm = _llm(client, ledger)
    reply = llm.complete([ChatMessage("user", "hi")])
    assert reply.text == "hello"
    assert client.requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert client.requests[0]["temperature"] == 0.0
    assert llm.model == "deepseek-flash+thinking=off"
    assert ledger.totals.calls == 1
    assert ledger.totals.cache_hit_tokens == 40


def test_identical_requests_hit_the_cache_and_cost_nothing(ledger, tmp_path):
    client = FakeClient(_completion("first"))
    llm = _llm(client, ledger, tmp_path)
    llm.complete([ChatMessage("user", "same")])
    again = _llm(FakeClient(), ledger, tmp_path).complete(
        [ChatMessage("user", "same")]
    )  # no reply queued
    assert again.text == "first"
    assert (ledger.totals.calls, ledger.totals.cache_hits) == (1, 1)


def test_call_cap_refuses_before_calling(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl", max_calls=1, max_cost_cny=10)
    client = FakeClient(_completion(), _completion())
    llm = _llm(client, ledger)
    llm.complete([ChatMessage("user", "1")])
    with pytest.raises(BudgetExceededError, match="call cap"):
        llm.complete([ChatMessage("user", "2")])
    assert len(client.requests) == 1  # the second request was never sent


def test_cost_cap_is_enforced_and_survives_a_restart(tmp_path):
    path = tmp_path / "u.jsonl"
    ledger = UsageLedger(path, max_calls=100, max_cost_cny=0.0001)
    _llm(FakeClient(_completion(prompt=100_000, hit=0, out=0)), ledger).complete(
        [ChatMessage("user", "x")]
    )
    reloaded = UsageLedger(path, max_calls=100, max_cost_cny=0.0001)  # a new process
    with pytest.raises(BudgetExceededError, match="cost cap"):
        _llm(FakeClient(_completion()), reloaded).complete([ChatMessage("user", "y")])


def test_transient_errors_are_retried_loudly(ledger):
    client = FakeClient(openai.APITimeoutError(request=REQ), _completion("recovered"))
    with pytest.warns(UserWarning) as caught:
        reply = _llm(client, ledger).complete([ChatMessage("user", "x")])
    messages = [str(w.message) for w in caught]
    assert any("retrying" in m for m in messages)
    assert any("succeeded on attempt 2" in m for m in messages)  # recovery is not silent
    assert reply.text == "recovered"
    assert ledger.totals.retries == 1


def test_persistent_errors_become_llm_unavailable(ledger):
    rate = openai.RateLimitError("429", response=httpx2.Response(429, request=REQ), body=None)
    client = FakeClient(rate, rate, rate)
    with pytest.warns(UserWarning), pytest.raises(LLMUnavailableError, match="after 3 attempts"):
        _llm(client, ledger).complete([ChatMessage("user", "x")])
    assert ledger.totals.failures == 1


def test_rejected_credentials_stop_the_run_instead_of_becoming_env_failures(ledger):
    auth = openai.AuthenticationError("401", response=httpx2.Response(401, request=REQ), body=None)
    with pytest.raises(LiveConfigError, match="credentials rejected"):
        _llm(FakeClient(auth), ledger).complete([ChatMessage("user", "x")])


def test_empty_json_mode_content_is_retried_then_raised(ledger):
    client = FakeClient(_completion(""), _completion("not json"), _completion("{}"))
    with pytest.warns(UserWarning):
        reply = _llm(client, ledger).complete(
            [ChatMessage("user", "reply in json")], response_format={"type": "json_object"}
        )
    assert reply.text == "{}"
    assert ledger.totals.calls == 3  # every attempt was paid for and counted


def test_tool_calls_round_trip_with_their_ids(ledger):
    tc = NS(id="call_abc", function=NS(name="memory_search", arguments='{"query": "beta"}'))
    client = FakeClient(_completion(text=None, tool_calls=[tc]))
    spec = ToolSpec("memory_search", "search", {"type": "object"})
    reply = _llm(client, ledger).complete([ChatMessage("user", "q")], tools=[spec])
    assert reply.tool_calls == (ToolCall("memory_search", {"query": "beta"}, id="call_abc"),)
    assert client.requests[0]["tools"][0]["function"]["name"] == "memory_search"

    history = [
        ChatMessage("user", "q"),
        ChatMessage("assistant", "", tool_call=reply.tool_calls[0]),
        ChatMessage("tool", json.dumps({"ok": True}), name="memory_search"),
    ]
    sent = _to_openai_messages(history)
    assert sent[1]["tool_calls"][0]["id"] == "call_abc"
    assert sent[2] == {"role": "tool", "tool_call_id": "call_abc", "content": '{"ok": true}'}


def test_streaming_measures_ttft_on_the_clock(ledger):
    clock = FakeClock()

    def chunks():
        clock.sleep(0.2)
        yield NS(
            usage=None,
            model="deepseek-flash",
            choices=[NS(delta=NS(content="Hel", tool_calls=None), finish_reason=None)],
        )
        clock.sleep(0.5)
        yield NS(
            usage=None,
            model="deepseek-flash",
            choices=[NS(delta=NS(content="lo", tool_calls=None), finish_reason="stop")],
        )
        final_usage = NS(prompt_tokens=5, completion_tokens=2, model_extra={},
                         prompt_tokens_details=None)  # fmt: skip
        yield NS(usage=final_usage, model="deepseek-flash", choices=[])

    client = FakeClient(chunks())
    reply = _llm(client, ledger, stream=True, clock=clock).complete([ChatMessage("user", "x")])
    assert reply.text == "Hello"
    assert reply.ttft_s == pytest.approx(0.2)
    assert client.requests[0]["stream_options"] == {"include_usage": True}


def test_missing_key_is_a_config_error(monkeypatch, ledger):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    llm = OpenAICompatLLM("deepseek-flash", ledger=ledger, cache=None, purpose="x")
    assert llm.model == "deepseek-flash+thinking=off"  # constructing needs no key (reports)
    with pytest.raises(LiveConfigError, match="DEEPSEEK_API_KEY"):
        llm.complete([ChatMessage("user", "x")])  # the first paid call does
    assert ledger.totals.calls == 0
