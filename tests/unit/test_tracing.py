"""Span tracer, clocks, and the ScriptedLLM's simulated streaming."""

from __future__ import annotations

import pytest

from agentmemeval.llm.base import ToolCall
from agentmemeval.llm.scripted import ScriptedLLM
from agentmemeval.tracing import FakeClock, Tracer, latency_breakdown

pytestmark = pytest.mark.unit


def test_spans_nest_and_record_parent_and_times():
    clock = FakeClock()
    tracer = Tracer(clock)
    with tracer.span("run", "r"):
        clock.sleep(1)
        with tracer.span("tool_call", "t", query="q") as s:
            clock.sleep(2)
            s.attributes["n_results"] = 3
    run, tool = tracer.trace.spans
    assert (tool.parent_id, tool.start, tool.end) == (run.span_id, 1, 3)
    assert run.duration == 3
    assert tool.attributes == {"query": "q", "n_results": 3}


def test_a_raising_span_is_closed_and_records_the_error():
    tracer = Tracer(FakeClock())
    with pytest.raises(ValueError), tracer.span("run", "r"):
        raise ValueError("boom")
    (span,) = tracer.trace.spans
    assert span.end is not None
    assert span.error == "ValueError: boom"


def test_unknown_span_kind_is_rejected():
    with pytest.raises(ValueError), Tracer().span("db_call", "x"):  # type: ignore[arg-type]
        pass


def test_fake_clock_refuses_negative_sleep():
    with pytest.raises(ValueError):
        FakeClock().sleep(-1)


def test_breakdown_counts_only_direct_children():
    clock = FakeClock()
    tracer = Tracer(clock)
    with tracer.span("run", "r"):
        with tracer.span("tool_call", "outer"):
            clock.sleep(1)
            with tracer.span("tool_call", "nested"):  # must not be counted twice
                clock.sleep(1)
        clock.sleep(0.5)
    b = latency_breakdown(tracer.trace)
    assert (b.total_ms, b.tool_ms, b.framework_ms) == (2500, 2000, 500)


class TestSimulatedStreaming:
    def test_ttft_is_measured_on_the_clock(self):
        clock = FakeClock()
        llm = ScriptedLLM(lambda m: "hello world, streamed", model="s", clock=clock,
                          latency_s=1.0, ttft_s=0.25)  # fmt: skip
        reply = llm.complete([])
        assert reply.text == "hello world, streamed"
        assert reply.ttft_s == pytest.approx(0.25)
        assert clock.now() == pytest.approx(1.0)

    def test_tool_call_replies_stream_too(self):
        clock = FakeClock()
        llm = ScriptedLLM(lambda m: ToolCall("memory_search", {"query": "x"}), model="s",
                          clock=clock, latency_s=0.5, ttft_s=0.1)  # fmt: skip
        reply = llm.complete([])
        assert reply.tool_calls == (ToolCall("memory_search", {"query": "x"}),)
        assert reply.ttft_s == pytest.approx(0.1)
        assert clock.now() == pytest.approx(0.5)

    def test_without_a_clock_there_is_no_ttft(self):
        assert ScriptedLLM(lambda m: "x", model="s").complete([]).ttft_s is None

    def test_latency_below_ttft_is_rejected(self):
        with pytest.raises(ValueError):
            ScriptedLLM(lambda m: "x", model="s", latency_s=0.1, ttft_s=0.2)
