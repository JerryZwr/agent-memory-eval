"""Real LLM calls through any OpenAI-compatible endpoint (DeepSeek, OpenRouter, ...). Live only.

Every call goes through one `UsageLedger` (hard caps, checked before the call)
and one `ResponseCache` (identical requests are never paid twice).

Error mapping (METHODOLOGY 1.4, 3.4):

* connection errors, timeouts, 429, 5xx  -> retried with backoff; each retry
  is logged and warned about (a retry that succeeds is not silent); after the
  last attempt -> `LLMUnavailableError` (an InfraError: env_failure);
* rejected credentials -> `LiveConfigError`: the whole run stops, instead of
  every question quietly becoming an env_failure;
* other 4xx -> raised as is: a bug in the request, not the environment.

DeepSeek specifics (verified 2026-10-19, api-docs.deepseek.com): thinking mode
is on by default and ignores `temperature`, so it is switched off explicitly
(`thinking=False`) to make temperature 0 effective; the model string recorded
in metadata says so.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Sequence
from typing import Any

from agentmemeval.errors import LLMUnavailableError
from agentmemeval.llm.base import LLM, ChatMessage, LLMResponse, ToolCall, ToolSpec
from agentmemeval.llm.usage import LiveConfigError, ResponseCache, UsageLedger
from agentmemeval.tracing import Clock, RealClock

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMOutputError(LLMUnavailableError):
    """The provider answered, but not in the requested shape (e.g. empty JSON-mode content)."""


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_call is not None:
            call = m.tool_call
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{len(out)}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
        elif m.role == "tool":
            previous = next(
                (o for o in reversed(out) if o["role"] == "assistant" and o.get("tool_calls")), None
            )
            call_id = previous["tool_calls"][0]["id"] if previous else "call_0"
            out.append({"role": "tool", "tool_call_id": call_id, "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return value if isinstance(value, dict) else {"__invalid_json__": raw}


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        model: str,
        *,
        ledger: UsageLedger,
        cache: ResponseCache | None,
        purpose: str,
        base_url: str = DEEPSEEK_BASE_URL,
        api_key_env: str = "DEEPSEEK_API_KEY",
        thinking: bool | None = False,
        max_tokens: int = 1024,
        timeout_s: float = 90.0,
        max_attempts: int = 3,
        stream: bool = False,
        clock: Clock | None = None,
        client: Any = None,
    ):
        self._model = model
        self.ledger = ledger
        self.cache = cache
        self.purpose = purpose
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.stream = stream
        self.clock = clock or RealClock()
        # The HTTP client (and the key check) is created on the first paid call, so building
        # components for a report or resuming a finished run never needs the key.
        self._client = client
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai  # live-only dependency

            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise LiveConfigError(f"{self._api_key_env} is not set (put it in .env)")
            self._client = openai.OpenAI(
                api_key=api_key, base_url=self._base_url, timeout=self._timeout_s, max_retries=0
            )
        return self._client

    @property
    def model(self) -> str:
        suffix = "" if self.thinking is None else f"+thinking={'on' if self.thinking else 'off'}"
        return f"{self._model}{suffix}"

    # ------------------------------------------------------------------ request

    def _request(
        self,
        messages: Sequence[ChatMessage],
        temperature: float,
        tools: Sequence[ToolSpec] | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            req["tools"] = _to_openai_tools(tools)
        if response_format:
            req["response_format"] = response_format
        if self.thinking is not None:
            req["extra_body"] = {"thinking": {"type": "enabled" if self.thinking else "disabled"}}
        return req

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
        use_cache: bool = True,
        purpose: str | None = None,
    ) -> LLMResponse:
        """`use_cache=False` for probes: a cached reply proves nothing about reachability.

        Such calls neither read nor write the cache. `purpose` overrides the ledger label.
        """
        purpose = purpose or self.purpose
        req = self._request(messages, temperature, tools, response_format)
        key = ResponseCache.key({**req, "stream": self.stream})
        cached = self.cache.get(key) if self.cache and use_cache else None
        if cached is not None:
            self.ledger.record("cache_hit", purpose=purpose, model=self._model)
            return self._response_from(cached, ttft_s=None)

        want_json = bool(response_format and response_format.get("type") == "json_object")
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.ledger.check()  # before every paid attempt
            try:
                raw, ttft = self._call(req, purpose)
            except LLMUnavailableError as exc:
                last_error = exc
                self._note_retry(attempt, exc, purpose)
                continue
            if want_json and not _valid_json(raw["text"]):
                last_error = LLMOutputError(f"{self._model} returned no valid JSON in JSON mode")
                self._note_retry(attempt, last_error, purpose)
                continue
            if self.cache and use_cache:
                self.cache.put(key, raw)
            if attempt > 1:
                warnings.warn(f"{self._model}: succeeded on attempt {attempt}", stacklevel=2)
            return self._response_from(raw, ttft_s=ttft)
        self.ledger.record("failure", purpose=purpose, model=self._model, error=str(last_error))
        raise LLMUnavailableError(
            f"{self._model}: failed after {self.max_attempts} attempts: {last_error}"
        )

    def _note_retry(self, attempt: int, exc: Exception, purpose: str) -> None:
        if attempt < self.max_attempts:
            self.ledger.record("retry", purpose=purpose, model=self._model, error=str(exc))
            warnings.warn(
                f"{self._model}: attempt {attempt} failed ({exc}); retrying", stacklevel=3
            )
            time.sleep(min(2**attempt, 20))

    def _call(self, req: dict[str, Any], purpose: str) -> tuple[dict[str, Any], float | None]:
        import openai

        start = self.clock.now()
        try:
            if self.stream:
                raw, ttft = self._call_streaming(req, start)
            else:
                resp = self.client.chat.completions.create(**req)
                raw, ttft = self._raw_from_completion(resp), None
        except openai.AuthenticationError as exc:
            raise LiveConfigError(f"credentials rejected by the endpoint: {exc}") from None
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as exc:
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        self.ledger.record_call(
            purpose=purpose,
            model=self._model,
            prompt_tokens=raw["usage"]["prompt_tokens"],
            cache_hit_tokens=raw["usage"]["cache_hit_tokens"],
            completion_tokens=raw["usage"]["completion_tokens"],
        )
        return raw, ttft

    @staticmethod
    def _usage(usage: Any) -> dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "cache_hit_tokens": 0, "completion_tokens": 0}
        extra = getattr(usage, "model_extra", None) or {}
        hit = extra.get("prompt_cache_hit_tokens")
        if hit is None:
            details = getattr(usage, "prompt_tokens_details", None)
            hit = getattr(details, "cached_tokens", 0) if details else 0
        return {
            "prompt_tokens": int(usage.prompt_tokens or 0),
            "cache_hit_tokens": int(hit or 0),
            "completion_tokens": int(usage.completion_tokens or 0),
        }

    def _raw_from_completion(self, resp: Any) -> dict[str, Any]:
        msg = resp.choices[0].message
        calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        return {
            "text": msg.content or "",
            "tool_calls": calls,
            "finish_reason": resp.choices[0].finish_reason,
            "model": resp.model,
            "usage": self._usage(resp.usage),
        }

    def _call_streaming(self, req: dict[str, Any], start: float) -> tuple[dict[str, Any], float]:
        stream = self.client.chat.completions.create(
            **req, stream=True, stream_options={"include_usage": True}
        )
        ttft: float | None = None
        text: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = None
        finish = None
        model = req["model"]
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if ttft is None and (delta.content or delta.tool_calls):
                ttft = self.clock.now() - start  # first content-bearing chunk
            if delta.content:
                text.append(delta.content)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            finish = choice.finish_reason or finish
            model = getattr(chunk, "model", model)
        raw = {
            "text": "".join(text),
            "tool_calls": [calls[i] for i in sorted(calls)],
            "finish_reason": finish,
            "model": model,
            "usage": self._usage(usage),
        }
        return raw, ttft if ttft is not None else self.clock.now() - start

    def _response_from(self, raw: dict[str, Any], *, ttft_s: float | None) -> LLMResponse:
        calls = tuple(
            ToolCall(c["name"], _parse_arguments(c["arguments"]), id=c.get("id"))
            for c in raw["tool_calls"]
        )
        return LLMResponse(
            text=raw["text"], model=self.model, usage=raw["usage"], tool_calls=calls, ttft_s=ttft_s
        )


def _valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(text.strip())
