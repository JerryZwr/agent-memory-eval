"""Cost control for real LLM calls (CLAUDE.md §2.2): a persistent capped ledger, and a cache.

* `UsageLedger` is shared by every live component of a session (answering,
  judging, the agent, mem0's extraction calls). It is checked *before* each
  call and refuses once the call or cost cap is reached: the run stops with
  `BudgetExceededError`, it never continues silently. Every call (and every
  cache hit and retry) is appended to the ledger file, which is the source of
  "how many calls did this cost" in reports.
* `ResponseCache` keys on the exact request (model, messages, tools, params),
  so resuming a run or re-judging the same answers never pays twice.

Costs use the published *peak* price (conservative). Prices verified on
2026-10-19 from https://api-docs.deepseek.com/quick_start/pricing (CNY per 1M tokens).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentmemeval.errors import AgentMemEvalError
from agentmemeval.provenance import utc_now


class BudgetExceededError(AgentMemEvalError):
    """The configured call or cost cap was reached. Deliberately not an InfraError."""


class LiveConfigError(AgentMemEvalError):
    """Live calls are misconfigured (missing key, rejected credentials, unknown model)."""


@dataclass(frozen=True)
class Price:
    input_miss: float  # CNY per 1M input tokens, cache miss
    input_hit: float  # CNY per 1M input tokens, cache hit
    output: float  # CNY per 1M output tokens


PRICES_CNY: dict[str, Price] = {
    "deepseek-flash": Price(input_miss=2.0, input_hit=0.04, output=8.0),
    "deepseek-v4-pro": Price(input_miss=9.0, input_hit=0.30, output=27.0),
}


def cost_cny(
    model: str, prompt_tokens: int, cache_hit_tokens: int, completion_tokens: int
) -> float:
    price = PRICES_CNY.get(model)
    if price is None:
        raise LiveConfigError(f"no price for model {model!r}; add it to PRICES_CNY before using it")
    miss = max(0, prompt_tokens - cache_hit_tokens)
    return (
        miss * price.input_miss + cache_hit_tokens * price.input_hit
        + completion_tokens * price.output
    ) / 1_000_000  # fmt: skip


@dataclass
class UsageTotals:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    by_purpose: dict[str, dict[str, float]] = field(default_factory=dict)


class UsageLedger:
    def __init__(self, path: Path, *, max_calls: int, max_cost_cny: float):
        if max_calls < 1 or max_cost_cny <= 0:
            raise LiveConfigError("caps must be positive")
        self.path = path
        self.max_calls = max_calls
        self.max_cost_cny = max_cost_cny
        self.totals = UsageTotals()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def _apply(self, event: dict[str, Any]) -> None:
        t = self.totals
        kind = event["event"]
        if kind == "call":
            t.calls += 1
            t.prompt_tokens += event["prompt_tokens"]
            t.cache_hit_tokens += event["cache_hit_tokens"]
            t.completion_tokens += event["completion_tokens"]
            t.cost_cny += event["cost_cny"]
            bucket = t.by_purpose.setdefault(event["purpose"], {"calls": 0, "cost_cny": 0.0})
            bucket["calls"] += 1
            bucket["cost_cny"] += event["cost_cny"]
        elif kind == "cache_hit":
            t.cache_hits += 1
        elif kind == "retry":
            t.retries += 1
        elif kind == "failure":
            t.failures += 1

    def _append(self, event: dict[str, Any]) -> None:
        event = {"at": utc_now(), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._apply(event)

    def check(self) -> None:
        """Call before every paid request."""
        t = self.totals
        if t.calls >= self.max_calls:
            raise BudgetExceededError(f"call cap reached: {t.calls}/{self.max_calls} calls")
        if t.cost_cny >= self.max_cost_cny:
            raise BudgetExceededError(
                f"cost cap reached: ¥{t.cost_cny:.4f} of ¥{self.max_cost_cny:.2f}"
            )

    def record_call(self, *, purpose: str, model: str, prompt_tokens: int, cache_hit_tokens: int,
                    completion_tokens: int) -> float:  # fmt: skip
        cost = cost_cny(model, prompt_tokens, cache_hit_tokens, completion_tokens)
        self._append(
            {
                "event": "call",
                "purpose": purpose,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "completion_tokens": completion_tokens,
                "cost_cny": cost,
            }
        )
        return cost

    def record(self, event: str, **fields: Any) -> None:
        if event not in ("cache_hit", "retry", "failure"):
            raise ValueError(f"unknown ledger event {event!r}")
        self._append({"event": event, **fields})

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self.totals),
            "max_calls": self.max_calls,
            "max_cost_cny": self.max_cost_cny,
            "ledger": self.path.as_posix(),
        }


class ResponseCache:
    """Append-only JSONL cache of raw responses, keyed by the full request."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self._entries[item["key"]] = item["response"]

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, response: dict[str, Any]) -> None:
        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
