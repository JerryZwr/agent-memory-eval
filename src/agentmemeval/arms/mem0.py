"""mem0 (open source, local) as a memory arm. Live only: ingest calls an LLM.

Configuration (docs/decisions.md D-009):

* extraction LLM: mem0's own LLM object is replaced by an adapter over
  `OpenAICompatLLM`, so every extraction call goes through the shared usage
  ledger (hard caps), the response cache, and DeepSeek thinking-off. mem0's
  bundled DeepSeek provider defaults to a retired model name and cannot turn
  thinking off (checked in mem0ai 2.2.1 source);
* embeddings: local Ollama (`nomic-embed-text`); DeepSeek has no embeddings API;
* vector store: local Qdrant on disk inside the run directory, one collection
  and one `user_id` per run (METHODOLOGY 1.12: a fresh, recorded namespace).

Ingest is one `add()` per (date, group): the day's messages in that group,
with the date and group as metadata. mem0 stores *extracted* memories, not
messages, so retrieved items carry `kind="memory"`: the message-level hit
criterion does not apply to them, the date-level one does (D-006).

mem0 wraps LLM failures in its own `LLMError`; the arm unwraps them so an
infrastructure failure stays an `InfraError` and a budget stop stays a stop.
mem0 telemetry is switched off before mem0 is imported.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

os.environ.setdefault("MEM0_TELEMETRY", "False")  # before any mem0 import (module level)

from agentmemeval.arms.base import MemoryArm, RetrievedItem
from agentmemeval.data.models import Dataset, Message
from agentmemeval.errors import AgentMemEvalError, MemoryServiceError, StageError
from agentmemeval.llm.base import LLM, ChatMessage

NOMIC_DIMS = 768


class Mem0LLMAdapter:
    """What mem0 calls `self.llm`: only `generate_response` is used for text messages."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate_response(self, messages, tools=None, tool_choice="auto", response_format=None,
                          **kwargs):  # fmt: skip
        if tools:
            raise MemoryServiceError("mem0 asked for tool calls; this adapter only does text")
        chat = [ChatMessage(m["role"], m["content"]) for m in messages]
        complete: Callable[..., Any] = self.llm.complete
        if response_format is not None:
            return complete(chat, temperature=0.0, response_format=response_format).text
        return complete(chat, temperature=0.0).text


def _unwrap(exc: BaseException) -> BaseException:
    """mem0 re-raises LLM failures as its own LLMError; surface our original error."""
    cause = exc
    while cause.__cause__ is not None and not isinstance(cause, AgentMemEvalError):
        cause = cause.__cause__
    return cause


def _batches(messages: tuple[Message, ...]) -> dict[tuple[str, str], list[Message]]:
    groups: dict[tuple[str, str], list[Message]] = defaultdict(list)
    for m in messages:
        groups[(m.date, m.group)].append(m)
    return dict(sorted(groups.items()))


class Mem0Arm(MemoryArm):
    name = "mem0"
    # Offline tests replace how mem0 is built (fake embedder); None = Memory.from_config.
    default_memory_factory: ClassVar[Callable[[dict[str, Any]], Any] | None] = None

    def __init__(
        self,
        *,
        store_dir: str,
        namespace: str,
        llm: LLM | None = None,
        embedder_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        threshold: float = 0.1,
        memory_factory: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.store_dir = Path(store_dir)
        self.namespace = namespace
        self.llm = llm
        self.embedder_model = embedder_model
        self.ollama_url = ollama_url
        self.threshold = threshold
        self._memory_factory = memory_factory
        self._memory: Any = None

    def params(self) -> dict[str, Any]:
        # Behaviour only: locations and namespaces are recorded in the ingest summary.
        return {
            "implementation": "mem0ai-oss",
            "mem0_version": _mem0_version(),
            "extraction_llm": self.llm.model if self.llm else None,
            "embedder": f"ollama/{self.embedder_model}",
            "vector_store": "qdrant-local",
            "batching": "date_group",
            "threshold": self.threshold,
        }

    def _config(self) -> dict[str, Any]:
        return {
            # mem0 requires an LLM config; the object it builds is replaced by our adapter.
            "llm": {"provider": "deepseek", "config": {"model": "deepseek-flash",
                                                         "api_key": "replaced-by-adapter"}},
            "embedder": {
                "provider": "ollama",
                "config": {"model": self.embedder_model, "ollama_base_url": self.ollama_url,
                           "embedding_dims": NOMIC_DIMS},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": self.namespace,
                           "path": str(self.store_dir / "qdrant"),
                           "on_disk": True, "embedding_model_dims": NOMIC_DIMS},
            },
            # Default is ~/.mem0/history.db, shared by every run on the machine.
            "history_db_path": str(self.store_dir / "history.db"),
        }  # fmt: skip

    def _open(self) -> Any:
        if self._memory is None:
            if self.llm is None:
                raise StageError("mem0 arm needs a live LLM (run with a live configuration)")
            self.store_dir.mkdir(parents=True, exist_ok=True)
            factory = self._memory_factory or type(self).default_memory_factory
            if factory is not None:
                memory = factory(self._config())
            else:
                from mem0 import Memory

                memory = Memory.from_config(self._config())
            memory.llm = Mem0LLMAdapter(self.llm)
            self._memory = memory
        return self._memory

    def _guarded(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except AgentMemEvalError:
            raise
        except Exception as exc:
            original = _unwrap(exc)
            if isinstance(original, AgentMemEvalError):
                raise original from exc
            raise MemoryServiceError(f"mem0: {type(exc).__name__}: {exc}") from exc

    def add(self, dataset: Dataset) -> dict[str, Any]:
        memory = self._open()
        batches = _batches(dataset.messages)
        empty_batches = 0
        for (date, group), msgs in batches.items():
            payload = [
                {"role": "user", "content": f"[{m.time}] {m.speaker}: {m.text}"} for m in msgs
            ]
            meta = {"date": date, "group": group, "n_messages": len(msgs)}
            result = self._guarded(lambda p=payload, md=meta: memory.add(p, user_id=self.namespace,
                                                                         metadata=md))  # fmt: skip
            # mem0 logs (does not raise) when an extraction reply can't be parsed and stores
            # nothing: count such batches so a silent loss shows up in the ingest summary.
            if not (result or {}).get("results"):
                empty_batches += 1
        if batches and empty_batches == len(batches):
            raise MemoryServiceError("mem0 stored no memories for any batch; extraction failed")
        stored = self._guarded(
            lambda: memory.get_all(filters={"user_id": self.namespace}, top_k=100_000)
        )
        return {
            "namespace": self.namespace,
            "store_dir": self.store_dir.as_posix(),
            "n_batches": len(batches),
            "n_batches_without_memories": empty_batches,
            "n_memories": len(stored.get("results", [])),
            "message_ids_sha256": dataset.message_ids_sha256,
        }

    def attach(self, dataset: Dataset, summary: dict[str, Any]) -> None:
        if summary.get("message_ids_sha256") != dataset.message_ids_sha256:
            raise StageError("mem0: the corpus differs from the one recorded at ingest")
        if summary.get("namespace") != self.namespace:
            raise StageError("mem0: ingest used a different namespace")
        self._open()

    def close(self) -> None:
        """Release the local Qdrant lock so another instance (stage, probe) can open the store."""
        if self._memory is not None:
            client = getattr(self._memory.vector_store, "client", None)
            if client is not None:
                client.close()
            self._memory = None

    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        memory = self._open()
        found = self._guarded(
            lambda: memory.search(query, top_k=top_k, threshold=self.threshold,
                                  filters={"user_id": self.namespace})
        )  # fmt: skip
        items = []
        for r in found.get("results", [])[:top_k]:
            meta = r.get("metadata") or {}
            date = str(meta.get("date", ""))
            items.append(
                RetrievedItem(
                    message_id=f"mem0:{r['id']}",
                    date=date,
                    time=f"{date} 00:00:00" if date else "",
                    group=str(meta.get("group", "")),
                    speaker="mem0",
                    text=str(r.get("memory", "")),
                    score=round(float(r.get("score") or 0.0), 6),
                    kind="memory",
                )
            )
        return items


def _mem0_version() -> str:
    try:
        from importlib.metadata import version

        return version("mem0ai")
    except Exception:  # pragma: no cover - not installed
        return "not-installed"
