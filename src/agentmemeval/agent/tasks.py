"""The agent's task set and how its answers are judged.

* memory tasks: the questions a memory-layer run would ask (same evidence-window
  selection), expected route `memory` by the agent's declared rule;
* web and chitchat tasks: tests/fixtures/agent_tasks.json, labelled by the same rule.

Expected routes come from the system's own declared rule (`ROUTING_RULE`), not
from the evaluator's judgement (METHODOLOGY 2.3 #3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentmemeval.agent.core import AgentTask
from agentmemeval.analysis.outcome import Outcome
from agentmemeval.errors import DatasetError
from agentmemeval.judge.base import Judge, JudgeInput
from agentmemeval.judge.mc import extract_choice
from agentmemeval.pipeline.config import DEFAULT_FIXTURE, Components

TaskKind = Literal["memory", "web", "chitchat"]
EXPECTED_ROUTE: dict[str, str] = {"memory": "memory", "web": "web", "chitchat": "chitchat"}
DEFAULT_AGENT_TASKS = DEFAULT_FIXTURE.parent / "agent_tasks.json"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: TaskKind
    task: AgentTask
    gold: str | None
    qtype: str | None = None
    window_class: str | None = None

    @property
    def expected_route(self) -> str:
        return EXPECTED_ROUTE[self.kind]


def memory_tasks(components: Components) -> list[TaskSpec]:
    classes = components.window_classes
    return [
        TaskSpec(
            task_id=q.qid,
            kind="memory",
            task=AgentTask(q.question, q.options),
            gold=q.gold,
            qtype=q.qtype,
            window_class=classes[q.qid].value,
        )
        for q in components.questions()
    ]


def extra_tasks(path: Path = DEFAULT_AGENT_TASKS) -> list[TaskSpec]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path}: {exc}") from None
    specs = []
    for item in raw:
        if item["kind"] not in ("web", "chitchat"):
            raise DatasetError(f"{item['task_id']}: kind must be web or chitchat")
        specs.append(TaskSpec(item["task_id"], item["kind"], AgentTask(item["question"]),
                              item.get("gold")))  # fmt: skip
    return specs


def all_tasks(components: Components, extra: Path = DEFAULT_AGENT_TASKS) -> list[TaskSpec]:
    tasks = memory_tasks(components) + extra_tasks(extra)
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise DatasetError("duplicate task ids")
    return tasks


def judge_answer(spec: TaskSpec, outcome: Outcome, answer: str | None, judge: Judge) -> str | None:
    """CORRECT / WRONG for delivered answers with a gold; None otherwise (chitchat has no gold)."""
    if outcome is not Outcome.DELIVERED or answer is None or spec.gold is None:
        return None
    if spec.task.options is not None:
        return "CORRECT" if extract_choice(answer).letter == spec.gold else "WRONG"
    verdict = judge.judge(
        JudgeInput(qid=spec.task_id, qtype=spec.qtype or spec.kind, question=spec.task.question,
                   gold=spec.gold, answer=answer)
    )  # fmt: skip
    return verdict.label
