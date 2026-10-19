"""Setting up and extending live (paid) evaluation sessions (PLAN M5).

A session directory holds one run directory per arm, all sharing one usage
ledger (hard caps) and one response cache:

    runs/<session>/
      usage.jsonl        every paid call, retry, failure and cache hit
      llm_cache.jsonl    responses keyed by the exact request
      null/  bm25/  mem0/        judged with prompt v1
      v2/null/ v2/bm25/ v2/mem0/ the *same* answers re-judged with prompt v2

Nothing here calls a model by itself: `init_session` only writes configs,
`rejudge` copies upstream stage files and reruns only the judge stage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agentmemeval.errors import StageError
from agentmemeval.pipeline.config import (
    AnswerSpec,
    ArmSpec,
    DatasetSpec,
    JudgeSpec,
    LiveSpec,
    QuestionSpec,
    RunConfig,
)
from agentmemeval.pipeline.records import STAGES
from agentmemeval.pipeline.runner import RUN_CONFIG, load_run_config
from agentmemeval.pipeline.store import stage_path, write_json_atomic

DEFAULT_DATA = Path("data/raw/evermembench/a6b210a3")
LIVE_ARMS_DEFAULT = ("null", "bm25", "mem0")


def init_session(
    out: Path,
    *,
    data: Path = DEFAULT_DATA,
    days: int = 90,
    sample: int | None = None,
    arms: tuple[str, ...] = LIVE_ARMS_DEFAULT,
    answer_model: str = "deepseek-flash",
    judge_model: str = "deepseek-v4-pro",
    judge_version: str = "v1",
    max_calls: int = 4000,
    max_cost_cny: float = 30.0,
    top_k: int = 5,
) -> list[Path]:
    """Write one run.json per arm. Refuses to overwrite an existing run configuration."""
    out = out.resolve()
    live = LiveSpec(
        ledger=(out / "usage.jsonl").as_posix(),
        cache=(out / "llm_cache.jsonl").as_posix(),
        max_calls=max_calls,
        max_cost_cny=max_cost_cny,
    )
    dirs = []
    for arm in arms:
        run_dir = out / arm
        if (run_dir / RUN_CONFIG).exists():
            raise StageError(f"{run_dir} already holds a run; use a new session directory")
        params = {"store_dir": (run_dir / "mem0_store").as_posix()} if arm == "mem0" else {}
        config = RunConfig(
            run_id=f"{out.name}-{arm}",
            dataset=DatasetSpec(
                kind="legacy_batch", path=data.resolve().as_posix(), first_days=days
            ),
            arm=ArmSpec(name=arm, params=params, top_k=top_k),
            questions=QuestionSpec(sample=sample),
            answer=AnswerSpec(kind="openai_compat", model=answer_model),
            judge=JudgeSpec(kind="llm", version=judge_version, model=judge_model),
            live=live,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_dir / RUN_CONFIG, config.model_dump(mode="json"))
        dirs.append(run_dir)
    return dirs


def rejudge(source: Path, target: Path, *, version: str, model: str | None = None) -> Path:
    """Prepare `target` to judge the *same* answers as `source` with another prompt version.

    Copies run.json (judge changed) and the ingest, search and answer stage files
    byte for byte; the judge stage is all that runs when `target` is resumed.
    """
    if (target / RUN_CONFIG).exists():
        raise StageError(f"{target} already holds a run")
    config = load_run_config(source)
    if config.judge.kind != "llm":
        raise StageError("rejudge is for runs judged by an LLM")
    for stage in STAGES[:3]:
        if not stage_path(source, stage).exists():
            raise StageError(f"{source}: {stage}.json is missing; finish the source run first")
    judge = config.judge.model_copy(
        update={"version": version, **({"model": model} if model else {})}
    )
    new = config.model_copy(update={"run_id": f"{config.run_id}-{version}", "judge": judge})
    target.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[:3]:
        shutil.copy2(stage_path(source, stage), stage_path(target, stage))
    write_json_atomic(target / RUN_CONFIG, new.model_dump(mode="json"))
    return target
