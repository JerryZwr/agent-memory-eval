"""Reading and writing stage files.

Writes are atomic (temp file + replace) so an interrupted stage never leaves a
half-written file that a later run would mistake for a result. Reads verify
the records hash, so a hand-edited file is refused rather than trusted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agentmemeval.errors import StageError
from agentmemeval.pipeline.records import (
    RECORD_TYPES,
    SCHEMA_VERSION,
    FileMeta,
    Stage,
    StageFile,
    Upstream,
    records_sha256,
)

M = TypeVar("M", bound=BaseModel)


def stage_path(run_dir: Path, stage: Stage) -> Path:
    return run_dir / f"{stage}.json"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(tmp, path)


def write_stage(
    run_dir: Path,
    stage: Stage,
    meta: FileMeta,
    upstream: Upstream | None,
    records: list[BaseModel],
) -> StageFile:
    dumped = [r.model_dump(mode="json") for r in records]
    sf = StageFile(
        schema_version=SCHEMA_VERSION,
        stage=stage,
        meta=meta,
        upstream=upstream,
        records_sha256=records_sha256(dumped),
        records=dumped,
    )
    write_json_atomic(stage_path(run_dir, stage), sf.model_dump(mode="json"))
    return sf


def read_stage(run_dir: Path, stage: Stage) -> StageFile:
    path = stage_path(run_dir, stage)
    try:
        sf = StageFile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StageError(f"{path} does not exist") from None
    except ValidationError as exc:
        raise StageError(f"{path} is not a valid stage file: {exc}") from None
    if sf.schema_version != SCHEMA_VERSION:
        raise StageError(f"{path}: schema_version {sf.schema_version}, expected {SCHEMA_VERSION}")
    if sf.stage != stage:
        raise StageError(f"{path}: contains stage {sf.stage!r}, expected {stage!r}")
    if records_sha256(sf.records) != sf.records_sha256:
        raise StageError(f"{path}: records do not match records_sha256 (edited by hand?)")
    return sf


def parse_records(sf: StageFile, model: type[M]) -> list[M]:
    if RECORD_TYPES[sf.stage] is not model:
        raise TypeError(f"stage {sf.stage} records are {RECORD_TYPES[sf.stage].__name__}")
    try:
        return [model.model_validate(r) for r in sf.records]
    except ValidationError as exc:
        raise StageError(f"{sf.stage}: malformed record: {exc}") from None
