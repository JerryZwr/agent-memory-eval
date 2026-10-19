"""Allure metadata files: environment.properties and categories.json (METHODOLOGY 3.2).

allure-pytest writes test results only; these two files are ours.

* environment.properties: what was measured (corpus, arm, models, judge
  version, commit). Values are kept ASCII so the file reads the same under
  any properties decoder.
* categories.json: failures sorted into the three conclusions of P3 (product
  defect / test defect / environment), keyed on message prefixes this
  project's tests emit.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from agentmemeval.pipeline.config import RunConfig
from agentmemeval.pipeline.records import FileMeta
from agentmemeval.provenance import git_commit

# Prefix of assertion messages that blame the system under test (see tests/eval).
PRODUCT_DEFECT_PREFIX = "[PRODUCT] "

CATEGORIES = [
    {
        "name": "环境故障 / 未测到（不算被测系统失败）",
        "matchedStatuses": ["skipped"],
        "messageRegex": r"(?s).*(not_measured: |environment probe ).*",
    },
    {
        "name": "产品缺陷：被测系统答错或未交付",
        "matchedStatuses": ["failed"],
        "messageRegex": r"(?s).*\[PRODUCT\].*",
    },
    {
        "name": "测试缺陷：用例或框架代码出错",
        "matchedStatuses": ["broken"],
    },
    {
        "name": "测试缺陷：框架自测断言失败",
        "matchedStatuses": ["failed"],
    },
]


def _ascii(value: object) -> str:
    text = str(value)
    return text.encode("ascii", "backslashreplace").decode("ascii").replace("\n", " ")


def write_allure_environment(
    alluredir: Path, eval_config: RunConfig, meta: FileMeta | None, round_reasons: list[str]
) -> Path:
    props = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "eval_arm": eval_config.arm.name,
        "eval_dataset_kind": eval_config.dataset.kind,
        "round_status": "not_measured" if round_reasons else "measured",
    }
    if meta is not None:
        props.update(
            {
                "corpus_id": meta.corpus_id,
                "arm_config_hash": meta.arm_config_hash,
                "answer_model": meta.answer_model,
                "answer_prompt_version": meta.answer_prompt_version,
                "judge_model": meta.judge_model,
                "judge_prompt_version": meta.judge_prompt_version,
                "scored_classes": meta.scored_classes,
            }
        )
    alluredir.mkdir(parents=True, exist_ok=True)
    path = alluredir / "environment.properties"
    lines = [f"{k}={_ascii(v)}" for k, v in props.items()]
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return path


def write_allure_categories(alluredir: Path) -> Path:
    alluredir.mkdir(parents=True, exist_ok=True)
    path = alluredir / "categories.json"
    path.write_text(json.dumps(CATEGORIES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
