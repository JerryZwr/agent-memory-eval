"""Shared test configuration.

`--live`, `requires_env` and the other evaluation options come from the
`pytest_agenteval` plugin (loaded through its pytest11 entry point).

This file only labels tests for Allure (METHODOLOGY 3.2): epic = which layer,
feature = which capability. Labels set with decorators in a test file win
over these defaults.
"""

from __future__ import annotations

import pytest

EPIC_BY_DIR = {"unit": "框架自测", "plugin": "框架自测", "eval": "记忆层", "agent": "Agent 层"}

FEATURE_BY_MODULE = {
    "test_check_denylist": "工程：敏感词钩子",
    "test_download_data": "工程：数据下载",
    "test_cli": "工程：命令行",
    "test_demo": "工程：命令行",
    "test_loaders": "数据：加载与证据恢复",
    "test_models": "数据：加载与证据恢复",
    "test_arms": "被测对象：null / bm25",
    "test_reader": "答题：ScriptedLLM",
    "test_judging": "判分：选择题提取与评分员",
    "test_outcome_and_scoring": "结局分类与计分",
    "test_pipeline": "流水线：落盘、续跑与拒绝",
    "test_windows": "分析：证据窗口",
    "test_stats": "分析：统计方法",
    "test_compare": "分析：配对比较",
    "test_breakdown_and_baseline": "分析：拆分、2×2 与基线",
    "test_report": "评测报告",
    "test_probes": "环境探针",
}
CRITICAL_MODULES = {"test_outcome_and_scoring", "test_pipeline", "test_probes"}


def _label(value: str, label_type: str) -> pytest.MarkDecorator:
    # allure.epic()/feature() produce exactly this marker; using it directly keeps
    # labelling working even when allure-pytest is not active.
    return pytest.mark.allure_label(value, label_type=label_type)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = item.path.parts
        layer = next((d for d in ("unit", "plugin", "eval", "agent") if d in parts), None)
        if layer is None:
            continue
        item.add_marker(_label(EPIC_BY_DIR[layer], "epic"))
        module = item.path.stem
        feature = "pytest 插件" if layer == "plugin" else FEATURE_BY_MODULE.get(module)
        if feature:
            item.add_marker(_label(feature, "feature"))
        if module in CRITICAL_MODULES:
            item.add_marker(_label("critical", "severity"))
