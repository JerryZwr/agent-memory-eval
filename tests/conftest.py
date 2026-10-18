"""Shared test configuration.

`--live` gate: tests marked `live` make real (paid) network calls and are
skipped unless `--live` is passed explicitly. This moves into the
`pytest_agenteval` plugin in M3.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests marked `live` (real LLM / memory-service calls; needs .env, costs money)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live test: pass --live to run (needs .env, costs money)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
