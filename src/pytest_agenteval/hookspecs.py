"""Hooks that pytest_agenteval adds to pytest (registered via `pytest_addhooks`)."""

from __future__ import annotations

import pytest


@pytest.hookspec(firstresult=True)
def pytest_agenteval_probes(config: pytest.Config, keys: list[str], phase: str):
    """Run environment probes and return ``{key: ProbeResult}``.

    ``keys`` are the probes to run (dependencies included, in evaluation
    order); ``phase`` is ``"pre"`` (before the first test that needs them) or
    ``"post"`` (at session end). The first non-None result wins. The plugin
    ships a ``trylast`` default that probes the configured evaluation stack;
    conftest files can implement this hook to substitute fake probes.

    Whatever an implementation returns, the plugin re-applies the dependency
    rule: a key below a red upstream becomes NOT_RUN, and a key the
    implementation forgot becomes NOT_RUN.
    """
