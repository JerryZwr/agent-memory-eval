from __future__ import annotations

import pytest

from agentmemeval.pipeline.config import RunConfig
from agentmemeval.pipeline.runner import LoadedRun, load_run, run_pipeline


@pytest.fixture(scope="session")
def eval_run(eval_config: RunConfig, tmp_path_factory: pytest.TempPathFactory) -> LoadedRun:
    """One pipeline run per session (the configured stack), shared by every question case."""
    run_dir = tmp_path_factory.mktemp("eval-run")
    run_pipeline(run_dir, eval_config)
    return load_run(run_dir)
