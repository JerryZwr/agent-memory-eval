from __future__ import annotations

from pathlib import Path

import pytest

from agentmemeval.data.loaders import load_fixture
from agentmemeval.data.models import Dataset

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MINI_CORPUS = FIXTURES / "mini_corpus"


@pytest.fixture(scope="session")
def mini_corpus_path() -> Path:
    return MINI_CORPUS


@pytest.fixture(scope="session")
def mini() -> Dataset:
    return load_fixture(MINI_CORPUS)
