"""Smoke tests for the console entry point."""

from __future__ import annotations

import pytest

import agentmemeval
from agentmemeval.cli import main

pytestmark = pytest.mark.unit


def test_version_flag_prints_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert agentmemeval.__version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: agentmemeval" in capsys.readouterr().out
