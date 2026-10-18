"""Where a result came from: git commit and timestamps (METHODOLOGY P4)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_PACKAGE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Short commit of the code that produced a result; `+dirty` if uncommitted changes exist."""
    sha = _git("rev-parse", "--short=12", "HEAD")
    if not sha:
        return "unknown"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{sha}+dirty" if dirty else sha


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
