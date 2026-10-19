"""Minimal `.env` loader (CLAUDE.md §2.2: keys live only in .env or CI secrets).

Sets variables that are not already set; never prints or logs values.
Supports `KEY=VALUE`, optional quotes, comments and blank lines.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> list[str]:
    """Load `path` into os.environ; return the names that were set (never the values).

    Disabled when AGENTMEMEVAL_NO_DOTENV is set (the test suite sets it, so offline
    tests can never pick up a real key and make a paid call).
    """
    if os.environ.get("AGENTMEMEVAL_NO_DOTENV") or not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
