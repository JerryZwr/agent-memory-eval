"""Command-line entry point (`agentmemeval`).

Subcommands are added milestone by milestone (see PLAN.md): `demo` in M1,
`report` in M2, `agent-eval` in M4.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from agentmemeval import __version__


def _force_utf8_console() -> None:
    """Avoid UnicodeEncodeError on Windows consoles that default to GBK."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentmemeval",
        description="Evaluate AI memory systems and tool-calling agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    # Each subcommand registers its handler via `set_defaults(func=...)`.
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
