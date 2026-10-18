"""Command-line entry point (`agentmemeval`).

Subcommands are added milestone by milestone (see PLAN.md): `demo` and `run`
in M1, `report` in M2, `agent-eval` in M4.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from agentmemeval import __version__
from agentmemeval.errors import AgentMemEvalError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini_corpus"
DEMO_ARMS = ("null", "bm25")


def _force_utf8_console() -> None:
    """Avoid UnicodeEncodeError on Windows consoles that default to GBK."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _print_stage(prefix: str):
    def report(stage: str, action: object) -> None:
        print(f"  [{prefix}] {stage:<7} {action}")

    return report


def cmd_demo(args: argparse.Namespace) -> int:
    from agentmemeval.console import format_results
    from agentmemeval.pipeline.config import ArmSpec, DatasetSpec, RunConfig
    from agentmemeval.pipeline.runner import run_pipeline

    fixture = Path(args.fixture)
    if not (fixture / "qa.json").is_file():
        print(f"demo: fixture not found at {fixture}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("runs") / f"demo-{stamp}"
    started = time.perf_counter()
    print(f"Offline demo: synthetic corpus, scripted reader, scripted judge -> {out}")
    results = []
    for arm in DEMO_ARMS:
        config = RunConfig(
            run_id=f"{out.name}-{arm}",
            dataset=DatasetSpec(kind="fixture", path=fixture.resolve().as_posix()),
            arm=ArmSpec(name=arm),
        )
        results.append(run_pipeline(out / arm, config, on_stage=_print_stage(arm)))
    print()
    print(format_results(results))
    print()
    print(f"Done in {time.perf_counter() - started:.1f}s. Stage files: {out}")
    print(f"Delete any stage file and resume with: agentmemeval run --run-dir {out / 'bm25'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from agentmemeval.console import format_results
    from agentmemeval.pipeline.runner import run_pipeline

    run_dir = Path(args.run_dir)
    print(f"Resuming {run_dir}")
    result = run_pipeline(run_dir, force=args.force_stage, on_stage=_print_stage(run_dir.name))
    print()
    print(format_results([result]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentmemeval",
        description="Evaluate AI memory systems and tool-calling agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    demo = sub.add_parser(
        "demo", help="offline demo: synthetic corpus, null vs bm25, no keys or network"
    )
    demo.add_argument("--out", help="run directory (default: runs/demo-<timestamp>)")
    demo.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="synthetic corpus directory")
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="run or resume the pipeline in an existing run directory")
    run.add_argument("--run-dir", required=True, help="directory containing run.json")
    run.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=["ingest", "search", "answer", "judge"],
        help="re-run this stage even if its file is valid (repeatable)",
    )
    run.set_defaults(func=cmd_run)
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
    try:
        return int(handler(args))
    except AgentMemEvalError as exc:
        # Deliberate refusals (P2) are reported plainly; anything else is a bug and
        # keeps its traceback.
        print(f"agentmemeval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
