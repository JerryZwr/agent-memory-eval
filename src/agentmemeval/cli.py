"""Command-line entry point (`agentmemeval`).

Subcommands: `demo` and `run` (M1), `report` and `baseline` (M2), `agent-eval` (M4).
`report --run <dir>` also includes agent results found in `<dir>/agent/`.
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
from agentmemeval.pipeline.config import DEFAULT_FIXTURE

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
    from agentmemeval.agent.eval import find_agent_runs, run_all, summarize, summary_lines
    from agentmemeval.pipeline.config import build_components
    from agentmemeval.report.build import AGENT_DIR, write_report

    agent_dir = out / AGENT_DIR
    if not find_agent_runs(agent_dir):
        run_all(build_components(results[-1].components.config), agent_dir)
    print("Agent layer: 6 scripted policies on the same corpus (simulated clock)")
    print("\n".join(summary_lines([summarize(r) for r in find_agent_runs(agent_dir)])))
    report = write_report(out, open_browser=args.open)
    print()
    print(f"Done in {time.perf_counter() - started:.1f}s. Stage files: {out}")
    print(f"Report: {report}")
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


def cmd_agent_eval(args: argparse.Namespace) -> int:
    from agentmemeval.agent.eval import (
        find_agent_runs,
        run_agent_eval,
        summarize,
        summary_lines,
        write_agent_run,
    )
    from agentmemeval.agent.policies import POLICIES
    from agentmemeval.pipeline.config import RunConfig, build_components, default_eval_config

    if args.eval_config:
        config = RunConfig.model_validate_json(Path(args.eval_config).read_text(encoding="utf-8"))
    else:
        config = default_eval_config()
    components = build_components(config)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("runs") / f"agent-{stamp}"
    if args.live:
        return _live_agent_eval(args, config, components, out)
    for policy in args.policy or list(POLICIES):
        run = run_agent_eval(
            policy, components, real_clock=args.real_clock, run_id=f"{out.name}-{policy}"
        )
        write_agent_run(out / f"{policy}.json", run)
    clock = "real clock" if args.real_clock else "simulated clock"
    print(f"Agent evaluation on {components.meta.corpus_id} ({clock}) -> {out}")
    print("\n".join(summary_lines([summarize(r) for r in find_agent_runs(out)])))
    return 0


def _live_agent_eval(args, config, components, out: Path) -> int:
    import hashlib

    from agentmemeval.agent.eval import run_agent_eval, summarize, summary_lines, write_agent_run
    from agentmemeval.agent.tasks import extra_tasks, memory_tasks

    if components.live is None:
        print("agent-eval --live needs --eval-config with a `live` section", file=sys.stderr)
        return 2
    memory = sorted(
        memory_tasks(components),
        key=lambda t: hashlib.sha256(f"{config.questions.seed}:{t.task_id}".encode()).hexdigest(),
    )[: args.sample]
    tasks = memory + extra_tasks()
    model = config.answer.model

    def make_llm(clock):
        return components.live.llm(model, "agent", base_url=config.answer.base_url,
                                   max_tokens=512, stream=True, clock=clock)  # fmt: skip

    run = run_agent_eval(f"live:{model}", components, tasks=tasks, make_llm=make_llm,
                         run_id=f"{out.name}-live")  # fmt: skip
    write_agent_run(out / "live.json", run)
    print(f"Live agent evaluation ({len(tasks)} tasks, real clock) on {components.meta.corpus_id}")
    print("\n".join(summary_lines([summarize(run)])))
    return 0


def cmd_query_shape(args: argparse.Namespace) -> int:
    import json

    from agentmemeval.analysis.query_shape import summary_ratios, write_query_shape
    from agentmemeval.pipeline.runner import load_run

    run = load_run(Path(args.run))
    path = write_query_shape(run)
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{run.meta.arm}: query-shape experiment ({data['short_query']}) -> {path}")
    for s in summary_ratios(data):
        print(f"  {s['shape']:<13} evidence hit {s['hits']!s:<26} empty {s['empty']!s:<24} "
              f"({s['criterion']} criterion)")  # fmt: skip
    return 0


def cmd_live_init(args: argparse.Namespace) -> int:
    from agentmemeval.live import init_session

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    dirs = init_session(
        Path(args.out), data=Path(args.data), days=args.days, sample=args.sample, arms=arms,
        max_calls=args.max_calls, max_cost_cny=args.max_cost, top_k=args.top_k,
    )  # fmt: skip
    print(f"Live session at {Path(args.out)} (caps: {args.max_calls} calls, ¥{args.max_cost}).")
    print("Nothing has been called yet. Run each arm with:")
    for d in dirs:
        print(f"  agentmemeval run --run-dir {d}")
    return 0


def cmd_rejudge(args: argparse.Namespace) -> int:
    from agentmemeval.console import format_results
    from agentmemeval.live import rejudge
    from agentmemeval.pipeline.runner import run_pipeline

    target = rejudge(Path(args.source), Path(args.target), version=args.judge_version)
    result = run_pipeline(target, on_stage=_print_stage(target.name), probe=False)
    print()
    print(format_results([result]))
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    from agentmemeval.llm.usage import UsageLedger

    ledger = UsageLedger(Path(args.ledger), max_calls=10**9, max_cost_cny=10.0**9)
    t = ledger.totals
    print(f"{args.ledger}: {t.calls} paid calls, ¥{t.cost_cny:.4f}, {t.cache_hits} cache hits, "
          f"{t.retries} retries, {t.failures} failures")  # fmt: skip
    print(f"tokens: {t.prompt_tokens} in ({t.cache_hit_tokens} cache hits), "
          f"{t.completion_tokens} out")  # fmt: skip
    for purpose, b in sorted(t.by_purpose.items()):
        print(f"  {purpose:<14} {int(b['calls']):>5} calls  ¥{b['cost_cny']:.4f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from agentmemeval.report.build import write_report

    out = Path(args.out) if args.out else None
    report = write_report(Path(args.run), out, open_browser=args.open, public=args.public)
    print(f"Report: {report}")
    return 0


def cmd_baseline_freeze(args: argparse.Namespace) -> int:
    from agentmemeval.analysis.baseline import freeze
    from agentmemeval.pipeline.runner import load_run

    path = Path(args.out) if args.out else Path("baselines") / f"{args.name}.json"
    baseline = freeze(load_run(Path(args.run)), path, name=args.name, overwrite=args.overwrite)
    print(f"Froze {len(baseline.entries)} questions from {args.run} -> {path}")
    return 0


def cmd_baseline_diff(args: argparse.Namespace) -> int:
    from agentmemeval.analysis.baseline import DiffStatus, diff_run, load_baseline
    from agentmemeval.pipeline.runner import load_run

    baseline = load_baseline(Path(args.baseline))
    result = diff_run(
        baseline, load_run(Path(args.run)), allow_incomparable=args.allow_incomparable
    )
    if result.incomparable:
        red, reset = "[31m", "[0m"
        print(f"{red}WARNING: comparing runs with different settings (--allow-incomparable):")
        for problem in result.incomparable:
            print(f"  {problem}")
        print(f"These differences are not attributable to the system alone.{reset}")
    for row in result.rows:
        if row.status in (DiffStatus.FIXED, DiffStatus.REGRESSED):
            print(f"  {row.status.value:<10} {row.qid}")
    counts = ", ".join(f"{s.value}={n}" for s, n in result.counts.items())
    print(f"vs baseline {baseline.name!r}: {counts}")
    print(f"exact McNemar on fixed vs regressed: p = {result.mcnemar_p:.3f}")
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
    demo.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    demo.set_defaults(func=cmd_demo)

    agent = sub.add_parser("agent-eval", help="run the agent over the task set with each policy")
    agent.add_argument("--out", help="output directory (default: runs/agent-<timestamp>)")
    agent.add_argument(
        "--policy",
        action="append",
        choices=["good", "stubborn_searcher", "skips_memory", "narrator", "misrouted",
                 "acknowledger"],
        help="policy to run (repeatable; default: all six)",
    )  # fmt: skip
    agent.add_argument("--eval-config", help="run configuration (run.json format)")
    agent.add_argument(
        "--real-clock", action="store_true", help="measure wall time instead of simulated latency"
    )
    agent.add_argument(
        "--live", action="store_true", help="drive the agent with the real model (paid; needs a "
        "live --eval-config)",
    )  # fmt: skip
    agent.add_argument("--sample", type=int, default=13, help="live: memory tasks to sample")
    agent.set_defaults(func=cmd_agent_eval)

    live = sub.add_parser("live-init", help="write run configs for a live (paid) session")
    live.add_argument("--out", required=True, help="session directory, e.g. runs/live-v1")
    live.add_argument("--data", default="data/raw/evermembench/a6b210a3", help="download dir")
    live.add_argument("--days", type=int, default=90, help="ingest the first N days")
    live.add_argument("--sample", type=int, help="pilot: ask only this many questions")
    live.add_argument("--arms", default="null,bm25,mem0", help="comma-separated arms")
    live.add_argument("--max-calls", type=int, default=4000, help="hard cap on paid calls")
    live.add_argument("--max-cost", type=float, default=30.0, help="hard cap in CNY")
    live.add_argument("--top-k", type=int, default=5)
    live.set_defaults(func=cmd_live_init)

    rj = sub.add_parser("rejudge", help="judge the same answers again with another prompt version")
    rj.add_argument("--from", dest="source", required=True, help="finished run directory")
    rj.add_argument("--to", dest="target", required=True, help="new run directory")
    rj.add_argument("--judge-version", required=True, choices=["v1", "v2"])
    rj.set_defaults(func=cmd_rejudge)

    qs = sub.add_parser("query-shape", help="query-shape sensitivity experiment on a finished run")
    qs.add_argument("--run", required=True, help="finished run directory")
    qs.set_defaults(func=cmd_query_shape)

    usage = sub.add_parser("usage", help="show paid calls and cost from a usage ledger")
    usage.add_argument("--ledger", required=True, help="e.g. runs/live-v1/usage.jsonl")
    usage.set_defaults(func=cmd_usage)

    report = sub.add_parser("report", help="build the HTML evaluation report from finished runs")
    report.add_argument(
        "--run", required=True, help="a run directory, or a directory of run directories"
    )
    report.add_argument("--out", help="output file (default: <run>/report.html)")
    report.add_argument("--open", action="store_true", help="open the report in a browser")
    report.add_argument(
        "--public",
        action="store_true",
        help="omit question, gold, answer and retrieved text (for publishing; D-002)",
    )
    report.set_defaults(func=cmd_report)

    baseline = sub.add_parser("baseline", help="freeze a baseline or diff a run against one")
    bsub = baseline.add_subparsers(dest="baseline_command", metavar="<action>", required=True)
    bfreeze = bsub.add_parser("freeze", help="freeze a finished run as a baseline")
    bfreeze.add_argument("--run", required=True, help="run directory")
    bfreeze.add_argument("--name", required=True, help="baseline name")
    bfreeze.add_argument("--out", help="output file (default: baselines/<name>.json)")
    bfreeze.add_argument("--overwrite", action="store_true", help="replace an existing baseline")
    bfreeze.set_defaults(func=cmd_baseline_freeze)
    bdiff = bsub.add_parser("diff", help="per-question diff of a run against a baseline")
    bdiff.add_argument("--run", required=True, help="run directory")
    bdiff.add_argument("--baseline", required=True, help="baseline file")
    bdiff.add_argument(
        "--allow-incomparable",
        action="store_true",
        help="diff even if corpus or judging settings differ (printed as a warning)",
    )
    bdiff.set_defaults(func=cmd_baseline_diff)

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
    from agentmemeval.envfile import load_env_file

    load_env_file()  # keys for live runs; values are never printed
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
