"""
python -m agentweave.evals plan   evals/experiments/001-baseline-reference.json
python -m agentweave.evals run    evals/experiments/001-baseline-reference.json --yes
python -m agentweave.evals report
python -m agentweave.evals calibrate --yes     # judge vs human labels (UKPConvArg1Strict)
python -m agentweave.evals pairwise evals/experiments/005-tuned-model-in-debate.json --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agentweave.config import get_settings
from agentweave.evals.experiment import load_experiment
from agentweave.evals.judge import OpenAIJudge
from agentweave.evals.ledger import EVALS_DIR, Ledger
from agentweave.evals.metrics import LexicalEmbedder, OpenAIEmbedder
from agentweave.evals.report import load_experiments, render_report
from agentweave.evals.runner import estimate_calls, plan_runs, run_experiment
from agentweave.services.provider import OpenAIAgentProvider


def _print_plan(path: Path) -> int:
    experiment, topics = load_experiment(path)
    runs = plan_runs(experiment, topics)
    ledger = Ledger()
    latest = ledger.latest_by_run()
    todo = [
        run
        for run in runs
        if not (
            (row := latest.get(run.run_id))
            and row["status"] == "ok"
            and row["config_hash"] == run.config_hash
        )
    ]
    calls = sum(estimate_calls(experiment, run.variant) for run in todo)
    print(f"{experiment.id}: {len(experiment.variants)} variants x {len(topics)} topics x {experiment.reps} reps")
    print(f"hypothesis: {experiment.hypothesis}")
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already recorded, {len(todo)} to run")
    print(f"estimated model calls: ~{calls}")
    return len(todo)


def _calibrate(args) -> int:
    from datetime import datetime, timezone

    from agentweave.core.models import ConversationUsage
    from agentweave.evals import calibration as cal
    from agentweave.evals.ledger import git_state

    pairs = cal.load_pairs(args.data)
    if not pairs:
        print(f"no pairs found in {args.data}; download UKPConvArg1Strict-CSV there first")
        return 2
    sample = cal.sample_pairs(pairs, args.n, args.seed)
    calibration_dir = EVALS_DIR / "calibration"
    calibration_dir.mkdir(exist_ok=True)
    sample_file = calibration_dir / f"ukpconvarg1_sample_n{args.n}_seed{args.seed}.json"
    if not sample_file.exists():
        sample_file.write_text(json.dumps([asdict(pair) for pair in sample], indent=1))
    digest = cal.sample_digest(sample)
    transcripts = cal.load_transcripts(EVALS_DIR, args.degraded_from, 3) if (EVALS_DIR / "ledger.jsonl").exists() else []
    calls = len(sample) * 4 + len(transcripts) * 3
    print(f"{len(pairs)} labeled pairs available, sampling {len(sample)} across {len({p.debate for p in sample})} debates")
    print(f"sample {sample_file.name} digest {digest}; {len(transcripts)} transcripts for the degraded check")
    print(f"~{calls} short model calls")
    if not args.yes:
        print("dry run only; pass --yes to spend these calls")
        return 0

    settings = get_settings()
    provider = OpenAIAgentProvider(settings)
    usage = ConversationUsage()
    judge = cal.OpenAIPairJudge(provider, usage)
    summary, records = asyncio.run(cal.run_pair_calibration(sample, judge, concurrency=args.concurrency))
    degraded = (
        asyncio.run(cal.run_degraded_check(transcripts, OpenAIJudge(provider), usage)) if transcripts else None
    )

    stamp = datetime.now(timezone.utc)
    calibration_id = f"cal-{stamp:%Y%m%d-%H%M%S}"
    predictions = calibration_dir / "predictions" / f"{calibration_id}.json"
    predictions.parent.mkdir(exist_ok=True)
    predictions.write_text(json.dumps(records, indent=1))
    row = {
        "schema": 1,
        "calibration_id": calibration_id,
        "timestamp": stamp.isoformat(),
        "git": git_state(),
        "dataset": {**cal.DATASET, "sample_file": sample_file.name, "sample_digest": digest, "seed": args.seed},
        "judge": {"model": judge.model, "prompt_version": cal.CAL_VERSION},
        "summary": summary,
        "degraded": degraded,
        "usage": usage.model_dump(mode="json"),
        "predictions": f"predictions/{calibration_id}.json",
    }
    with (calibration_dir / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps({"summary": summary, "usage_total": row["usage"]["total"]}, indent=1))
    print(f"recorded {calibration_id}; run: python -m agentweave.evals report")
    return 0


def _pairwise(args) -> int:
    import itertools

    from agentweave.core.models import ConversationUsage
    from agentweave.evals import pairwise as pw

    experiment, _ = load_experiment(args.experiment)
    ledger = Ledger()
    ok = [r for r in ledger.latest_by_run().values() if r["experiment"] == experiment.id and r["status"] == "ok"]
    groups = {(r["topic"], r["rep"]) for r in ok}
    pairs = len(list(itertools.combinations(experiment.variants, 2)))
    print(f"{experiment.id}: {len(ok)} finished runs, {len(groups)} topic/rep groups, {pairs} variant pairs: ~{len(groups) * pairs * 2} judge calls")
    if not args.yes:
        print("dry run only; pass --yes to spend these calls")
        return 0
    provider = OpenAIAgentProvider(get_settings())
    usage = ConversationUsage()
    judge = pw.TranscriptPairJudge(provider, usage)
    outcome = asyncio.run(pw.compare_experiment(experiment, ledger, judge, concurrency=args.concurrency))
    pw.record(ledger, experiment, judge, outcome, usage)
    for name, p in outcome["pairs"].items():
        print(name, {k: p[k] for k in ("comparisons", "wins_a", "wins_b", "no_preference", "a_win_rate_when_decisive")})
    print("now run: python -m agentweave.evals report")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentweave.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "pairwise"):
        command = sub.add_parser(name)
        command.add_argument("experiment", type=Path)
    sub.choices["pairwise"].add_argument("--yes", action="store_true", help="confirm spending model calls")
    sub.choices["pairwise"].add_argument("--concurrency", type=int, default=4)
    run = sub.choices["run"]
    run.add_argument("--yes", action="store_true", help="confirm spending model calls")
    run.add_argument("--embedder", choices=["openai", "lexical"], default="openai")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--rerun", action="store_true", help="repeat runs already recorded as ok")
    run.add_argument("--limit-topics", type=int, help="use only the first N topics (smoke test)")
    sub.add_parser("report")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--n", type=int, default=100, help="number of labeled pairs")
    cal.add_argument("--seed", type=int, default=7)
    cal.add_argument("--concurrency", type=int, default=8)
    cal.add_argument("--data", type=Path, default=EVALS_DIR / "data" / "ukpconvarg1")
    cal.add_argument("--degraded-from", default="001-baseline-reference", help="experiment whose transcripts to degrade")
    cal.add_argument("--yes", action="store_true", help="confirm spending model calls")
    args = parser.parse_args(argv)

    if args.command == "plan":
        _print_plan(args.experiment)
        return 0

    if args.command == "report":
        experiments = load_experiments(EVALS_DIR / "experiments")
        (EVALS_DIR / "EVALS.md").write_text(render_report(experiments, Ledger()))
        print(f"wrote {EVALS_DIR / 'EVALS.md'}")
        return 0

    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "pairwise":
        return _pairwise(args)

    pending = _print_plan(args.experiment)
    if not args.yes:
        print("dry run only; pass --yes to spend these calls")
        return 0
    if pending == 0 and not args.rerun:
        return 0

    settings = get_settings()
    provider = OpenAIAgentProvider(settings)
    embedder = OpenAIEmbedder(settings) if args.embedder == "openai" else LexicalEmbedder()
    experiment, topics = load_experiment(args.experiment)
    if args.limit_topics:
        topics = topics[: args.limit_topics]
    counts = asyncio.run(
        run_experiment(
            experiment,
            topics,
            provider=provider,
            judge=OpenAIJudge(provider),
            embedder=embedder,
            ledger=Ledger(),
            models={
                "agent_default": settings.openai_default_model,
                "evaluator": settings.openai_evaluator_model,
            },
            concurrency=args.concurrency,
            rerun=args.rerun,
        )
    )
    print(counts)
    print("now run: python -m agentweave.evals report")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
