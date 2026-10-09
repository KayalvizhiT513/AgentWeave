"""
python -m agentweave.training gen   --topics evals/training/topics.json --out evals/training/pilot --limit 3 --yes
python -m agentweave.training sanity --out evals/training/pilot --yes
python -m agentweave.training build  --out evals/training/pilot
python -m agentweave.training compare --heldout H.jsonl --reply base=a.jsonl --reply tuned=b.jsonl --label NAME --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agentweave.config import get_settings
from agentweave.core.models import ConversationUsage
from agentweave.evals.experiment import Topic
from agentweave.services.provider import OpenAIAgentProvider
from agentweave.training.datagen import build_sft, run_generation, sanity_check
from agentweave.training.turn_judge import TurnJudge


def _compare(args) -> int:
    from datetime import datetime, timezone

    from agentweave.evals.ledger import EVALS_DIR, git_state
    from agentweave.training.compare import compare_systems

    heldout = [json.loads(l) for l in args.heldout.read_text().splitlines() if l.strip()]
    replies = {"hosted": [row["reference"] for row in heldout]}
    for item in args.reply:
        name, _, path = item.partition("=")
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        if len(rows) != len(heldout):
            print(f"{name}: {len(rows)} replies for {len(heldout)} prompts; regenerate it")
            return 2
        replies[name] = [r["reply"] for r in sorted(rows, key=lambda r: r["index"])]
    pairs = len(replies) * (len(replies) - 1) // 2
    print(f"{len(heldout)} prompts, systems {list(replies)}: ~{pairs * len(heldout) * 2} judge calls")
    if not args.yes:
        print("dry run only; pass --yes to spend these calls")
        return 0
    settings = get_settings()
    provider = OpenAIAgentProvider(settings)
    usage = ConversationUsage()
    judge = TurnJudge(provider, usage)
    result = asyncio.run(compare_systems(heldout, replies, judge, provider._role_brief))
    row = {
        "label": args.label, "timestamp": datetime.now(timezone.utc).isoformat(), "git": git_state(),
        "judge": {"version": judge.version, "model": judge.model}, "heldout": str(args.heldout),
        "usage": usage.model_dump(mode="json")["total"], **result,
    }
    with (EVALS_DIR / "training" / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentweave.training")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("gen")
    gen.add_argument("--topics", type=Path, required=True)
    gen.add_argument("--limit", type=int)
    gen.add_argument("--k", type=int, default=2, help="candidate replies per target-role turn")
    gen.add_argument("--rounds", type=int, default=10)
    gen.add_argument("--concurrency", type=int, default=4)
    gen.add_argument("--yes", action="store_true")
    sanity = sub.add_parser("sanity")
    sanity.add_argument("--yes", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--heldout", type=Path, required=True)
    compare.add_argument("--reply", action="append", default=[], help="NAME=path to a replies.jsonl (repeatable)")
    compare.add_argument("--label", required=True, help="short name for this comparison in the record")
    compare.add_argument("--yes", action="store_true")
    build = sub.add_parser("build")
    build.add_argument("--valid-fraction", type=float, default=0.1)
    for command in (gen, sanity, build):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "compare":
        return _compare(args)

    if args.command == "build":
        stats = build_sft(args.out / "samples.jsonl", args.out / "sft", valid_fraction=args.valid_fraction)
        print(json.dumps(stats, indent=1))
        return 0

    if args.command == "gen":
        topics = [Topic.model_validate(t) for t in json.loads(args.topics.read_text())][: args.limit]
        turns = len(topics) * (args.rounds - 1) * 3
        calls = len(topics) * (args.rounds * 3 + 6) + turns * (args.k - 1) + turns * 2 // 3
        print(f"{len(topics)} topics, k={args.k}: ~{turns} target turns (all speaking roles), ~{calls} model calls")
        if not args.yes:
            print("dry run only; pass --yes to spend these calls")
            return 0
        counts = asyncio.run(run_generation(topics, get_settings(), args.out, k=args.k, rounds=args.rounds, concurrency=args.concurrency))
        print(json.dumps(counts, indent=1))
        return 1 if counts["error"] else 0

    settings = get_settings()
    usage = ConversationUsage()
    judge = TurnJudge(OpenAIAgentProvider(settings), usage)
    if not args.yes:
        print("pass --yes to run the sanity check (about 60 short calls)")
        return 0
    result = asyncio.run(sanity_check(args.out / "samples.jsonl", judge))
    result["judge_version"] = judge.version
    (args.out / "sanity.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
