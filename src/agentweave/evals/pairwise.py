"""Pairwise transcript judging: which of two debates on the same topic was the better discussion?"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
from datetime import datetime, timezone
from typing import Any

from agentweave.core.models import Conversation, ConversationUsage
from agentweave.evals.experiment import Experiment
from agentweave.evals.judge import render_transcript
from agentweave.evals.ledger import Ledger, git_state
from agentweave.services.provider import OpenAIAgentProvider

# Bump when the rubric changes: verdicts from different versions are not comparable.
PAIRWISE_VERSION = "p1"

_INSTRUCTIONS = (
    "You compare two transcripts of a multi-agent discussion on the same topic and pick the better discussion. "
    "A better discussion: (1) raises ideas a thoughtful reader would not have predicted from the topic; "
    "(2) works its points through to mechanisms or consequences; (3) keeps distinct positions alive and sharpens "
    "the points of conflict instead of drifting into agreement; (4) stays responsive, with speakers answering what "
    "was actually said. Ignore style, length and speaker labels. Give a one-sentence rationale first, then the winner."
)
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rationale": {"type": "string"}, "winner": {"type": "string", "enum": ["first", "second"]}},
    "required": ["rationale", "winner"],
}


class TranscriptPairJudge:
    version = PAIRWISE_VERSION

    def __init__(self, provider: OpenAIAgentProvider, usage: ConversationUsage) -> None:
        self._provider, self._usage = provider, usage
        self.model = provider.settings.openai_evaluator_model

    async def _once(self, first: Conversation, second: Conversation) -> str:
        data = await self._provider.structured_output(
            instructions=_INSTRUCTIONS,
            user_input="First discussion:\n" + render_transcript(first)
            + "\n\n=====\n\nSecond discussion:\n" + render_transcript(second),
            schema_name="transcript_pairwise",
            schema=_SCHEMA,
            usage=self._usage,
        )
        return str(data["winner"])

    async def prefer(self, a: Conversation, b: Conversation) -> int | None:
        """0 if `a` wins, 1 if `b` wins, None if the two orders disagree."""
        forward, backward = await asyncio.gather(self._once(a, b), self._once(b, a))
        a_forward, a_backward = forward == "first", backward == "second"
        if a_forward != a_backward:
            return None
        return 0 if a_forward else 1


def _interval(wins: int, losses: int) -> float | None:
    n = wins + losses
    return 1.96 * math.sqrt(0.25 / n) if n else None


async def compare_experiment(
    experiment: Experiment, ledger: Ledger, judge: TranscriptPairJudge, *, concurrency: int = 4
) -> dict[str, Any]:
    latest = [r for r in ledger.latest_by_run().values() if r["experiment"] == experiment.id and r["status"] == "ok"]
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in latest:
        groups.setdefault((row["topic"], row["rep"]), {})[row["variant"]] = row

    gate = asyncio.Semaphore(concurrency)

    async def one(group: dict[str, dict[str, Any]], a: str, b: str) -> int | None:
        conversations = [
            Conversation.model_validate(json.loads((ledger.directory / group[v]["transcript"]).read_text()))
            for v in (a, b)
        ]
        async with gate:
            for attempt in range(3):
                try:
                    return await judge.prefer(*conversations)
                except Exception:
                    if attempt == 2:
                        return None
                    await asyncio.sleep(1.5 * (attempt + 1))

    result: dict[str, Any] = {}
    for a, b in itertools.combinations(experiment.variants, 2):
        keys = [k for k, g in sorted(groups.items()) if a in g and b in g]
        verdicts = await asyncio.gather(*(one(groups[k], a, b) for k in keys))
        wins_a, wins_b = verdicts.count(0), verdicts.count(1)
        result[f"{a} vs {b}"] = {
            "comparisons": len(keys), "wins_a": wins_a, "wins_b": wins_b, "no_preference": verdicts.count(None),
            "a_win_rate_when_decisive": wins_a / (wins_a + wins_b) if wins_a + wins_b else None,
            "ci95": _interval(wins_a, wins_b),
            "by_topic": {f"{k[0]}#r{k[1]}": v for k, v in zip(keys, verdicts)},
        }
    return {"pairs": result, "run_ids": sorted(r["run_id"] for r in latest)}


def record(ledger: Ledger, experiment: Experiment, judge: TranscriptPairJudge, outcome: dict[str, Any], usage: ConversationUsage) -> None:
    row = {
        "experiment": experiment.id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "judge": {"version": judge.version, "model": judge.model},
        "usage": usage.model_dump(mode="json")["total"],
        **outcome,
    }
    with (ledger.directory / "pairwise.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
