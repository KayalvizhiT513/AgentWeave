"""Compare systems' replies to the same held-out prompts: word-limit compliance and pairwise judge win rates."""

from __future__ import annotations

import asyncio
import itertools
import math
import re
from typing import Any, Callable

from agentweave.services.text import strip_reasoning_block
from agentweave.training.datagen import MAX_WORDS, _retry
from agentweave.training.turn_judge import TurnContext, TurnJudge


clean = strip_reasoning_block


def normalize(text: str) -> str:
    """What the debate would show: the orchestrator cuts a reply at the word limit."""
    words = clean(text).split()
    if len(words) > MAX_WORDS:
        return " ".join(words[:MAX_WORDS]).rstrip(".,;:") + "…"
    return " ".join(words)


def reply_stats(replies: list[str]) -> dict[str, Any]:
    replies = [clean(r) for r in replies]
    counts = [len(r.split()) for r in replies]
    return {
        "n": len(replies),
        "within_limit": sum(c <= MAX_WORDS for c in counts) / len(replies),
        "mean_words": sum(counts) / len(counts),
        "json_or_label_like": sum(bool(re.match(r"^\s*[{\[]|^\s*\w+:\s", r)) for r in replies) / len(replies),
    }


def _decisive_interval(wins: int, losses: int) -> float | None:
    n = wins + losses
    return 1.96 * math.sqrt(0.25 / n) if n else None


async def compare_systems(
    heldout: list[dict[str, Any]],
    replies: dict[str, list[str]],
    judge: TurnJudge,
    role_brief: Callable[[str], str],
    *,
    concurrency: int = 8,
) -> dict[str, Any]:
    gate = asyncio.Semaphore(concurrency)
    shown = {name: [normalize(r) for r in texts] for name, texts in replies.items()}

    async def one(row: dict[str, Any], a: str, b: str, index: int) -> int | None:
        context = TurnContext(
            topic=row["topic"], scene=row["scene"], role=row["role"],
            role_brief=role_brief(row["role"]), history=row["history_tail"],
        )
        async with gate:
            return await _retry(lambda: judge.prefer(context, shown[a][index], shown[b][index]))

    pairwise = {}
    for a, b in itertools.combinations(replies, 2):
        verdicts = await asyncio.gather(*(one(row, a, b, i) for i, row in enumerate(heldout)))
        wins_a, wins_b = verdicts.count(0), verdicts.count(1)
        pairwise[f"{a} vs {b}"] = {
            "wins_a": wins_a, "wins_b": wins_b, "no_preference": verdicts.count(None),
            "a_win_rate_when_decisive": wins_a / (wins_a + wins_b) if wins_a + wins_b else None,
            "ci95": _decisive_interval(wins_a, wins_b),
        }
    return {"prompts": len(heldout), "systems": {name: reply_stats(texts) for name, texts in replies.items()}, "pairwise": pairwise}
