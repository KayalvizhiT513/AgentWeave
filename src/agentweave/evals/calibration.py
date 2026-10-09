"""
Judge calibration against human labels (UKPConvArg1Strict, CC-BY-4.0, Habernal & Gurevych, ACL 2016).

Question answered: can an LLM judge tell the argument humans found more convincing, and does a 1-5 scale
resolve the difference or collapse it into ties? This is measured on public human labels, not on our own judge.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agentweave.core.models import Conversation, ConversationUsage
from agentweave.evals.judge import Judge
from agentweave.services.provider import OpenAIAgentProvider

CAL_VERSION = "v1"
DATASET = {
    "name": "UKPConvArg1Strict",
    "license": "CC-BY-4.0",
    "citation": "Habernal & Gurevych 2016, 'Which argument is more convincing?', ACL (P16-1150)",
    "reference_accuracy": "0.76-0.78 (paper's SVM / BiLSTM, cross-topic)",
}


@dataclass(slots=True)
class Pair:
    id: str
    debate: str
    stance: str
    winner: str  # the argument humans found more convincing
    loser: str


def load_pairs(directory: Path) -> list[Pair]:
    pairs = []
    for path in sorted(directory.glob("*.csv")):
        debate, _, stance = path.stem.partition("_")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(rows, None)
            for row in rows:
                if len(row) != 4 or row[1] not in {"a1", "a2"}:
                    continue
                first, second = row[2].strip(), row[3].strip()
                if not first or not second:
                    continue
                winner, loser = (first, second) if row[1] == "a1" else (second, first)
                pairs.append(
                    Pair(row[0], debate.replace("-", " "), stance.replace("-", " "), winner, loser)
                )
    return pairs


def sample_pairs(pairs: list[Pair], n: int, seed: int) -> list[Pair]:
    """Spread the sample across debates (round-robin), so no single topic dominates."""
    rng = random.Random(seed)
    by_debate: dict[tuple[str, str], list[Pair]] = {}
    for pair in pairs:
        by_debate.setdefault((pair.debate, pair.stance), []).append(pair)
    queues = [rng.sample(items, len(items)) for _, items in sorted(by_debate.items())]
    chosen: list[Pair] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop())
    return chosen


def sample_digest(sample: list[Pair]) -> str:
    return hashlib.sha256(json.dumps([asdict(p) for p in sample], sort_keys=True).encode()).hexdigest()[:12]


# ---- judges -------------------------------------------------------------------------------------

_PAIRWISE_INSTRUCTIONS = (
    "Two arguments take the same side of the same debate. Decide which would more readily convince a "
    "thoughtful reader who has not yet made up their mind. Judge reasoning, evidence and clarity, not length "
    "or tone. Give a one-sentence rationale first, then the winner."
)
_PAIRWISE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rationale": {"type": "string"}, "winner": {"type": "string", "enum": ["first", "second"]}},
    "required": ["rationale", "winner"],
}
_ABSOLUTE_INSTRUCTIONS = (
    "Rate how convincing one argument is for its side of a debate, as an integer 1-5. "
    "1 = a bare assertion, joke or off-topic; 2 = a weak claim with little support; 3 = a reasonable point "
    "with some support; 4 = a well-supported argument; 5 = compelling, specific and hard to dismiss. "
    "Give a one-sentence rationale first, then the score."
)
_ABSOLUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rationale": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["rationale", "score"],
}


class PairJudge(Protocol):
    model: str

    async def compare(self, debate: str, stance: str, first: str, second: str) -> str: ...
    async def rate(self, debate: str, stance: str, argument: str) -> int: ...


class OpenAIPairJudge:
    def __init__(self, provider: OpenAIAgentProvider, usage: ConversationUsage) -> None:
        self._provider, self._usage = provider, usage
        self.model = provider.settings.openai_evaluator_model

    def _framing(self, debate: str, stance: str) -> str:
        return f"Debate: {debate}\nSide being argued: {stance}\n"

    async def compare(self, debate: str, stance: str, first: str, second: str) -> str:
        data = await self._provider.structured_output(
            instructions=_PAIRWISE_INSTRUCTIONS,
            user_input=f"{self._framing(debate, stance)}\nFirst argument:\n{first}\n\nSecond argument:\n{second}",
            schema_name="cal_pairwise",
            schema=_PAIRWISE_SCHEMA,
            usage=self._usage,
        )
        return str(data["winner"])

    async def rate(self, debate: str, stance: str, argument: str) -> int:
        data = await self._provider.structured_output(
            instructions=_ABSOLUTE_INSTRUCTIONS,
            user_input=f"{self._framing(debate, stance)}\nArgument:\n{argument}",
            schema_name="cal_absolute",
            schema=_ABSOLUTE_SCHEMA,
            usage=self._usage,
        )
        return int(data["score"])


async def _retry(call: Callable[[], Awaitable[Any]], attempts: int = 3) -> Any | None:
    """Transient API failures are retried; a call that still fails returns None and is counted."""
    for attempt in range(attempts):
        try:
            return await call()
        except Exception:
            if attempt == attempts - 1:
                return None
            await asyncio.sleep(1.5 * (attempt + 1))


def _mean_ci(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"mean": mean, "ci95": half, "n": len(values)}


async def run_pair_calibration(
    sample: list[Pair], judge: PairJudge, *, concurrency: int = 8
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = asyncio.Semaphore(concurrency)

    async def limited(call: Callable[[], Awaitable[Any]]) -> Any | None:
        async with gate:
            return await _retry(call)

    async def evaluate(pair: Pair) -> dict[str, Any]:
        # Both orders: position bias cancels out of accuracy and is reported on its own.
        forward, backward, w_score, l_score = await asyncio.gather(
            limited(lambda: judge.compare(pair.debate, pair.stance, pair.winner, pair.loser)),
            limited(lambda: judge.compare(pair.debate, pair.stance, pair.loser, pair.winner)),
            limited(lambda: judge.rate(pair.debate, pair.stance, pair.winner)),
            limited(lambda: judge.rate(pair.debate, pair.stance, pair.loser)),
        )
        return {
            "id": pair.id,
            "debate": pair.debate,
            "forward": forward,  # winner shown first
            "backward": backward,  # winner shown second
            "winner_score": w_score,
            "loser_score": l_score,
        }

    records = list(await asyncio.gather(*(evaluate(pair) for pair in sample)))

    pair_scores, consistent, first_picks, votes, errors = [], 0, 0, 0, 0
    for record in records:
        picks = []
        if record["forward"] is not None:
            picks.append(record["forward"] == "first")  # correct if it picked the human winner
            first_picks += record["forward"] == "first"
            votes += 1
        if record["backward"] is not None:
            picks.append(record["backward"] == "second")
            first_picks += record["backward"] == "first"
            votes += 1
        errors += (record["forward"] is None) + (record["backward"] is None)
        if len(picks) == 2:
            consistent += picks[0] == picks[1]
        if picks:
            pair_scores.append(sum(picks) / len(picks))

    abs_scores, ties, abs_errors = [], 0, 0
    histogram = {str(score): 0 for score in range(1, 6)}
    for record in records:
        w, l = record["winner_score"], record["loser_score"]
        abs_errors += (w is None) + (l is None)
        for score in (w, l):
            if score is not None:
                histogram[str(score)] += 1
        if w is None or l is None:
            continue
        ties += w == l
        abs_scores.append(1.0 if w > l else 0.5 if w == l else 0.0)

    summary = {
        "pairs": len(sample),
        "pairwise": {
            **_mean_ci(pair_scores),
            "consistency": consistent / max(1, sum(1 for r in records if r["forward"] and r["backward"])),
            "position_bias_first": first_picks / votes if votes else None,
            "errors": errors,
        },
        "absolute_1to5": {
            **_mean_ci(abs_scores),
            "tie_rate": ties / len(abs_scores) if abs_scores else None,
            "score_histogram": histogram,
            "errors": abs_errors,
        },
    }
    return summary, records


# ---- degraded-transcript sanity check -----------------------------------------------------------


def degrade(conversation: Conversation, kind: str) -> Conversation:
    """truncated: only the first 4 turns. echo: every speaker repeats the first turn (fake agreement)."""
    clone = conversation.model_copy(deep=True)
    if kind == "truncated":
        clone.exchanges = clone.exchanges[:4]
    elif kind == "echo":
        first = clone.exchanges[0].content
        clone.exchanges = [turn.model_copy(update={"content": first}) for turn in clone.exchanges]
    elif kind != "original":
        raise ValueError(kind)
    return clone


async def run_degraded_check(conversations: list[Conversation], judge: Judge, usage: ConversationUsage) -> dict[str, Any]:
    kinds = ("original", "truncated", "echo")
    rows = []
    for index, conversation in enumerate(conversations):
        scores = {}
        for kind in kinds:
            result = await _retry(lambda c=degrade(conversation, kind): judge.score(c, usage))
            scores[kind] = (
                {k: v for k, v in result.items() if k != "judge_rationale"} if result else None
            )
        rows.append({"transcript": index, "scores": scores})
    return {"kinds": list(kinds), "rows": rows}


def load_transcripts(ledger_dir: Path, experiment: str, limit: int) -> list[Conversation]:
    rows = [json.loads(line) for line in (ledger_dir / "ledger.jsonl").read_text().splitlines() if line.strip()]
    paths = [row["transcript"] for row in rows if row["experiment"] == experiment and row["status"] == "ok"]
    return [
        Conversation.model_validate(json.loads((ledger_dir / path).read_text())) for path in paths[:limit]
    ]

