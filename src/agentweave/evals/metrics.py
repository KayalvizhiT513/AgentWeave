"""Transcript metrics that do not depend on the in-loop evaluator or the agents' self-scores."""

from __future__ import annotations

import math
import re
import statistics
import zlib
from collections import defaultdict
from typing import Protocol

import httpx

from agentweave.config import Settings
from agentweave.core.models import Conversation

Vector = list[float]


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[Vector]: ...


def _normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


class LexicalEmbedder:
    """Offline hashed bag-of-words. Measures word overlap, not meaning: use for tests and dry runs."""

    name = "lexical-hash-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[Vector]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9']+", text.lower()):
                vector[zlib.crc32(token.encode()) % self.dim] += 1.0
            vectors.append(_normalize(vector))
        return vectors


class OpenAIEmbedder:
    def __init__(self, settings: Settings, model: str = "text-embedding-3-small") -> None:
        self.name = model
        self._url = f"{settings.openai_base_url.rstrip('/')}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def embed(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            for start in range(0, len(texts), 256):
                response = await client.post(
                    self._url,
                    headers=self._headers,
                    json={"model": self.name, "input": texts[start : start + 256]},
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"Embeddings API error {response.status_code}: {response.text}")
                rows = sorted(response.json()["data"], key=lambda row: row["index"])
                vectors.extend(_normalize(row["embedding"]) for row in rows)
        return vectors


def _slope(ys: list[float]) -> float | None:
    """Least-squares slope of ys against 0..n-1; None when there are fewer than 2 points."""
    n = len(ys)
    if n < 2:
        return None
    mean_x = (n - 1) / 2
    mean_y = sum(ys) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ys)) / denom


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _distinct_2(texts: list[str]) -> float | None:
    bigrams = []
    for text in texts:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        bigrams.extend(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else None


async def transcript_metrics(conversation: Conversation, embedder: Embedder) -> dict:
    """
    Convergence and novelty of the visible transcript.

    sim_within_round: mean pairwise cosine between agents speaking in the same round (higher = agents agree).
    convergence_slope: slope of that curve over rounds (positive = the panel is converging).
    novelty_vs_history: 1 - max cosine to any earlier turn (higher = newer ground), also split by role.
    """
    turns = conversation.exchanges
    metrics: dict = {"embedder": embedder.name, "turns": len(turns)}
    if not turns:
        return metrics
    vectors = await embedder.embed([turn.content for turn in turns])

    rounds: dict[int, list[int]] = defaultdict(list)
    for index, turn in enumerate(turns):
        rounds[turn.round_number].append(index)
    curve = []
    for round_number in sorted(rounds):
        members = rounds[round_number]
        pairs = [
            _cosine(vectors[a], vectors[b])
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        if pairs:
            curve.append(sum(pairs) / len(pairs))

    novelty_by_role: dict[str, list[float]] = defaultdict(list)
    novelty_all = []
    for index in range(1, len(turns)):
        novelty = 1.0 - max(_cosine(vectors[index], vectors[earlier]) for earlier in range(index))
        novelty_all.append(novelty)
        novelty_by_role[turns[index].role.value].append(novelty)

    third = max(1, len(curve) // 3)
    metrics.update(
        sim_within_round_mean=_mean(curve),
        sim_within_round_curve=curve,
        sim_late=_mean(curve[-third:]),
        convergence_slope=_slope(curve),
        novelty_vs_history=_mean(novelty_all),
        novelty_vs_history_by_role={role: _mean(values) for role, values in novelty_by_role.items()},
        distinct_2=_distinct_2([turn.content for turn in turns]),
        # Share of turns the word cap cut mid-sentence (the orchestrator marks them with a trailing ellipsis).
        cut_rate=sum(turn.content.endswith("…") for turn in turns) / len(turns),
        mean_turn_words=statistics.mean(len(turn.content.split()) for turn in turns),
    )
    return metrics


def loop_scores(conversation: Conversation) -> dict:
    """The system's own self-reported scores. Recorded for debugging, not for drawing conclusions."""
    evaluations = conversation.evaluations
    return {
        "evaluations": len(evaluations),
        "mean_progress": _mean([e.progress_score for e in evaluations]),
        "mean_redundancy": _mean([e.redundancy_score for e in evaluations]),
        "mean_novelty": _mean([e.novelty_score for e in evaluations]),
        "agent_mean_novelty": _mean([turn.novelty_score for turn in conversation.exchanges]),
    }
