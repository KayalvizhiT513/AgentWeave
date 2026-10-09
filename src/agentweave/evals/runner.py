from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agentweave.core.enums import ConversationStatus, ReplacementPressure
from agentweave.core.models import Conversation, ConversationUsage
from agentweave.evals.experiment import Experiment, Topic
from agentweave.evals.judge import Judge
from agentweave.evals.ledger import Ledger, git_state
from agentweave.evals.metrics import Embedder, loop_scores, transcript_metrics
from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentProvider
from agentweave.services.store import ConversationStore
from agentweave.tuning import config_hash

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlannedRun:
    run_id: str
    variant: str
    topic: Topic
    rep: int
    config_hash: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def plan_runs(experiment: Experiment, topics: list[Topic]) -> list[PlannedRun]:
    runs = []
    for variant in experiment.variants:
        runtime, tuning = experiment.resolve(variant)
        digest = config_hash({"runtime": runtime.model_dump(mode="json"), "tuning": tuning.model_dump(mode="json")})
        for topic in topics:
            for rep in range(1, experiment.reps + 1):
                run_id = f"{experiment.id}__{variant}__{_slug(topic.id)}__r{rep}"
                runs.append(PlannedRun(run_id, variant, topic, rep, digest))
    return runs


def estimate_calls(experiment: Experiment, variant: str) -> int:
    """Rough model-call count per run: turns, evaluations, diversity audits, map, judge, embedding."""
    runtime, _ = experiment.resolve(variant)
    evaluations = runtime.max_rounds // runtime.evaluation_interval
    audits = evaluations if runtime.diversity_pressure else 0
    return runtime.max_rounds * 3 + evaluations + audits + (1 if runtime.perspective_dimensions else 0) + 2


async def _drive(orchestrator: ConversationOrchestrator, conversation: Conversation) -> Conversation:
    for _ in range(conversation.runtime.max_rounds + 2):
        if conversation.status == ConversationStatus.COMPLETED:
            break
        conversation = await orchestrator.step_conversation(conversation.id)
    return conversation


async def _execute(
    run: PlannedRun,
    experiment: Experiment,
    *,
    provider: AgentProvider,
    judge: Judge,
    embedder: Embedder,
    models: dict[str, str],
    attempt: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    runtime, tuning = experiment.resolve(run.variant)
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "attempt": attempt,
        "experiment": experiment.id,
        "variant": run.variant,
        "topic": run.topic.id,
        "rep": run.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "config": {"runtime": runtime.model_dump(mode="json"), "tuning": tuning.model_dump(mode="json")},
        "config_hash": run.config_hash,
        "models": {**models, "embedder": embedder.name, "judge": judge.model, "judge_version": judge.version},
    }
    orchestrator = ConversationOrchestrator(ConversationStore(), EventBus(), provider)
    conversation: Conversation | None = None
    try:
        conversation = await orchestrator.create_conversation(
            run.topic.topic, run.topic.scene, run.topic.constraints, runtime, tuning
        )
        conversation = await _drive(orchestrator, conversation)
        judge_usage = ConversationUsage()
        metrics = await transcript_metrics(conversation, embedder)
        metrics.update(await judge.score(conversation, judge_usage))
        pressures = [event.pressure for event in conversation.replacements]
        speaker_calls = sum(
            bucket.calls for name, bucket in conversation.usage.by_purpose.items() if name in {"agent_turn", "agent_turn_local"}
        )
        row.update(
            status="ok",
            outcome={
                "rounds": conversation.current_round,
                "final_summary": conversation.final_summary,
                "speaker_retries": speaker_calls - len(conversation.exchanges),
                "replacements_quality": pressures.count(ReplacementPressure.QUALITY),
                "replacements_exploration": pressures.count(ReplacementPressure.EXPLORATION),
            },
            metrics=metrics,
            loop_scores=loop_scores(conversation),
            usage=conversation.usage.model_dump(mode="json"),
            judge_usage=judge_usage.model_dump(mode="json"),
        )
    except Exception as error:  # a failed run is a data point; record it rather than drop it
        logger.exception("Run %s failed", run.run_id)
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    transcript = conversation.model_dump(mode="json") if conversation is not None else None
    return row, transcript


async def run_experiment(
    experiment: Experiment,
    topics: list[Topic],
    *,
    provider: AgentProvider,
    judge: Judge,
    embedder: Embedder,
    ledger: Ledger,
    models: dict[str, str],
    concurrency: int = 1,
    rerun: bool = False,
) -> dict[str, int]:
    """Run every planned (variant, topic, rep). Skips runs already recorded OK with the same config."""
    latest = ledger.latest_by_run()
    pending = []
    skipped = 0
    for run in plan_runs(experiment, topics):
        previous = latest.get(run.run_id)
        if not rerun and previous and previous["status"] == "ok" and previous["config_hash"] == run.config_hash:
            skipped += 1
        else:
            pending.append(run)

    gate = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "error": 0, "skipped": skipped}
    write_lock = asyncio.Lock()

    async def one(run: PlannedRun) -> None:
        async with gate:
            attempt = ledger.next_attempt(run.run_id)
            row, transcript = await _execute(
                run,
                experiment,
                provider=provider,
                judge=judge,
                embedder=embedder,
                models=models,
                attempt=attempt,
            )
        async with write_lock:
            ledger.append(row, transcript)
        counts[row["status"]] += 1

    await asyncio.gather(*(one(run) for run in pending))
    return counts
