"""
Builds training data for one role from AgentWeave's own debates, with no human labels.

At each turn of a target role the hosted model proposes k candidate replies. Replies that fit the word limit
are preferred: if two fit, a pairwise judge picks (both orders; if the orders disagree it is recorded as a tie
and the first is used); if only one fits it is taken; if none fit there is no training row. The selected reply
continues the debate. Every candidate and how it was selected (`selected_by`) are kept, so the data can feed
SFT now and DPO later (judge-selected pairs only).
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Awaitable, Callable

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, ConversationStatus
from agentweave.core.models import AgentProfile, Conversation, ConversationUsage, RuntimeConfig
from agentweave.evals.experiment import Topic
from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentResponse, OpenAIAgentProvider
from agentweave.services.store import ConversationStore
from agentweave.training.turn_judge import TurnContext, TurnJudge

MAX_WORDS = 50

async def _retry(call: Callable[[], Awaitable[Any]], attempts: int = 3) -> Any | None:
    for attempt in range(attempts):
        try:
            return await call()
        except Exception:
            if attempt == attempts - 1:
                return None
            await asyncio.sleep(1.5 * (attempt + 1))


def compliant(response: AgentResponse) -> bool:
    """Fits the word limit without being cut: the orchestrator marks a cut reply with a trailing ellipsis."""
    return not response.content.endswith("…") and len(response.content.split()) <= MAX_WORDS


NON_SPEAKING = {AgentRole.EVALUATOR, AgentRole.MASTER}


def response_json(response: AgentResponse) -> dict[str, Any]:
    state = response.state_update.model_dump() if response.state_update else {}
    return {
        "content": response.content,
        "contribution_score": round(response.contribution_score, 2),
        "novelty_score": round(response.novelty_score, 2),
        "repetition_score": round(response.repetition_score, 2),
        "state_update": {
            "core_thesis": state.get("core_thesis", ""),
            **{
                name: state.get(name, [])
                for name in ("assumptions", "causal_model", "claims", "concessions", "unresolved_attacks")
            },
        },
    }


class TurnDataProvider(OpenAIAgentProvider):
    """Normal provider, except the target role proposes k candidates and a judge picks the winner."""

    def __init__(
        self,
        settings: Settings,
        judge: TurnJudge,
        sink: Callable[[dict[str, Any]], None],
        *,
        target_roles: set[AgentRole] | None = None,
        k: int = 2,
    ) -> None:
        super().__init__(settings)
        self.judge, self.sink, self.k = judge, sink, k
        self.target_roles = target_roles if target_roles is not None else set(AgentRole) - NON_SPEAKING

    async def _propose(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        result = await _retry(lambda: OpenAIAgentProvider.respond(self, conversation, agent))
        if result is None:
            raise RuntimeError("agent turn failed after retries")
        return result

    async def respond(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        if agent.role not in self.target_roles or conversation.current_round < 2:
            return await self._propose(conversation, agent)

        outcomes = await asyncio.gather(
            *(self._propose(conversation, agent) for _ in range(self.k)), return_exceptions=True
        )
        candidates = [o for o in outcomes if isinstance(o, AgentResponse)]
        if not candidates:
            raise outcomes[0]  # type: ignore[misc]

        history = conversation.shared_context.history[-conversation.runtime.history_window :]
        fits = [index for index, c in enumerate(candidates) if compliant(c)]
        selected: int | None = None
        selected_by: str | None = None
        if len(fits) >= 2:
            context = TurnContext(
                topic=conversation.topic,
                scene=conversation.scene,
                role=agent.role.value,
                role_brief=self._role_brief(agent.role),
                history=history,
            )
            first, second = fits[0], fits[1]
            verdict = await _retry(
                lambda: self.judge.prefer(context, candidates[first].content, candidates[second].content)
            )
            if verdict is None:
                selected, selected_by = first, "tie"
            else:
                selected, selected_by = (first, second)[verdict], "judge"
        elif len(fits) == 1:
            selected, selected_by = fits[0], "compliance"
        self.sink(
            {
                "conversation_id": conversation.id,
                "topic": conversation.topic,
                "scene": conversation.scene,
                "round": conversation.current_round,
                "agent_id": agent.id,
                "role": agent.role.value,
                "system": self._agent_instructions(agent),
                "user": self._agent_input(conversation, agent)[0]["content"],
                # What a model that only speaks (no JSON) is shown; these become the training prompts.
                "system_plain": self._agent_instructions(agent, plain=True),
                "user_plain": self._agent_input(conversation, agent, plain=True)[0]["content"],
                "history_tail": history,
                "candidates": [
                    {"response": response_json(c), "truncated": not compliant(c)} for c in candidates
                ],
                "winner": selected,
                "selected_by": selected_by,
                "judge": {"version": self.judge.version, "model": self.judge.model},
            }
        )
        return candidates[selected if selected is not None else 0]


async def run_generation(
    topics: list[Topic],
    settings: Settings,
    out_dir: Path,
    *,
    k: int = 2,
    rounds: int = 10,
    concurrency: int = 4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path, runs_path = out_dir / "samples.jsonl", out_dir / "runs.jsonl"
    done = set()
    if runs_path.exists():
        done = {json.loads(l)["topic_id"] for l in runs_path.read_text().splitlines() if l.strip() and json.loads(l)["status"] == "ok"}
    todo = [t for t in topics if t.id not in done]

    base = OpenAIAgentProvider(settings)
    usage = ConversationUsage()
    judge = TurnJudge(base, usage)
    topic_by_text = {t.topic: t.id for t in topics}

    def sink(sample: dict[str, Any]) -> None:
        sample["topic_id"] = topic_by_text.get(sample["topic"], sample["topic"])
        with samples_path.open("a") as handle:
            handle.write(json.dumps(sample) + "\n")

    provider = TurnDataProvider(settings, judge, sink, k=k)
    gate = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "error": 0, "skipped": len(done)}

    async def one(topic: Topic) -> None:
        async with gate:
            orchestrator = ConversationOrchestrator(ConversationStore(), EventBus(), provider)
            row: dict[str, Any] = {"topic_id": topic.id}
            try:
                runtime = RuntimeConfig(max_rounds=rounds, agent_turn_delay_seconds=0)
                conversation = await orchestrator.create_conversation(
                    topic.topic, topic.scene, topic.constraints, runtime
                )
                for _ in range(rounds + 2):
                    if conversation.status == ConversationStatus.COMPLETED:
                        break
                    conversation = await orchestrator.step_conversation(conversation.id)
                row.update(status="ok", rounds=conversation.current_round, usage=conversation.usage.model_dump(mode="json"))
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
            with runs_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            counts[row["status"]] += 1

    await asyncio.gather(*(one(t) for t in todo))
    counts["judge_usage"] = usage.model_dump(mode="json")["total"]
    return counts


def _valid_target(response: dict[str, Any], truncated: bool) -> bool:
    content = response.get("content", "")
    return bool(content) and not truncated and len(content.split()) <= MAX_WORDS


def build_sft(
    samples_path: Path, out_dir: Path, *, valid_fraction: float = 0.1, seed: int = 11, judge_only: bool = False
) -> dict[str, Any]:
    """
    Selected replies become chat-format SFT rows (plain-text prompt -> spoken reply); the split is by topic so no debate leaks across train and valid.
    Preference pairs (pairs.jsonl) come only from consistent judge decisions.
    """
    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    if samples and "system_plain" not in samples[0]:
        raise ValueError("samples predate plain-text prompts (no system_plain); regenerate them with `gen`")
    topic_ids = sorted({s["topic_id"] for s in samples})
    random.Random(seed).shuffle(topic_ids)
    n_valid = max(1, round(len(topic_ids) * valid_fraction)) if len(topic_ids) >= 5 else 0
    valid_topics = set(topic_ids[:n_valid])

    stats: dict[str, Any] = {
        "samples": len(samples), "no_usable_reply": 0, "skipped_by_filter": 0,
        "train": 0, "valid": 0, "pairs": 0, "by_selection": {}, "by_role": {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: (out_dir / f"{name}.jsonl").open("w") for name in ("train", "valid", "pairs", "heldout")}
    try:
        for sample in samples:
            winner, how = sample["winner"], sample.get("selected_by")
            if winner is None:
                stats["no_usable_reply"] += 1
                continue
            chosen = sample["candidates"][winner]
            if (judge_only and how != "judge") or not _valid_target(chosen["response"], chosen["truncated"]):
                stats["skipped_by_filter"] += 1
                continue
            split = "valid" if sample["topic_id"] in valid_topics else "train"
            system, user = sample["system_plain"], sample["user_plain"]
            row = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": chosen["response"]["content"]},
                ]
            }
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            if split == "valid":
                # Everything a judge needs to compare other models' replies to this same prompt.
                handles["heldout"].write(
                    json.dumps(
                        {
                            "topic_id": sample["topic_id"], "topic": sample["topic"], "scene": sample["scene"],
                            "role": sample["role"], "history_tail": sample["history_tail"],
                            "system": system, "user": user, "reference": chosen["response"]["content"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stats[split] += 1
            stats["by_selection"][how] = stats["by_selection"].get(how, 0) + 1
            stats["by_role"][sample["role"]] = stats["by_role"].get(sample["role"], 0) + 1
            if how == "judge":
                rejected = sample["candidates"][1 - winner]
                handles["pairs"].write(
                    json.dumps(
                        {
                            "system": system, "user": user, "chosen": chosen["response"]["content"],
                            "rejected": rejected["response"]["content"], "topic_id": sample["topic_id"],
                            "role": sample["role"], "judge": sample["judge"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["pairs"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    stats["valid_topics"] = sorted(valid_topics)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


async def sanity_check(samples_path: Path, judge: TurnJudge, limit: int = 30) -> dict[str, Any]:
    """The judge should clearly prefer a chosen reply over a lazy echo of the previous speaker."""
    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    rows = [s for s in samples if s.get("selected_by") == "judge" and s["history_tail"]][:limit]
    verdicts = []
    for sample in rows:
        context = TurnContext(
            topic=sample["topic"], scene=sample["scene"], role=sample["role"],
            role_brief="challenge the discussion", history=sample["history_tail"],
        )
        chosen = sample["candidates"][sample["winner"]]["response"]["content"]
        verdict = await _retry(lambda: judge.prefer(context, chosen, sample["history_tail"][-1]))
        verdicts.append(verdict)
    wins = sum(v == 0 for v in verdicts)
    return {
        "checked": len(verdicts),
        "prefers_real_reply": wins,
        "prefers_echo": sum(v == 1 for v in verdicts),
        "inconsistent_or_failed": sum(v is None for v in verdicts),
        "rate": wins / len(verdicts) if verdicts else None,
    }
