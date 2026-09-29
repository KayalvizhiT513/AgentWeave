from __future__ import annotations

import asyncio
from datetime import datetime

from agentweave.core.enums import (
    AgentRole,
    AgentStatus,
    ConversationStatus,
    EventType,
    EvaluationRecommendation,
)
from agentweave.core.models import (
    AgentProfile,
    Conversation,
    ConversationEvent,
    Exchange,
    MemorySummary,
    ReplacementEvent,
    SharedContext,
)
from agentweave.services.event_stream import EventBus
from agentweave.services.provider import AgentProvider
from agentweave.services.store import ConversationStore


ROLE_LIBRARY: dict[AgentRole, dict[str, float | str]] = {
    AgentRole.CHATTER: {
        "personality": "unconventional, expansive, and ideationally bold",
        "confidence": 0.72,
        "priority": 0.78,
        "expertise_weight": 0.62,
    },
    AgentRole.CRITIC: {
        "personality": "sharply skeptical, rigorous, and anti-consensus",
        "confidence": 0.68,
        "priority": 0.7,
        "expertise_weight": 0.7,
    },
    AgentRole.MODERATOR: {
        "personality": "anti-groupthink, scope-protective, and order-focused",
        "confidence": 0.66,
        "priority": 0.74,
        "expertise_weight": 0.58,
    },
    AgentRole.EVALUATOR: {
        "personality": "measured, objective, and novelty-exacting",
        "confidence": 0.8,
        "priority": 0.85,
        "expertise_weight": 0.76,
    },
    AgentRole.LISTENER: {
        "personality": "quiet, discerning, and conflict-sensitive synthesizer",
        "confidence": 0.61,
        "priority": 0.45,
        "expertise_weight": 0.67,
    },
    AgentRole.SYNTHESIZER: {
        "personality": "integrative, dialectical, and trade-off clarifying",
        "confidence": 0.71,
        "priority": 0.65,
        "expertise_weight": 0.71,
    },
    AgentRole.DOMAIN_EXPERT: {
        "personality": "hyper-specialized, nuance-exacting, and mechanics-focused",
        "confidence": 0.74,
        "priority": 0.67,
        "expertise_weight": 0.82,
    },
    AgentRole.PRACTICAL_ENGINEER: {
        "personality": "friction-seeking, execution-focused, and edge-case minded",
        "confidence": 0.72,
        "priority": 0.66,
        "expertise_weight": 0.78,
    },
    AgentRole.RATIONAL_ANALYST: {
        "personality": "coldly logical, premise-testing, and quantitative",
        "confidence": 0.76,
        "priority": 0.62,
        "expertise_weight": 0.74,
    },
    AgentRole.MEDIATOR: {
        "personality": "trade-off reframing, bridge-building, and pragmatic",
        "confidence": 0.66,
        "priority": 0.61,
        "expertise_weight": 0.6,
    },
    AgentRole.VISIONARY: {
        "personality": "paradigm-shifting, radical, and horizon-expanding",
        "confidence": 0.83,
        "priority": 0.73,
        "expertise_weight": 0.61,
    },
    AgentRole.CONSTRAINT_PLANNER: {
        "personality": "boundary-enforcing, cost-sensitive, and limit-focused",
        "confidence": 0.69,
        "priority": 0.72,
        "expertise_weight": 0.72,
    },
    AgentRole.CONTRARIAN: {
        "personality": "relentlessly dissenting, blind-spot exposing, and counter-narrative",
        "confidence": 0.65,
        "priority": 0.57,
        "expertise_weight": 0.66,
    },
}

REPLACEMENT_STRATEGY: dict[str, AgentRole] = {
    "too_chaotic": AgentRole.SYNTHESIZER,
    "too_repetitive": AgentRole.CONTRARIAN,
    "too_shallow": AgentRole.DOMAIN_EXPERT,
    "too_theoretical": AgentRole.PRACTICAL_ENGINEER,
    "too_emotional": AgentRole.RATIONAL_ANALYST,
    "deadlocked": AgentRole.MEDIATOR,
    "no_creativity": AgentRole.VISIONARY,
    "no_realism": AgentRole.CONSTRAINT_PLANNER,
}


def _make_agent(role: AgentRole) -> AgentProfile:
    profile = ROLE_LIBRARY[role]
    return AgentProfile(
        role=role,
        personality=str(profile["personality"]),
        confidence=float(profile["confidence"]),
        priority=float(profile["priority"]),
        expertise_weight=float(profile["expertise_weight"]),
    )


class ConversationOrchestrator:
    def __init__(
        self,
        store: ConversationStore,
        event_bus: EventBus,
        provider: AgentProvider,
    ) -> None:
        self.store = store
        self.event_bus = event_bus
        self.provider = provider
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def create_conversation(
        self,
        topic: str,
        scene: str | None,
        constraints: list[str],
        runtime,
    ) -> Conversation:
        conversation = Conversation(
            topic=topic,
            scene=scene,
            constraints=constraints,
            runtime=runtime,
            shared_context=SharedContext(goal=topic, scene=scene, constraints=constraints),
            agents=self._initial_agents(),
        )
        await self.store.create(conversation)
        await self._publish(EventType.CONVERSATION_CREATED, conversation, {"conversation": conversation})
        return conversation

    async def list_conversations(self) -> list[Conversation]:
        return await self.store.list()

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        return await self.store.get(conversation_id)

    async def start_conversation(self, conversation_id: str) -> Conversation:
        async with self._conversation_lock(conversation_id):
            conversation = await self._require_conversation(conversation_id)
            if conversation.status == ConversationStatus.RUNNING:
                return conversation
            if conversation.status == ConversationStatus.COMPLETED:
                return conversation

            conversation.status = ConversationStatus.RUNNING
            conversation.updated_at = datetime.now(conversation.updated_at.tzinfo)
            await self.store.update(conversation)
            await self._publish(EventType.CONVERSATION_STARTED, conversation, {"status": conversation.status})

            task = self._tasks.get(conversation.id)
            if task is None or task.done():
                self._tasks[conversation.id] = asyncio.create_task(
                    self._run_until_complete(conversation.id)
                )
            return conversation

    async def stop_conversation(self, conversation_id: str) -> Conversation:
        lock = self._conversation_lock(conversation_id)
        while lock.locked():
            task = self._tasks.get(conversation_id)
            if task:
                task.cancel()
            await asyncio.sleep(0)

        async with lock:
            conversation = await self._require_conversation(conversation_id)
            conversation.status = ConversationStatus.STOPPED
            conversation.updated_at = datetime.now(conversation.updated_at.tzinfo)
            task = self._tasks.pop(conversation_id, None)
            if task:
                task.cancel()
            await self.store.update(conversation)
            await self._publish(EventType.CONVERSATION_STOPPED, conversation, {"status": conversation.status})
            return conversation

    async def step_conversation(self, conversation_id: str) -> Conversation:
        async with self._conversation_lock(conversation_id):
            conversation = await self._require_conversation(conversation_id)
            if conversation.status == ConversationStatus.COMPLETED:
                return conversation
            if conversation.status == ConversationStatus.DRAFT:
                conversation.status = ConversationStatus.RUNNING

            conversation.current_round += 1
            for agent in conversation.active_agents():
                if agent.role == AgentRole.EVALUATOR:
                    continue
                response = await self.provider.respond(conversation, agent)
                agent.contribution_score = response.contribution_score
                agent.novelty_score = response.novelty_score
                agent.repetition_score = response.repetition_score
                agent.token_usage += len(response.content.split())
                agent.replacement_eligibility = min(
                    1.0,
                    (agent.repetition_score * 0.55) + ((1.0 - agent.contribution_score) * 0.45),
                )

                exchange = Exchange(
                    round_number=conversation.current_round,
                    agent_id=agent.id,
                    role=agent.role,
                    content=response.content,
                    contribution_score=response.contribution_score,
                    novelty_score=response.novelty_score,
                    repetition_score=response.repetition_score,
                )
                conversation.exchanges.append(exchange)
                conversation.shared_context.history.append(exchange.content)
                await self.store.update(conversation)
                await self._publish(
                    EventType.AGENT_RESPONDED,
                    conversation,
                    {"exchange": exchange, "agent": agent},
                )
                if conversation.runtime.agent_turn_delay_seconds:
                    await asyncio.sleep(conversation.runtime.agent_turn_delay_seconds)

            await self._summarize_if_needed(conversation)
            await self._evaluate_if_needed(conversation)
            await self._complete_if_needed(conversation)
            conversation.updated_at = datetime.now(conversation.updated_at.tzinfo)
            await self.store.update(conversation)
            await self._publish(
                EventType.ROUND_COMPLETED,
                conversation,
                {
                    "round": conversation.current_round,
                    "exchange_count": len(conversation.exchanges),
                    "status": conversation.status,
                },
            )
            return conversation

    def _initial_agents(self) -> list[AgentProfile]:
        return [
            _make_agent(AgentRole.CHATTER),
            _make_agent(AgentRole.CRITIC),
            _make_agent(AgentRole.CONTRARIAN),
            _make_agent(AgentRole.MODERATOR),
            _make_agent(AgentRole.EVALUATOR),
        ]

    async def _run_until_complete(self, conversation_id: str) -> None:
        try:
            while True:
                conversation = await self._require_conversation(conversation_id)
                if conversation.status != ConversationStatus.RUNNING:
                    return
                if conversation.current_round >= conversation.runtime.max_rounds:
                    await self._finish(conversation, "Reached max rounds without stronger convergence.")
                    return
                await self.step_conversation(conversation_id)
                conversation = await self._require_conversation(conversation_id)
                if conversation.status == ConversationStatus.COMPLETED:
                    return
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    async def _summarize_if_needed(self, conversation: Conversation) -> None:
        if len(conversation.shared_context.history) <= conversation.runtime.max_history_entries:
            return
        window = conversation.runtime.summary_window
        segment = conversation.shared_context.history[:window]
        summary = MemorySummary(
            round_number=conversation.current_round,
            summary=" ".join(segment[:3]) + f" ... ({len(segment)} exchanges compressed)",
        )
        conversation.summaries.append(summary)
        conversation.shared_context.history = [summary.summary] + conversation.shared_context.history[window:]
        await self._publish(EventType.SUMMARY_UPDATED, conversation, {"summary": summary})

    async def _evaluate_if_needed(self, conversation: Conversation) -> None:
        if conversation.current_round % conversation.runtime.evaluation_interval != 0:
            return
        evaluation = await self.provider.evaluate(conversation)
        conversation.evaluations.append(evaluation)
        conversation.stall_count = (
            conversation.stall_count + 1
            if evaluation.recommendation in {EvaluationRecommendation.REPLACE, EvaluationRecommendation.RESTRUCTURE}
            else 0
        )
        await self._publish(EventType.EVALUATION_CREATED, conversation, {"evaluation": evaluation})

        if evaluation.recommendation in {EvaluationRecommendation.REPLACE, EvaluationRecommendation.RESTRUCTURE}:
            failure_mode = self._determine_failure_mode(conversation, evaluation)
            await self._replace_weakest_agent(conversation, failure_mode)
        elif evaluation.recommendation == EvaluationRecommendation.STOP:
            await self._finish(conversation, evaluation.rationale)

    async def _replace_weakest_agent(self, conversation: Conversation, failure_mode: str) -> None:
        candidates = [
            agent
            for agent in conversation.active_agents()
            if agent.role not in {AgentRole.MODERATOR, AgentRole.EVALUATOR}
        ]
        if not candidates:
            return
        weakest = max(candidates, key=lambda agent: agent.replacement_eligibility)
        weakest.status = AgentStatus.REPLACED
        new_role = REPLACEMENT_STRATEGY[failure_mode]
        replacement = _make_agent(new_role)
        conversation.agents.append(replacement)
        event = ReplacementEvent(
            round_number=conversation.current_round,
            removed_agent_id=weakest.id,
            removed_role=weakest.role,
            added_agent_id=replacement.id,
            added_role=replacement.role,
            failure_mode=failure_mode,
            reason=f"Replaced due to {failure_mode.replace('_', ' ')} during evaluation.",
        )
        conversation.replacements.append(event)
        await self._publish(EventType.AGENT_REPLACED, conversation, {"replacement": event})

    def _determine_failure_mode(self, conversation: Conversation, evaluation) -> str:
        if evaluation.redundancy_score > 0.62:
            return "too_repetitive"
        if evaluation.novelty_score < 0.35:
            return "no_creativity"
        if evaluation.depth_score < 0.45:
            return "too_shallow"
        if conversation.current_round >= conversation.runtime.restructuring_interval:
            return "too_theoretical"
        return "too_repetitive"

    async def _complete_if_needed(self, conversation: Conversation) -> None:
        if conversation.current_round >= conversation.runtime.max_rounds:
            await self._finish(conversation, "Reached max rounds.")
            return
        if conversation.stall_count >= conversation.runtime.stagnation_threshold:
            await self._finish(conversation, "Stopped due to repeated stagnation signals.")

    async def _finish(self, conversation: Conversation, reason: str) -> None:
        if conversation.status == ConversationStatus.COMPLETED:
            return
        conversation.status = ConversationStatus.COMPLETED
        conversation.final_summary = (
            f"Conversation completed after {conversation.current_round} rounds. {reason}"
        )
        await self.store.update(conversation)
        await self._publish(
            EventType.CONVERSATION_COMPLETED,
            conversation,
            {
                "status": conversation.status,
                "final_summary": conversation.final_summary,
            },
        )

    async def _require_conversation(self, conversation_id: str) -> Conversation:
        conversation = await self.store.get(conversation_id)
        if not conversation:
            raise KeyError(conversation_id)
        return conversation

    async def _publish(self, event_type: EventType, conversation: Conversation, payload: dict) -> None:
        await self.event_bus.publish(
            ConversationEvent(
                type=event_type.value,
                conversation_id=conversation.id,
                payload=payload,
            )
        )

    def _conversation_lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock
