import asyncio

from agentweave.core.enums import (
    AgentRole,
    ConversationStatus,
    EvaluationRecommendation,
    EventType,
)
from agentweave.core.models import EvaluationSnapshot, RuntimeConfig
from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentResponse, BaseAgentProvider
from agentweave.services.store import ConversationStore


class ScriptedProvider(BaseAgentProvider):
    def __init__(
        self,
        recommendations: list[EvaluationRecommendation],
        *,
        response_delay: float = 0.0,
    ) -> None:
        self._recommendations = recommendations.copy()
        self._response_delay = response_delay
        self.respond_calls = 0
        self.evaluate_calls = 0

    async def respond(self, conversation, agent) -> AgentResponse:
        self.respond_calls += 1
        if self._response_delay:
            await asyncio.sleep(self._response_delay)
        return AgentResponse(
            content=f"{agent.role.value} contributes in round {conversation.current_round}.",
            contribution_score=0.62,
            novelty_score=0.44,
            repetition_score=0.28,
        )

    async def evaluate(self, conversation) -> EvaluationSnapshot:
        self.evaluate_calls += 1
        recommendation = (
            self._recommendations.pop(0)
            if self._recommendations
            else EvaluationRecommendation.CONTINUE
        )
        return EvaluationSnapshot(
            round_number=conversation.current_round,
            progress_score=0.55,
            novelty_score=0.41,
            coherence_score=0.72,
            redundancy_score=0.25,
            goal_alignment_score=0.77,
            depth_score=0.51,
            conflict_utility_score=0.48,
            recommendation=recommendation,
            rationale=f"Recommendation: {recommendation.value}.",
        )


class BlockingProvider(BaseAgentProvider):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.respond_calls = 0

    async def respond(self, conversation, agent) -> AgentResponse:
        self.respond_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return AgentResponse(
            content=f"{agent.role.value} unblocked.",
            contribution_score=0.6,
            novelty_score=0.4,
            repetition_score=0.2,
        )

    async def evaluate(self, conversation) -> EvaluationSnapshot:
        return EvaluationSnapshot(
            round_number=conversation.current_round,
            progress_score=0.5,
            novelty_score=0.4,
            coherence_score=0.7,
            redundancy_score=0.2,
            goal_alignment_score=0.7,
            depth_score=0.5,
            conflict_utility_score=0.5,
            recommendation=EvaluationRecommendation.CONTINUE,
            rationale="Still running.",
        )


async def _wait_for_completion(
    orchestrator: ConversationOrchestrator,
    conversation_id: str,
    *,
    timeout: float = 1.0,
):
    async def _poll():
        while True:
            conversation = await orchestrator.get_conversation(conversation_id)
            assert conversation is not None
            if conversation.status == ConversationStatus.COMPLETED:
                return conversation
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_poll(), timeout=timeout)


async def _drain_event_types(queue: asyncio.Queue, expected: int) -> list[str]:
    event_types: list[str] = []
    for _ in range(expected):
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        event_types.append(event.type)
    return event_types


def test_orchestrator_e2e_workflow() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = ScriptedProvider(
            [EvaluationRecommendation.REPLACE, EvaluationRecommendation.STOP]
        )
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(
            agent_turn_delay_seconds=0,
            evaluation_interval=1,
            restructuring_interval=5,
            max_rounds=4,
            max_history_entries=10,
            summary_window=2,
            stagnation_threshold=3,
        )

        conversation = await orchestrator.create_conversation(
            topic="Design a resilient lunar habitat",
            scene="Colony planning session",
            constraints=["must tolerate radiation", "must reuse water"],
            runtime=runtime,
        )
        conversation.shared_context.history = [f"Seed exchange {index}" for index in range(11)]
        await store.update(conversation)
        queue = event_bus.subscribe(conversation.id)

        await orchestrator.start_conversation(conversation.id)
        completed = await _wait_for_completion(orchestrator, conversation.id)

        assert completed.status == ConversationStatus.COMPLETED
        assert completed.current_round == 2
        assert len(completed.exchanges) == 8
        assert len(completed.evaluations) == 2
        assert len(completed.summaries) >= 1
        assert len(completed.replacements) == 1
        assert completed.replacements[0].added_role == AgentRole.CONTRARIAN
        assert "Recommendation: stop." in completed.final_summary

        event_types = await _drain_event_types(queue, expected=16)
        assert event_types == [
            EventType.CONVERSATION_STARTED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.SUMMARY_UPDATED.value,
            EventType.EVALUATION_CREATED.value,
            EventType.AGENT_REPLACED.value,
            EventType.ROUND_COMPLETED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.AGENT_RESPONDED.value,
            EventType.SUMMARY_UPDATED.value,
            EventType.EVALUATION_CREATED.value,
            EventType.CONVERSATION_COMPLETED.value,
        ]

    asyncio.run(scenario())


def test_step_conversation_promotes_draft_to_running() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = ScriptedProvider([EvaluationRecommendation.CONTINUE])
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(
            evaluation_interval=1,
            restructuring_interval=5,
            max_rounds=3,
            max_history_entries=10,
            summary_window=2,
            stagnation_threshold=3,
        )

        conversation = await orchestrator.create_conversation(
            topic="Test draft transition",
            scene=None,
            constraints=[],
            runtime=runtime,
        )

        updated = await orchestrator.step_conversation(conversation.id)

        assert updated.status == ConversationStatus.RUNNING
        assert updated.current_round == 1
        assert len(updated.exchanges) == 4
        assert provider.respond_calls == 4
        assert provider.evaluate_calls == 1

    asyncio.run(scenario())


def test_start_completed_conversation_is_idempotent() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = ScriptedProvider([])
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(max_rounds=1)

        conversation = await orchestrator.create_conversation(
            topic="Already done",
            scene=None,
            constraints=[],
            runtime=runtime,
        )
        conversation.status = ConversationStatus.COMPLETED
        conversation.final_summary = "Finished earlier."
        await store.update(conversation)

        result = await orchestrator.start_conversation(conversation.id)

        assert result.status == ConversationStatus.COMPLETED
        assert provider.respond_calls == 0
        assert conversation.id not in orchestrator._tasks

    asyncio.run(scenario())


def test_stop_running_conversation_cancels_background_task() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = BlockingProvider()
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(
            evaluation_interval=1,
            restructuring_interval=5,
            max_rounds=3,
            max_history_entries=10,
            summary_window=2,
            stagnation_threshold=3,
        )

        conversation = await orchestrator.create_conversation(
            topic="Cancellation path",
            scene=None,
            constraints=[],
            runtime=runtime,
        )

        await orchestrator.start_conversation(conversation.id)
        await asyncio.wait_for(provider.started.wait(), timeout=1.0)

        stopped = await orchestrator.stop_conversation(conversation.id)
        await asyncio.wait_for(provider.cancelled.wait(), timeout=1.0)

        assert stopped.status == ConversationStatus.STOPPED
        assert conversation.id not in orchestrator._tasks

        latest = await orchestrator.get_conversation(conversation.id)
        assert latest is not None
        assert latest.status == ConversationStatus.STOPPED
        assert latest.current_round == 1
        assert len(latest.exchanges) == 0

    asyncio.run(scenario())


def test_concurrent_start_creates_only_one_running_task() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = BlockingProvider()
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(
            evaluation_interval=1,
            restructuring_interval=5,
            max_rounds=3,
            max_history_entries=10,
            summary_window=2,
            stagnation_threshold=3,
        )

        conversation = await orchestrator.create_conversation(
            topic="Concurrent start",
            scene=None,
            constraints=[],
            runtime=runtime,
        )

        await asyncio.gather(
            orchestrator.start_conversation(conversation.id),
            orchestrator.start_conversation(conversation.id),
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1.0)

        assert provider.respond_calls == 1
        assert len(orchestrator._tasks) == 1

        task = orchestrator._tasks[conversation.id]
        task.cancel()
        await asyncio.wait_for(provider.cancelled.wait(), timeout=1.0)
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_concurrent_step_calls_are_serialized_per_conversation() -> None:
    async def scenario() -> None:
        store = ConversationStore()
        event_bus = EventBus()
        provider = ScriptedProvider(
            [EvaluationRecommendation.CONTINUE, EvaluationRecommendation.CONTINUE],
            response_delay=0.01,
        )
        orchestrator = ConversationOrchestrator(store, event_bus, provider)
        runtime = RuntimeConfig(
            evaluation_interval=1,
            restructuring_interval=5,
            max_rounds=4,
            max_history_entries=10,
            summary_window=2,
            stagnation_threshold=3,
        )

        conversation = await orchestrator.create_conversation(
            topic="Concurrent step",
            scene=None,
            constraints=[],
            runtime=runtime,
        )

        await asyncio.gather(
            orchestrator.step_conversation(conversation.id),
            orchestrator.step_conversation(conversation.id),
        )
        updated = await orchestrator.get_conversation(conversation.id)

        assert updated is not None
        assert updated.current_round == 2
        assert [exchange.round_number for exchange in updated.exchanges] == [1, 1, 1, 1, 2, 2, 2, 2]
        assert provider.respond_calls == 8
        assert provider.evaluate_calls == 2

    asyncio.run(scenario())
