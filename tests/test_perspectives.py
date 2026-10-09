import asyncio
import json

from agentweave.core.enums import (
    AgentRole,
    DiversityGapType,
    EvaluationRecommendation,
    EventType,
    ReplacementPressure,
)
from agentweave.core.models import (
    AgentState,
    Conversation,
    DiversityAssessment,
    PerspectiveDimension,
    RuntimeConfig,
)
from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentResponse
from agentweave.services.store import ConversationStore
from tests.test_openai_provider import FakeAsyncClient, make_conversation
from tests.test_orchestrator import ScriptedProvider


DIMENSIONS = [
    ("weak ties", "Innovation emerges from spontaneous weak-tie interaction."),
    ("deep work", "Innovation emerges from uninterrupted individual deep work."),
    ("incentives", "Innovation is governed by institutional incentives."),
]


class PerspectiveProvider(ScriptedProvider):
    def __init__(self, recommendations, assessments=(), *, fail_map=False) -> None:
        super().__init__(recommendations)
        self.assessments = list(assessments)
        self.fail_map = fail_map
        self.assess_calls = 0

    async def map_perspectives(self, conversation, count):
        if self.fail_map:
            raise RuntimeError("mapping unavailable")
        return [PerspectiveDimension(name=n, theory=t) for n, t in DIMENSIONS][:count]

    async def assess_diversity(self, conversation):
        self.assess_calls += 1
        return self.assessments.pop(0) if self.assessments else None

    async def respond(self, conversation, agent) -> AgentResponse:
        response = await super().respond(conversation, agent)
        response.state_update = AgentState(
            core_thesis="ignored once a thesis exists",
            claims=[f"claim {n}" for n in range(10)],
        )
        return response


def _runtime(**overrides) -> RuntimeConfig:
    base = dict(
        agent_turn_delay_seconds=0,
        evaluation_interval=1,
        restructuring_interval=5,
        max_rounds=6,
        max_history_entries=50,
        summary_window=2,
        stagnation_threshold=5,
    )
    return RuntimeConfig(**{**base, **overrides})


def _orchestrator(provider):
    store = ConversationStore()
    return ConversationOrchestrator(store, EventBus(), provider), store


def test_seed_panel_gets_distinct_perspectives_and_unclaimed_reservoir() -> None:
    async def scenario() -> None:
        orchestrator, _ = _orchestrator(PerspectiveProvider([]))
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())

        by_role = {agent.role: agent for agent in conversation.agents}
        assert by_role[AgentRole.CHATTER].perspective == DIMENSIONS[0][1]
        assert by_role[AgentRole.CRITIC].perspective == DIMENSIONS[1][1]
        assert by_role[AgentRole.CHATTER].state.core_thesis == DIMENSIONS[0][1]
        assert by_role[AgentRole.MODERATOR].perspective is None
        assert by_role[AgentRole.EVALUATOR].perspective is None
        claimed = [d.agent_id for d in conversation.perspective_map]
        assert claimed == [by_role[AgentRole.CHATTER].id, by_role[AgentRole.CRITIC].id, None]

    asyncio.run(scenario())


def test_mapping_failure_degrades_to_personality_only() -> None:
    async def scenario() -> None:
        orchestrator, _ = _orchestrator(PerspectiveProvider([], fail_map=True))
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())

        assert conversation.perspective_map == []
        assert all(agent.perspective is None for agent in conversation.agents)

    asyncio.run(scenario())


def test_perspective_map_can_be_disabled() -> None:
    async def scenario() -> None:
        orchestrator, _ = _orchestrator(PerspectiveProvider([]))
        conversation = await orchestrator.create_conversation(
            "Remote work", None, [], _runtime(perspective_dimensions=0)
        )

        assert conversation.perspective_map == []

    asyncio.run(scenario())


def test_private_state_persists_and_is_capped() -> None:
    async def scenario() -> None:
        orchestrator, _ = _orchestrator(PerspectiveProvider([EvaluationRecommendation.CONTINUE]))
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())
        await orchestrator.step_conversation(conversation.id)

        chatter = next(a for a in conversation.agents if a.role == AgentRole.CHATTER)
        assert chatter.state.core_thesis == DIMENSIONS[0][1]
        assert chatter.state.claims == [f"claim {n}" for n in range(4, 10)]

    asyncio.run(scenario())


def test_diversity_gap_triggers_exploration_replacement() -> None:
    async def scenario() -> None:
        assessment = DiversityAssessment(
            round_number=1,
            gap=DiversityGapType.MISSING_DIMENSION,
            rationale="Nobody argued from incentives.",
            dimension_name="incentives",
            proposed_theory=DIMENSIONS[2][1],
        )
        provider = PerspectiveProvider([EvaluationRecommendation.CONTINUE], [assessment])
        orchestrator, _ = _orchestrator(provider)
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())
        queue = orchestrator.event_bus.subscribe(conversation.id)

        await orchestrator.step_conversation(conversation.id)

        assert len(conversation.replacements) == 1
        event = conversation.replacements[0]
        assert event.pressure == ReplacementPressure.EXPLORATION
        assert event.added_role == AgentRole.EXPLORER
        assert event.failure_mode == "diversity_gap:missing_dimension"
        newcomer = next(a for a in conversation.agents if a.id == event.added_agent_id)
        assert newcomer.perspective == DIMENSIONS[2][1]
        # The matching unclaimed dimension is claimed rather than duplicated.
        assert len(conversation.perspective_map) == 3
        assert conversation.perspective_map[2].agent_id == newcomer.id
        # The retired agent's theory returns to the pool.
        retired = next(a for a in conversation.agents if a.id == event.removed_agent_id)
        assert all(d.agent_id != retired.id for d in conversation.perspective_map)
        assert len(conversation.diversity_assessments) == 1

        types = []
        while not queue.empty():
            types.append((await queue.get()).type)
        assert EventType.DIVERSITY_ASSESSED.value in types
        assert types.count(EventType.AGENT_REPLACED.value) == 1

    asyncio.run(scenario())


def test_converged_models_replaces_one_of_the_converged_agents() -> None:
    async def scenario() -> None:
        provider = PerspectiveProvider([EvaluationRecommendation.CONTINUE])
        orchestrator, _ = _orchestrator(provider)
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())
        critic = next(a for a in conversation.agents if a.role == AgentRole.CRITIC)
        chatter = next(a for a in conversation.agents if a.role == AgentRole.CHATTER)
        chatter.replacement_eligibility = 0.9  # would be the default victim
        provider.assessments.append(
            DiversityAssessment(
                round_number=1,
                gap=DiversityGapType.CONVERGED_MODELS,
                rationale="Same causal model.",
                dimension_name="novel",
                proposed_theory="A theory built on a different causal explanation.",
                converged_agent_ids=[critic.id],
            )
        )
        await orchestrator.step_conversation(conversation.id)

        assert conversation.replacements[0].removed_agent_id == critic.id
        assert conversation.perspective_map[-1].name == "novel"

    asyncio.run(scenario())


def test_quality_replacement_suppresses_exploration_pressure() -> None:
    async def scenario() -> None:
        provider = PerspectiveProvider(
            [EvaluationRecommendation.REPLACE],
            [
                DiversityAssessment(
                    round_number=1, gap=DiversityGapType.MISSING_DIMENSION, proposed_theory="x"
                )
            ],
        )
        orchestrator, _ = _orchestrator(provider)
        conversation = await orchestrator.create_conversation("Remote work", None, [], _runtime())
        await orchestrator.step_conversation(conversation.id)

        assert provider.assess_calls == 0
        assert len(conversation.replacements) == 1
        event = conversation.replacements[0]
        assert event.pressure == ReplacementPressure.QUALITY
        assert event.added_role == AgentRole.CONTRARIAN
        # The newcomer picks up the unclaimed theory; it does not inherit the retired agent's.
        newcomer = next(a for a in conversation.agents if a.id == event.added_agent_id)
        assert newcomer.perspective == DIMENSIONS[2][1]

    asyncio.run(scenario())


def test_diversity_pressure_can_be_disabled() -> None:
    async def scenario() -> None:
        provider = PerspectiveProvider([EvaluationRecommendation.CONTINUE])
        orchestrator, _ = _orchestrator(provider)
        conversation = await orchestrator.create_conversation(
            "Remote work", None, [], _runtime(diversity_pressure=False)
        )
        await orchestrator.step_conversation(conversation.id)

        assert provider.assess_calls == 0

    asyncio.run(scenario())


def _settings():
    from agentweave.config import Settings

    return Settings(provider_mode="openai", openai_api_key="test-key")


def test_openai_provider_maps_perspectives(monkeypatch) -> None:
    from agentweave.services.provider import OpenAIAgentProvider

    FakeAsyncClient.payloads = [
        {"output_text": json.dumps({"dimensions": [{"name": "a", "theory": "T-a"}, {"name": " ", "theory": "x"}]})}
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    dimensions = asyncio.run(OpenAIAgentProvider(_settings()).map_perspectives(make_conversation(), 3))

    assert [(d.name, d.theory) for d in dimensions] == [("a", "T-a")]
    assert "Return exactly 3 dimensions" in FakeAsyncClient.calls[0]["json"]["input"][0]["content"]


def test_openai_provider_assesses_diversity(monkeypatch) -> None:
    from agentweave.services.provider import OpenAIAgentProvider

    FakeAsyncClient.payloads = [
        {
            "output_text": json.dumps(
                {
                    "gap": "shared_assumption",
                    "rationale": "All assume growth.",
                    "dimension_name": "degrowth",
                    "proposed_theory": "Growth is the problem.",
                    "converged_agent_ids": [],
                }
            )
        }
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    assessment = asyncio.run(OpenAIAgentProvider(_settings()).assess_diversity(make_conversation()))

    assert assessment.gap == DiversityGapType.SHARED_ASSUMPTION
    assert assessment.proposed_theory == "Growth is the problem."


def test_openai_respond_includes_perspective_state_and_parses_update(monkeypatch) -> None:
    from agentweave.services.provider import OpenAIAgentProvider

    FakeAsyncClient.payloads = [
        {
            "output_text": json.dumps(
                {
                    "content": "Weak ties drive it.",
                    "contribution_score": 0.7,
                    "novelty_score": 0.6,
                    "repetition_score": 0.1,
                    "state_update": {
                        "core_thesis": "t",
                        "assumptions": ["a"],
                        "causal_model": ["x -> y"],
                        "claims": ["c"],
                        "concessions": [],
                        "unresolved_attacks": ["u"],
                    },
                }
            )
        }
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    conversation: Conversation = make_conversation()
    agent = conversation.agents[0]
    agent.perspective = "Weak ties drive innovation."
    agent.state = AgentState(core_thesis="Weak ties drive innovation.", claims=["earlier claim"])
    response = asyncio.run(OpenAIAgentProvider(_settings()).respond(conversation, agent))

    call = FakeAsyncClient.calls[0]["json"]
    assert "Weak ties drive innovation." in call["instructions"]
    assert "Your private state" in call["input"][0]["content"]
    assert "earlier claim" in call["input"][0]["content"]
    assert response.state_update.causal_model == ["x -> y"]
    assert response.state_update.unresolved_attacks == ["u"]


def test_usage_is_recorded_by_purpose(monkeypatch) -> None:
    from agentweave.services.provider import OpenAIAgentProvider

    def payload(body: dict, tokens_in: int, tokens_out: int) -> dict:
        return {"output_text": json.dumps(body), "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out}}

    FakeAsyncClient.payloads = [
        payload({"dimensions": [{"name": "a", "theory": "T"}]}, 100, 20),
        payload(
            {
                "gap": "none",
                "rationale": "",
                "dimension_name": "",
                "proposed_theory": "",
                "converged_agent_ids": [],
            },
            300,
            40,
        ),
        payload(
            {
                "gap": "none",
                "rationale": "",
                "dimension_name": "",
                "proposed_theory": "",
                "converged_agent_ids": [],
            },
            50,
            10,
        ),
        {"output_text": json.dumps({"dimensions": []})},  # no usage block: counts as a call, zero tokens
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    provider = OpenAIAgentProvider(_settings())
    conversation = make_conversation()
    asyncio.run(provider.map_perspectives(conversation, 1))
    asyncio.run(provider.assess_diversity(conversation))
    asyncio.run(provider.assess_diversity(conversation))
    asyncio.run(provider.map_perspectives(conversation, 1))

    usage = conversation.usage
    assert usage.by_purpose["diversity"].calls == 2
    assert usage.by_purpose["diversity"].input_tokens == 350
    assert usage.by_purpose["perspective_map"].calls == 2
    assert usage.by_purpose["perspective_map"].output_tokens == 20
    assert usage.total.calls == 4
    assert usage.total.input_tokens == 450
    assert usage.model_dump()["total"]["output_tokens"] == 70
