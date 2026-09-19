import asyncio
import json

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, EvaluationRecommendation
from agentweave.core.models import AgentProfile, Conversation, RuntimeConfig, SharedContext
from agentweave.services.provider import OpenAIAgentProvider


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    payloads: list[dict] = []
    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url: str, headers: dict, json: dict):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payloads.pop(0))


def make_conversation() -> Conversation:
    return Conversation(
        topic="Debate consciousness in AI",
        runtime=RuntimeConfig(),
        current_round=1,
        shared_context=SharedContext(goal="Debate consciousness in AI"),
        agents=[AgentProfile(role=AgentRole.CHATTER, personality="creative")],
        exchanges=[],
    )


def test_openai_provider_respond(monkeypatch) -> None:
    FakeAsyncClient.payloads = [
        {
            "output_text": json.dumps(
                {
                    "content": "A useful agent contribution.",
                    "contribution_score": 0.8,
                    "novelty_score": 0.7,
                    "repetition_score": 0.1,
                }
            )
        }
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    settings = Settings(
        provider_mode="openai",
        openai_api_key="test-key",
        openai_default_model="gpt-5.4-mini",
        openai_evaluator_model="gpt-5.4-mini",
    )
    provider = OpenAIAgentProvider(settings)
    conversation = make_conversation()
    response = asyncio.run(provider.respond(conversation, conversation.agents[0]))

    assert response.content == "A useful agent contribution."
    assert response.contribution_score == 0.8
    assert FakeAsyncClient.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert FakeAsyncClient.calls[0]["url"].endswith("/responses")
    instructions = FakeAsyncClient.calls[0]["json"]["instructions"]
    assert "natural human speech" in instructions
    assert "Do not label yourself with prefixes" in instructions
    assert "Resist premature consensus, groupthink" in instructions
    input_text = FakeAsyncClient.calls[0]["json"]["input"][0]["content"]
    assert "Recent dialogue:" in input_text
    assert "Guidance for exploration:" in input_text
    assert "Do NOT simply agree with or echo prior speakers." in input_text


def test_openai_provider_evaluate(monkeypatch) -> None:
    FakeAsyncClient.payloads = [
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "progress_score": 0.66,
                                    "novelty_score": 0.61,
                                    "coherence_score": 0.72,
                                    "redundancy_score": 0.21,
                                    "goal_alignment_score": 0.87,
                                    "depth_score": 0.69,
                                    "conflict_utility_score": 0.58,
                                    "recommendation": "continue",
                                    "rationale": "Still productive.",
                                }
                            ),
                        }
                    ],
                }
            ]
        }
    ]
    FakeAsyncClient.calls = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", FakeAsyncClient)

    settings = Settings(
        provider_mode="openai",
        openai_api_key="test-key",
        openai_default_model="gpt-5.4-mini",
        openai_evaluator_model="gpt-5.4-mini",
    )
    provider = OpenAIAgentProvider(settings)
    conversation = make_conversation()
    evaluation = asyncio.run(provider.evaluate(conversation))

    assert evaluation.progress_score == 0.66
    assert evaluation.recommendation == EvaluationRecommendation.CONTINUE
    instructions = FakeAsyncClient.calls[0]["json"]["instructions"]
    assert "premature convergence" in instructions
    assert "groupthink" in instructions


def test_openai_provider_role_briefs_and_divergence() -> None:
    settings = Settings(
        provider_mode="openai",
        openai_api_key="test-key",
        openai_default_model="gpt-5.4-mini",
        openai_evaluator_model="gpt-5.4-mini",
    )
    provider = OpenAIAgentProvider(settings)
    assert "out-of-the-box" in provider._role_brief(AgentRole.CHATTER)
    assert "premature agreement" in provider._role_brief(AgentRole.CRITIC)
    assert "prevailing consensus" in provider._role_brief(AgentRole.CONTRARIAN)
    assert "disruptive" in provider._role_brief(AgentRole.VISIONARY)
