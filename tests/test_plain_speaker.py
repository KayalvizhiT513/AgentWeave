import asyncio
import json

import pytest

from agentweave.config import Settings
from agentweave.core.enums import AgentRole
from agentweave.core.models import AgentProfile, AgentState, Conversation, SharedContext
from agentweave.services.provider import OpenAIAgentProvider, OpenAIProviderError
from agentweave.tuning import SamplingProfile, TuningConfig


def run(coro):
    return asyncio.run(coro)


class Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._payload


def make_client(log, chat_replies, bookkeeping_ok=True):
    """Fake httpx client: local chat endpoint plus hosted Responses API."""
    replies = list(chat_replies)

    class Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None

        async def post(self, url, headers, json):
            log.append((url, json))
            if url.endswith("/chat/completions"):
                return Resp({"choices": [{"message": {"content": replies.pop(0)}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
            if json.get("text", {}).get("format", {}).get("name") == "turn_bookkeeping":
                if not bookkeeping_ok:
                    return Resp({"error": "boom"}, 500)
                return Resp({"output_text": json_module.dumps({
                    "contribution_score": 0.7, "novelty_score": 0.6, "repetition_score": 0.2,
                    "state_update": {"core_thesis": "T", "assumptions": ["a"], "causal_model": [], "claims": ["c"], "concessions": [], "unresolved_attacks": []},
                })})
            return Resp({"output_text": "A hosted plain reply."})

    return Client


json_module = json


def setup(profile: SamplingProfile):
    agent = AgentProfile(role=AgentRole.CRITIC, personality="sharp", state=AgentState(core_thesis="T"))
    conversation = Conversation(
        topic="t", shared_context=SharedContext(goal="t"), agents=[agent], current_round=2,
        tuning=TuningConfig(default=profile),
    )
    return OpenAIAgentProvider(Settings(openai_api_key="k")), conversation, agent


def test_local_endpoint_speaker_with_hosted_bookkeeping(monkeypatch) -> None:
    log: list = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client(log, ["<think>\n\n</think>\n\nThat claim assumes the ban is permanent."]))
    provider, conversation, agent = setup(SamplingProfile(endpoint="http://127.0.0.1:8082/v1", temperature=0.7, top_p=0.95))
    response = run(provider.respond(conversation, agent))

    assert response.content == "That claim assumes the ban is permanent."  # think block stripped
    assert (response.contribution_score, response.novelty_score, response.repetition_score) == (0.7, 0.6, 0.2)
    assert response.state_update.claims == ["c"]
    chat_url, chat_body = log[0]
    assert chat_url == "http://127.0.0.1:8082/v1/chat/completions"
    assert chat_body["temperature"] == 0.7 and chat_body["top_p"] == 0.95
    assert "structured JSON" not in chat_body["messages"][0]["content"]  # plain-text prompt
    assert "state_update" not in chat_body["messages"][1]["content"]
    assert conversation.usage.by_purpose["agent_turn_local"].output_tokens == 20
    assert conversation.usage.by_purpose["bookkeeping"].calls == 1


def test_hosted_plain_speaker_uses_responses_api_without_a_schema(monkeypatch) -> None:
    log: list = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client(log, []))
    provider, conversation, agent = setup(SamplingProfile(plain_text=True))
    response = run(provider.respond(conversation, agent))
    speak = log[0][1]
    assert "text" not in speak and "structured JSON" not in speak["instructions"]
    assert response.content == "A hosted plain reply."


def test_default_profile_keeps_the_original_structured_path(monkeypatch) -> None:
    log: list = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client(log, []))
    provider, conversation, agent = setup(SamplingProfile())
    with pytest.raises(Exception):  # the fake returns plain text, which the structured path cannot parse
        run(provider.respond(conversation, agent))
    assert log[0][1]["text"]["format"]["name"] == "agent_turn"


def test_empty_local_reply_is_retried_then_fails(monkeypatch) -> None:
    log: list = []
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client(log, ["", "Second try works."]))
    provider, conversation, agent = setup(SamplingProfile(endpoint="http://x/v1"))
    assert run(provider.respond(conversation, agent)).content == "Second try works."

    log.clear()
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client(log, ["", "<think></think>", " ", "\n"]))
    with pytest.raises(OpenAIProviderError, match="empty reply 4 times"):
        run(provider.respond(conversation, agent))


def test_long_local_reply_is_cut_like_any_other(monkeypatch) -> None:
    long_reply = " ".join(["word"] * 70)
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client([], [long_reply]))
    provider, conversation, agent = setup(SamplingProfile(endpoint="http://x/v1"))
    content = run(provider.respond(conversation, agent)).content
    assert content.endswith("…") and len(content.split()) == 50


def test_bookkeeping_failure_falls_back_to_neutral_scores(monkeypatch) -> None:
    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", make_client([], ["A fine reply."], bookkeeping_ok=False))
    provider, conversation, agent = setup(SamplingProfile(endpoint="http://x/v1"))
    response = run(provider.respond(conversation, agent))
    assert response.content == "A fine reply."
    assert (response.contribution_score, response.repetition_score, response.state_update) == (0.5, 0.0, None)
