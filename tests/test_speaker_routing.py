import asyncio

import httpx

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, SpeakerProvider


def test_roles_fall_back_to_openai_without_keys():
    settings = Settings(openai_api_key="k", claude_api_key="", perplexity_api_key=None)
    assert settings.speaker_provider(AgentRole.CHATTER) == SpeakerProvider.OPENAI
    assert settings.speaker_provider(AgentRole.EXPLORER) == SpeakerProvider.OPENAI


def test_roles_route_by_vendor_when_keys_present():
    settings = Settings(openai_api_key="k", claude_api_key="c", perplexity_api_key="p")
    assert settings.speaker_provider(AgentRole.CRITIC) == SpeakerProvider.OPENAI
    assert settings.speaker_provider(AgentRole.CHATTER) == SpeakerProvider.CLAUDE
    assert settings.speaker_provider(AgentRole.CONTRARIAN) == SpeakerProvider.PERPLEXITY
    assert settings.speaker_provider(AgentRole.EVALUATOR) == SpeakerProvider.OPENAI


def test_claude_and_perplexity_turns(monkeypatch):
    from agentweave.core.models import Conversation, SharedContext
    from agentweave.services.provider import OpenAIAgentProvider
    from agentweave.tuning import SamplingProfile

    seen: list[str] = []

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): ...
        async def post(self, url, headers=None, json=None):
            seen.append(url)
            if "anthropic" in url:
                body = {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 3, "output_tokens": 2}}
            else:
                body = {"choices": [{"message": {"content": "sourced[1] claim [2]"}}], "usage": {}}
            return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = OpenAIAgentProvider(Settings(openai_api_key="k", claude_api_key="c", perplexity_api_key="p"))
    conversation = Conversation(topic="t", shared_context=SharedContext(goal="t"))

    async def run():
        return [
            await provider._vendor_chat(vendor, SamplingProfile(), "s", "u", conversation)
            for vendor in (SpeakerProvider.CLAUDE, SpeakerProvider.PERPLEXITY)
        ]

    claude, pplx = asyncio.run(run())
    assert claude == "hi"
    assert pplx == "sourced claim"
    assert "agent_turn_claude" in conversation.usage.by_purpose
