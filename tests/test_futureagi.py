import asyncio

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, ConversationStatus
from agentweave.core.models import AgentProfile, Conversation, Exchange, RuntimeConfig, SharedContext
from agentweave.services.futureagi import (
    FutureAGIClient,
    build_futureagi_dataset_row,
    build_futureagi_flat_dataset_row,
)


def make_conversation() -> Conversation:
    conversation = Conversation(
        topic="Machine Learning",
        scene="Bias-Variance Tradeoff Debate",
        constraints=["Start with intuition", "End with heuristics"],
        status=ConversationStatus.COMPLETED,
        current_round=2,
        final_summary="Done.",
        runtime=RuntimeConfig(),
        shared_context=SharedContext(goal="Machine Learning", scene="Bias-Variance Tradeoff Debate"),
        agents=[AgentProfile(role=AgentRole.CHATTER, personality="creative")],
        exchanges=[],
    )
    conversation.exchanges.extend(
        [
            Exchange(
                round_number=1,
                agent_id=conversation.agents[0].id,
                role=AgentRole.CHATTER,
                content="First turn.",
                contribution_score=0.9,
                novelty_score=0.8,
                repetition_score=0.1,
            ),
            Exchange(
                round_number=2,
                agent_id=conversation.agents[0].id,
                role=AgentRole.CHATTER,
                content="Second turn.",
                contribution_score=0.85,
                novelty_score=0.7,
                repetition_score=0.1,
            ),
        ]
    )
    return conversation


def test_build_futureagi_dataset_row() -> None:
    conversation = make_conversation()
    row = build_futureagi_dataset_row(conversation, "openai")
    assert row["conversation_id"] == conversation.id
    assert row["provider_mode"] == "openai"
    assert "Chatter: First turn." in row["transcript"]
    assert "Chatter: Second turn." in row["transcript"]


def test_build_futureagi_flat_exports() -> None:
    conversation = make_conversation()
    flat = build_futureagi_flat_dataset_row(conversation, "openai")
    assert flat["constraints"].startswith("[")
    assert "Chatter: First turn." in flat["transcript"]


def test_upload_conversation_to_dataset(monkeypatch) -> None:
    settings = Settings(
        provider_mode="openai",
        openai_api_key="test-openai",
        fi_api_key="test-fi-key",
        fi_secret_key="test-fi-secret",
    )
    client = FutureAGIClient(settings)
    calls: list[tuple[str, str]] = []

    def fake_sync_upload(conversation, dataset_name):
        calls.append((conversation.id, dataset_name))
        from agentweave.services.futureagi import FutureAGIUploadResult

        return FutureAGIUploadResult(
            status="success",
            message="ok",
            raw_response={"dataset_name": dataset_name},
        )

    monkeypatch.setattr(client, "_upload_conversation_to_dataset_sync", fake_sync_upload)
    result = asyncio.run(client.upload_conversation_to_dataset(make_conversation(), "dataset-name"))
    assert result.status == "success"
    assert calls[0][1] == "dataset-name"
