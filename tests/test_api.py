from fastapi.testclient import TestClient

from agentweave.app import app
from agentweave.services.provider import AgentResponse, OpenAIAgentProvider


client = TestClient(app)


async def _test_response(_self, conversation, agent) -> AgentResponse:
    return AgentResponse(
        content=f"{agent.role.value} test contribution for round {conversation.current_round}.",
        contribution_score=0.7,
        novelty_score=0.6,
        repetition_score=0.2,
    )


async def _no_perspectives(_self, conversation, count):
    return []


def test_create_and_step_conversation(monkeypatch) -> None:
    monkeypatch.setattr(OpenAIAgentProvider, "respond", _test_response)
    monkeypatch.setattr(OpenAIAgentProvider, "map_perspectives", _no_perspectives)
    response = client.post(
        "/api/v1/conversations",
        json={
            "topic": "Design a sustainable underwater city",
            "constraints": ["must be economically viable"],
        },
    )
    assert response.status_code == 201
    conversation = response.json()
    assert conversation["topic"] == "Design a sustainable underwater city"
    assert conversation["status"] == "draft"

    conversation_id = conversation["id"]
    step_response = client.post(f"/api/v1/conversations/{conversation_id}/step")
    assert step_response.status_code == 200
    stepped = step_response.json()
    assert stepped["current_round"] == 1
    assert len(stepped["exchanges"]) >= 3


def test_list_conversations() -> None:
    response = client.get("/api/v1/conversations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
