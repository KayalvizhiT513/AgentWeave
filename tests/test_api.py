from fastapi.testclient import TestClient

from agentweave.app import app


client = TestClient(app)


def test_create_and_step_conversation() -> None:
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
