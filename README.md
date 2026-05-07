# AgentWeave

AgentWeave is an adaptive multi-agent reasoning backend designed around a dynamic conversational society of agents. The current implementation provides a clean HTTP API and streaming surface that a frontend can consume later without backend rewrites.

## Current Backend Scope

- typed conversation creation and retrieval APIs
- orchestration loop with role-based agents
- periodic evaluation and stagnation detection
- adaptive agent replacement
- shared context and history compression
- server-sent event stream for live frontend updates
- provider abstraction for future LLM integration

## Project Layout

```text
src/agentweave/
  api/          FastAPI routes and request/response schemas
  core/         Domain enums and state models
  services/     Store, event bus, provider abstraction, orchestrator
tests/          Basic API tests
```

## Run Locally

Install dependencies:

```bash
python3 -m pip install -e .[dev]
```

Start the API:

```bash
uvicorn agentweave.app:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Primary API Endpoints

- `POST /api/v1/conversations`
- `GET /api/v1/conversations`
- `GET /api/v1/conversations/{conversation_id}`
- `POST /api/v1/conversations/{conversation_id}/start`
- `POST /api/v1/conversations/{conversation_id}/step`
- `POST /api/v1/conversations/{conversation_id}/stop`
- `GET /api/v1/conversations/{conversation_id}/events`

## Frontend Integration Notes

The API contracts are intentionally stable and UI-friendly:

- summary and detail responses are separated
- event streaming is isolated to a dedicated endpoint
- orchestration state is serializable without backend-only shapes
- provider logic is abstracted from the API layer

## Next Backend Steps

- replace the simulated provider with real model-backed role execution
- persist state in Redis or Postgres
- add auth, rate limits, and workspace/session scoping
- add websocket transport if bidirectional control becomes necessary
