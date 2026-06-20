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

Create a local `.env` file and add your key:

```env
APP_ENV=development
PROVIDER_MODE=simulated
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-4.1-mini
OPENAI_EVALUATOR_MODEL=gpt-4.1-mini
```

Notes:

- keep `PROVIDER_MODE=simulated` until the provider implementation is switched to real OpenAI calls
- when you later set `PROVIDER_MODE=openai`, startup will fail if `OPENAI_API_KEY` is missing

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
- `GET /api/v1/conversations/{conversation_id}/futureagi/row`
- `POST /api/v1/conversations/{conversation_id}/futureagi/upload`

## Frontend Integration Notes

The API contracts are intentionally stable and UI-friendly:

- summary and detail responses are separated
- event streaming is isolated to a dedicated endpoint
- orchestration state is serializable without backend-only shapes
- provider logic is abstracted from the API layer

## Future AGI Export

You can normalize a conversation into a dataset-ready row and upload it to an existing Future AGI dataset.

Set these in `.env`:

```env
FI_API_KEY=
FI_SECRET_KEY=
FI_BASE_URL=https://api.futureagi.com
```

Workflow:

1. Create or choose a dataset in Future AGI.
2. Use `GET /api/v1/conversations/{conversation_id}/futureagi/row` to inspect the normalized row payload.
3. Use `POST /api/v1/conversations/{conversation_id}/futureagi/upload` with a `dataset_name` to push the conversation into that dataset through the Future AGI SDK.

The SDK path creates the dataset if missing, attempts to add the required columns, and then appends one row with `conversation_id`, `topic`, `scene`, `constraints`, `provider_mode`, `transcript`, and `final_summary`.

Example request:

```json
{
  "dataset_name": "agentweave-conversations"
}
```

## Secrets

- store secrets only in the local `.env`
- `.env` is ignored by git
- use `.env.example` as the committed template

## Next Backend Steps

- expand the OpenAI-backed provider prompts and role tuning
- persist state in Redis or Postgres
- add auth, rate limits, and workspace/session scoping
- add websocket transport if bidirectional control becomes necessary

## Provider Modes

- `PROVIDER_MODE=simulated`: runs deterministic placeholder agents for plumbing and UI development
- `PROVIDER_MODE=openai`: uses `OPENAI_API_KEY` from `.env` and calls the OpenAI Responses API

To enable real model-backed conversations:

```env
PROVIDER_MODE=openai
OPENAI_API_KEY=your_key_here
```
