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

## Agent Diversity Model

Agents differ on two axes: **personality** (how they reason) and **perspective** (the causal theory they defend).

- **Perspective map:** on creation, the provider maps the topic's major dimensions, each with a defensible theory. Chatter and Critic each claim a distinct theory; Moderator and Evaluator stay invariant. Unclaimed dimensions form a reservoir.
- **Private agent state:** each agent keeps a persistent `AgentState` (core thesis, assumptions, causal model, claims, concessions, unresolved attacks), updated every turn. Dialogue is evidence, not identity.
- **Two replacement pressures:**
  - *Quality* (existing): evaluator failure mode picks a new personality (repetitive → Contrarian, shallow → Domain Expert, ...).
  - *Exploration* (new): a diversity audit looks for a missing dimension, a shared assumption, or converged causal models, and swaps in an `explorer` agent defending a theory that fills the gap. At most one roster change per evaluation; quality takes precedence.

Cost tracking: every model call is recorded on `conversation.usage`, by purpose (`agent_turn`, `evaluation`, `diversity`, `perspective_map`) with call counts and input/output tokens, plus a `total`. This makes the overhead of the evaluator and diversity audit measurable per conversation.

Runtime knobs (per conversation, useful for baselines): `perspective_dimensions` (0 disables the map), `diversity_pressure`, `history_window`.

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
PROVIDER_MODE=openai
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-5.4-mini
OPENAI_EVALUATOR_MODEL=gpt-5.4-mini
```

Notes:

- OpenAI is the only supported runtime provider; startup fails if `OPENAI_API_KEY` is missing.

Start the API:

```bash
uvicorn agentweave.app:app --reload
```

Open:

- `http://127.0.0.1:8000/` — live reasoning workspace
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Tuning and Evals

Tuning here means settings, not training. `TuningConfig` (`src/agentweave/tuning.py`) holds per-role sampling
(`model`, `temperature`, `top_p`, `reasoning_effort`) and the orchestration thresholds that used to be hardcoded.
Unset fields are not sent, so defaults behave exactly as before.

Every tuning claim goes through the eval harness, which keeps a permanent record in `evals/`:

```bash
export PYTHONPATH=src
python -m agentweave.evals plan evals/experiments/001-baseline-reference.json   # shows run count and call estimate
python -m agentweave.evals run  evals/experiments/001-baseline-reference.json --yes
python -m agentweave.evals report                                                 # regenerates evals/EVALS.md
```

- `evals/experiments/*.json`: pre-registered hypothesis, primary metrics and variants.
- `evals/ledger.jsonl`: append-only, one row per run with full config, git sha, models, metrics, transcript path.
- `evals/EVALS.md`: generated tables. `evals/JOURNAL.md`: what we concluded and changed.

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

## Provider

AgentWeave uses `OPENAI_API_KEY` from `.env` and calls the OpenAI Responses API.
