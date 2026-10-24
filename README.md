# AgentWeave

AgentWeave is an adaptive multi-agent reasoning application with a FastAPI backend and a built-in live reasoning dashboard. Role-based agents discuss a topic, maintain private reasoning state, and are evaluated and replaced as the conversation develops.

## Current Backend Scope

- typed conversation creation and retrieval APIs
- orchestration loop with role-based agents
- periodic evaluation and stagnation detection
- adaptive agent replacement
- shared context and history compression
- server-sent event stream for live frontend updates
- OpenAI orchestration with optional Claude, Perplexity, and local-model speakers
- built-in dashboard served at `/`
- experiment tracking, judge calibration, and pairwise evaluations
- training-data generation, SFT dataset building, and local LoRA training scripts

Conversation state and event delivery are in memory within one process. Conversations are lost when the server restarts; eval and training artifacts are written to disk.

## Agent Diversity Model

Agents differ on two axes: **personality** (how they reason) and **perspective** (the causal theory they defend).

- **Perspective map:** on creation, the provider maps the topic's major dimensions, each with a defensible theory. Chatter and Critic each claim a distinct theory; Moderator and Evaluator stay invariant. Unclaimed dimensions form a reservoir.
- **Private agent state:** each agent keeps a persistent `AgentState` (core thesis, assumptions, causal model, claims, concessions, unresolved attacks), updated every turn. Dialogue is evidence, not identity.
- **Two replacement pressures:**
  - *Quality* (existing): evaluator failure mode picks a new personality (repetitive → Contrarian, shallow → Domain Expert, ...).
  - *Exploration* (new): a diversity audit looks for a missing dimension, a shared assumption, or converged causal models, and swaps in an `explorer` agent defending a theory that fills the gap. At most one roster change per evaluation; quality takes precedence.

Usage tracking: conversation model calls are recorded on `conversation.usage` with call counts and input/output tokens by purpose, plus a `total`. Purposes include agent turns, evaluation, diversity, perspective mapping, research, and separate vendor/local speaker and bookkeeping calls. These are usage counts, not monetary cost estimates.

Runtime knobs (per conversation, useful for baselines): `perspective_dimensions` (0 disables the map), `diversity_pressure`, `history_window`.

## Project Layout

```text
src/agentweave/
  api/          FastAPI routes and request/response schemas
  core/         Domain enums and state models
  services/     In-memory store, event bus, providers, orchestrator, Future AGI export
  web/          Dashboard HTML, CSS, and JavaScript
  evals/        Experiment runner, metrics, judges, calibration, reports
  training/     Candidate generation, SFT dataset building, model comparisons
  tuning.py     Sampling profiles and orchestration thresholds
evals/          Experiment definitions, ledgers, transcripts, reports, training data
train/          Local MLX LoRA training and held-out reply generation scripts
tests/          API, orchestration, provider routing, diversity, eval, and training tests
```

## Run Locally

Use Python 3.11 or newer. From the repository root, install dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

Copy the committed template, then set `OPENAI_API_KEY` in `.env`:

```bash
cp .env.example .env
```

The minimum OpenAI configuration is:

```env
APP_ENV=development
PROVIDER_MODE=openai
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-5.4-mini
OPENAI_EVALUATOR_MODEL=gpt-5.4-mini
```

Notes:

- `PROVIDER_MODE` must remain `openai`; startup requires `OPENAI_API_KEY`. Optional speaker providers are configured separately (see Provider below).

Start the API:

```bash
uvicorn agentweave.app:app --reload
```

Open:

- `http://127.0.0.1:8000/` — live reasoning workspace
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Tuning and Evals

`TuningConfig` (`src/agentweave/tuning.py`) controls default and per-role sampling
(`model`, `temperature`, `top_p`, `reasoning_effort`), evaluator settings, and orchestration thresholds.
Speaker profiles also accept `plain_text` and `endpoint` for local models. Unset sampling fields are omitted from requests.
Experiment variants can supply tuning; the conversation-creation HTTP request currently exposes runtime settings, not `TuningConfig`.
Model training is a separate workflow described below.

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

Additional eval commands:

```bash
python -m agentweave.evals pairwise evals/experiments/005-tuned-model-in-debate.json --yes
python -m agentweave.evals calibrate --n 100 --yes
```

`run`, `pairwise`, and `calibrate` require `--yes` to execute model calls. Calibration uses labeled argument pairs and degraded transcripts; see `python -m agentweave.evals calibrate --help` for data and source-experiment options.

## Training and Local Model Comparison

The training CLI generates candidate replies and judged preferences, builds SFT datasets, and compares held-out replies:

```bash
python -m agentweave.training gen --topics evals/training/topics.json --out evals/training/pilot --limit 3 --yes
python -m agentweave.training sanity --out evals/training/pilot --yes
python -m agentweave.training build --out evals/training/pilot
```

Generation and judging use hosted model calls; `build` processes existing samples locally.

`train/train_lora.sh [ITERS]` trains a local MLX LoRA adapter. It requires a separate MLX environment at `~/.venvs/agentweave-mlx` and a local model; these are not installed by the project dependencies. The script defaults to a Qwen3 4B MLX model under `~/.lmstudio/models/`; override `MODEL`, `DATA`, and `OUT` for your setup.

`train/generate_local.py` generates held-out replies from a base model or adapter using that MLX environment. Compare generated reply files with the hosted references using:

```bash
python -m agentweave.training compare \
  --heldout evals/training/full/sft/heldout.jsonl \
  --reply base=evals/training/full/replies_base.jsonl \
  --reply tuned=evals/training/full/replies_tuned.jsonl \
  --label base-vs-tuned --yes
```

Comparisons append results to `evals/training/results.jsonl`. `train/overnight.sh` combines dataset building, training, and reply generation after a full generation run; review its fixed paths and completion checks before using it.

## Tests

```bash
python3 -m pytest
```

Tests cover API behavior, orchestration, provider routing, speaker output, perspectives, evals, calibration, and training utilities. Test configuration sets a placeholder OpenAI key and disables optional vendor routing.

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

## Dashboard and Frontend Integration

The built-in dashboard is served at `/`, with static assets under `/assets`. Additional frontends can use the same HTTP and SSE endpoints:

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

`OpenAIAgentProvider` coordinates model calls. OpenAI uses the Responses API for structured turns, evaluation, perspective mapping, diversity assessment, and bookkeeping.

Optional speaker configuration in `.env`:

```env
CLAUDE_API_KEY=
CLAUDE_BASE_URL=https://api.anthropic.com/v1
CLAUDE_MODEL=claude-sonnet-5-5
PERPLEXITY_API_KEY=
PERPLEXITY_BASE_URL=https://api.perplexity.ai
PERPLEXITY_MODEL=sonar-pro
```

Default role routing is defined in `src/agentweave/config.py`:

- **OpenAI:** Critic, Rational Analyst, Domain Expert, Constraint Planner, Order.
- **Claude:** Chatter, Visionary, Mediator, Synthesizer, Listener, Moderator.
- **Perplexity:** Explorer, Contrarian, Practical Engineer.

Roles fall back to OpenAI when their optional vendor key is absent or blank. With Perplexity configured, selected roles also receive external research briefs. Evaluator calls remain on OpenAI.

A role's `SamplingProfile.endpoint` overrides its speaker with an OpenAI-compatible chat-completions server. Supply the API base including `/v1` where required (for example, `http://127.0.0.1:1234/v1`); the provider appends `/chat/completions`. This path sends no authentication headers. Setting `plain_text=true` uses plain-text speaking without requiring a local endpoint. Both paths use a separate OpenAI call to score the reply and update private state, so local speakers still require the OpenAI configuration.
