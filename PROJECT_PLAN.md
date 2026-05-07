# AgentWeave Project Plan

## Project Definition

AgentWeave is an adaptive multi-agent reasoning framework built around a dynamic conversational society of agents. The system accepts an open-ended topic, problem, debate, simulation, or creative scenario, then orchestrates a structured long-horizon discussion among specialized agents. The core idea must remain strict:

- role specialization
- self-monitoring
- adaptive agent replacement
- convergence testing
- long-horizon discussion management

This is not a basic chatbot and not a flat multi-bot wrapper. It is a managed cognitive simulation framework that continuously evaluates its own discussion quality and changes its internal team when progress weakens.

## Product Goal

Build a production-capable system where a master orchestrator dynamically creates, supervises, evaluates, restructures, and terminates sub-agent discussions in pursuit of better reasoning quality, better topic coverage, and lower failure rates over long conversations.

## Core User Flow

The user provides a topic, scene, problem, debate, simulation, or creative prompt such as:

- "Design a sustainable underwater city"
- "Simulate a political negotiation"
- "Create a startup strategy"
- "Debate consciousness in AI"
- "Solve urban traffic problems"

The system then:

1. interprets the goal and constraints
2. determines the initial role mix required
3. spawns specialized agents
4. runs structured conversation rounds
5. evaluates progress every `n` exchanges
6. replaces weak or mismatched agents when needed
7. performs larger restructuring every `m` exchanges
8. stops only on convergence, exhaustion, or max-depth conditions

## Architectural Principles

The following principles are mandatory and should not be loosened during implementation:

1. Agents are not equal. They have different responsibilities, weights, and intervention patterns.
2. The master agent owns orchestration, lifecycle management, and termination logic.
3. Progress must be measured explicitly, not inferred casually from fluent output.
4. Weak agents must be replaceable at runtime.
5. Shared memory is required to prevent drift, repetition, and hallucinated continuity.
6. Long conversations require rolling memory, summarization, and compression.
7. The system must tolerate model or provider instability through routing and fallback support.

## Agent Roles

### 1. Chatter / Explorer Agent

Purpose:

- drive idea generation
- create hypotheses
- maintain momentum

Behavior:

- high creativity
- expansive reasoning
- possibility generation

### 2. Listener + Value Adder Agent

Purpose:

- observe most of the time
- intervene only when contribution quality is high

Behavior:

- sparse output
- synthesis
- connection detection
- token efficiency

### 3. Order / Constraint Agent

Purpose:

- prevent derailment
- enforce scope
- detect repetition
- check internal consistency

Behavior:

- moderation
- constraint enforcement
- contradiction detection

### 4. Progress Evaluator Agent

Runs every `n` exchanges.

Purpose:

- judge whether the discussion is converging, stagnant, repetitive, chaotic, or productive

Outputs:

- progress score
- novelty score
- coherence score
- recommendation

### 5. Agent Evolution Manager

Runs every `m` exchanges.

Purpose:

- remove weak agents
- introduce better-suited personalities
- restructure the team when local optimization is no longer enough

## Master Agent Responsibilities

The master agent is the operating center of the system. It must:

1. understand the topic and user goal
2. derive relevant roles and personalities
3. spawn initial sub-agents
4. maintain conversation health
5. collect metrics and scores
6. decide whether to continue, compress, replace, or terminate
7. perform adaptive restructuring without losing discussion continuity

## Core Conversation Loop

```text
User Input
    ↓
Master Agent analyzes
    ↓
Create sub-agents
    ↓
Conversation rounds begin
    ↓
Every n exchanges:
    Progress evaluation
    ↓
If stagnating:
    Replace weakest agent
    ↓
Continue discussion
    ↓
Every m exchanges:
    Major restructuring
    ↓
Repeat until:
    convergence OR max depth
```

## Agent Lifecycle Model

Each agent should expose a tracked runtime profile similar to:

```python
{
    "id": "agent_4",
    "role": "critic",
    "personality": "skeptical",
    "status": "active",
    "contribution_score": 0.71,
    "repetition_score": 0.22,
    "novelty_score": 0.65,
    "token_usage": 18000
}
```

Required lifecycle dimensions:

- identity
- role
- personality
- activity status
- contribution quality
- repetition level
- novelty level
- token consumption
- replacement eligibility

## Replacement Logic

Adaptive replacement is a core innovation, not an optional optimization.

When progress drops or discussion quality degrades, the system should remove the weakest or least-fit agent and introduce a more appropriate one. Initial replacement mapping:

| Failure Mode | Replacement Type |
| --- | --- |
| Too chaotic | Synthesizer |
| Too repetitive | Contrarian |
| Too shallow | Domain expert |
| Too theoretical | Practical engineer |
| Too emotional | Rational analyst |
| Deadlocked | Mediator |
| No creativity | Visionary |
| No realism | Constraint planner |

Replacement should consider:

- recent contribution quality
- redundancy trend
- topic alignment
- cost efficiency
- fit against current failure mode

## Progress Evaluation Framework

Every `n` exchanges, the evaluator layer must score the conversation on:

- Novelty: are materially new ideas appearing?
- Convergence: are ideas becoming clearer or more actionable?
- Redundancy: are agents repeating prior content?
- Goal alignment: is the discussion still centered on the user objective?
- Depth: is the reasoning deep enough?
- Conflict utility: is disagreement useful rather than noisy?

Illustrative trigger:

```python
if exchange_count % n == 0:
    evaluate_progress()
```

## End Conditions

The system should stop when one or more of the following conditions are met:

- convergence score exceeds threshold
- no meaningful new ideas appear for a configured span
- max rounds is reached
- evaluator judges the discussion sufficiently complete

Illustrative logic:

```python
if (
    convergence_score > 0.9
    or no_new_ideas > threshold
    or max_rounds_exceeded
):
    stop()
```

## Weighted Influence Model

All agents should not influence outcomes equally. Each agent should carry:

- confidence
- priority
- expertise weight

This prevents noisy or overly creative agents from dominating the final result. Final synthesis decisions should consider weighted quality rather than turn count or verbosity alone.

## Shared Memory Model

All active agents require access to a shared state object such as:

```python
shared_context = {
    "goal": "...",
    "history": [...],
    "decisions": [...],
    "active_conflicts": [...],
    "constraints": [...]
}
```

Minimum memory domains:

- user goal
- constraints
- compressed discussion history
- accepted decisions
- unresolved conflicts
- active agent roster
- evaluation snapshots

## Memory and Compression Strategy

Long-horizon discussion will fail without memory control. The implementation must include:

- short-term rolling memory
- periodic summarization
- conversation compression
- evaluator-aware state reduction

Illustrative trigger:

```python
if history_tokens > threshold:
    summarize_history()
```

## Recommended Technical Stack

| Layer | Recommended Technology |
| --- | --- |
| LLM orchestration | Python |
| Async runtime | `asyncio` |
| Agent framework | LangGraph or custom orchestration layer |
| API provider | OpenAI |
| Multi-provider fallback | AssemblyAI Gateway |
| State store | Redis or Postgres |
| Memory | Vector DB |
| Streaming transport | WebSockets |
| Frontend | React |

## Why Gateway Routing Matters

Gateway routing, streaming, fallback handling, tool calling, and OpenAI-compatible endpoints are part of the production plan.

### OpenAI-Compatible Endpoint

Allows provider switching without rewriting orchestration logic. This is important for:

- role-specific model assignment
- evaluator cost control
- provider substitution

### Automatic Fallbacks

Long-running agent discussions cannot collapse because one model or provider becomes unavailable. Fallback routing is required for resilience.

### Streaming

Streaming is necessary for:

- live agent output
- evaluator updates
- visible restructuring events

### Tool Calling

Tool access is important for:

- search
- retrieval
- scoring workflows
- database access
- evaluation routines

## Suggested Role Hierarchy

```text
                    MASTER
                       |
    ------------------------------------------------
    |              |              |               |
  Chatter       Critic        Synthesizer     Moderator
    |                                              |
    ----------------Evaluation Layer---------------
                       |
              Evolution Manager
```

## Default Runtime Parameters

Recommended initial defaults:

```python
n = 5
m = 25
max_rounds = 100
```

Where:

- `n` is the mini evaluation interval
- `m` is the major restructuring interval
- `max_rounds` is the hard discussion cap

## Model Allocation Strategy

Use different model classes by role to control cost and improve behavior:

| Role | Model Style |
| --- | --- |
| Chatter | creative |
| Evaluator | logical |
| Moderator | cheap fast model |
| Synthesizer | balanced |
| Long-memory summarizer | cheap model |

## MVP Scope

The MVP must stay disciplined. Do not start with a large uncontrolled agent pool.

Initial MVP roster:

1. Master
2. Chatter
3. Critic
4. Moderator
5. Evaluator

The first milestone is not "maximum realism." It is proving that structured multi-agent discussion can remain coherent, measurable, and adaptive over time.

## Delivery Phases

### Phase 1: Static Multi-Agent Conversation

Build:

- master orchestrator
- static 4-agent discussion loop
- shared context object
- transcript capture

Success criteria:

- agents can discuss a prompt coherently for several rounds
- the master can preserve goal and constraints

### Phase 2: Periodic Evaluation

Build:

- evaluator every 5 turns
- scoring schema
- progress snapshots

Success criteria:

- system can distinguish productive vs repetitive discussion

### Phase 3: Agent Replacement

Build:

- weak-agent detection
- replacement policy
- runtime agent insertion/removal

Success criteria:

- system can recover from stagnation or mismatch

### Phase 4: Adaptive Personality Generation

Build:

- personality selection logic
- role mutation and fit-based spawning

Success criteria:

- newly spawned agents improve discussion state more often than random replacement

### Phase 5: Memory and Streaming UI

Build:

- history compression
- long-run memory handling
- websocket streaming
- React interface

Success criteria:

- conversations remain observable and affordable at longer depths

## Data Model Plan

Core entities:

- `Conversation`
- `AgentProfile`
- `Exchange`
- `EvaluationSnapshot`
- `ReplacementEvent`
- `MemorySummary`
- `ConstraintSet`

Each entity should be designed for replay, analysis, and future experimentation.

## Backend Workstreams

1. orchestration engine
2. agent runtime abstraction
3. evaluation pipeline
4. lifecycle and replacement manager
5. memory compression pipeline
6. provider routing and fallback integration
7. streaming transport
8. persistence and analytics

## Frontend Workstreams

1. live conversation timeline
2. active agent roster with status and scores
3. evaluation snapshots
4. replacement event feed
5. convergence and progress indicators
6. final synthesis display

## Observability Requirements

The system should expose:

- per-agent token usage
- contribution quality trends
- repetition trends
- replacement reasons
- evaluation intervals
- convergence state
- provider/model usage by role

Without observability, adaptive behavior cannot be trusted or improved.

## Key Risks

The primary technical risks are:

1. context explosion
2. token cost
3. agent repetition
4. coordination collapse
5. infinite loops
6. false progress
7. memory compression failure

The evaluator and compression layers are first-class responses to these risks and should be implemented early.

## Research Alignment

This system is closely aligned with:

- Society of Mind
- Multi-Agent Deliberation
- Reflexive Agents
- Evolutionary Agent Systems
- Self-Healing AI Systems
- Dynamic Role Assignment
- Autonomous Debate Systems
- Tree of Thoughts
- Graph of Thoughts

## Explicit Non-Goals For Early Versions

Do not prioritize these before the MVP proves stable:

- large agent counts
- fully persistent identities across all sessions
- emotional simulation
- recursive agent spawning by sub-agents
- elaborate reputation economies

These are later extensions, not launch requirements.

## Future Extensions

After the core system is stable, add:

- emotion models
- reputation systems
- persistent personalities
- agent genetics
- recursive specialist spawning

## Implementation Order

1. define core schemas and shared context
2. implement master orchestration loop
3. implement static agent roster
4. implement evaluator cadence and scoring
5. implement replacement logic
6. add memory compression
7. add provider routing and fallback support
8. add streaming UI
9. add analytics and tuning

## Project Outcome

If executed correctly, AgentWeave becomes an adaptive cognitive simulation framework for collaborative reasoning rather than a conventional chatbot. Its core value comes from measured discussion quality, runtime restructuring, and sustained long-horizon coherence under changing conditions.
