from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentweave.config import DEFAULT_RUNTIME
from agentweave.core.enums import (
    AgentRole,
    AgentStatus,
    ConversationStatus,
    EvaluationRecommendation,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeConfig(BaseModel):
    agent_turn_delay_seconds: float = Field(default=0.45, ge=0.0, le=5.0)
    evaluation_interval: int = Field(
        default=DEFAULT_RUNTIME.evaluation_interval,
        ge=1,
    )
    restructuring_interval: int = Field(
        default=DEFAULT_RUNTIME.restructuring_interval,
        ge=1,
    )
    max_rounds: int = Field(default=DEFAULT_RUNTIME.max_rounds, ge=1, le=500)
    max_history_entries: int = Field(
        default=DEFAULT_RUNTIME.max_history_entries,
        ge=10,
    )
    summary_window: int = Field(default=DEFAULT_RUNTIME.summary_window, ge=2)
    stagnation_threshold: int = Field(
        default=DEFAULT_RUNTIME.stagnation_threshold,
        ge=1,
    )


class AgentProfile(BaseModel):
    id: str = Field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    role: AgentRole
    personality: str
    status: AgentStatus = AgentStatus.ACTIVE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    expertise_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    contribution_score: float = Field(default=0.5, ge=0.0, le=1.0)
    repetition_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    token_usage: int = Field(default=0, ge=0)
    replacement_eligibility: float = Field(default=0.0, ge=0.0, le=1.0)


class Exchange(BaseModel):
    id: str = Field(default_factory=lambda: f"ex_{uuid4().hex[:10]}")
    round_number: int = Field(ge=1)
    agent_id: str
    role: AgentRole
    content: str
    contribution_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    repetition_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationSnapshot(BaseModel):
    id: str = Field(default_factory=lambda: f"eval_{uuid4().hex[:10]}")
    round_number: int = Field(ge=1)
    progress_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    redundancy_score: float = Field(ge=0.0, le=1.0)
    goal_alignment_score: float = Field(ge=0.0, le=1.0)
    depth_score: float = Field(ge=0.0, le=1.0)
    conflict_utility_score: float = Field(ge=0.0, le=1.0)
    recommendation: EvaluationRecommendation
    rationale: str
    created_at: datetime = Field(default_factory=utc_now)


class ReplacementEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"rep_{uuid4().hex[:10]}")
    round_number: int = Field(ge=1)
    removed_agent_id: str
    removed_role: AgentRole
    added_agent_id: str
    added_role: AgentRole
    failure_mode: str
    reason: str
    created_at: datetime = Field(default_factory=utc_now)


class MemorySummary(BaseModel):
    id: str = Field(default_factory=lambda: f"sum_{uuid4().hex[:10]}")
    round_number: int = Field(ge=1)
    summary: str
    created_at: datetime = Field(default_factory=utc_now)


class SharedContext(BaseModel):
    goal: str
    scene: str | None = None
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    active_conflicts: list[str] = Field(default_factory=list)
    competing_perspectives: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: f"conv_{uuid4().hex}")
    topic: str
    scene: str | None = None
    constraints: list[str] = Field(default_factory=list)
    status: ConversationStatus = ConversationStatus.DRAFT
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    current_round: int = Field(default=0, ge=0)
    stall_count: int = Field(default=0, ge=0)
    final_summary: str | None = None
    shared_context: SharedContext
    agents: list[AgentProfile] = Field(default_factory=list)
    exchanges: list[Exchange] = Field(default_factory=list)
    evaluations: list[EvaluationSnapshot] = Field(default_factory=list)
    replacements: list[ReplacementEvent] = Field(default_factory=list)
    summaries: list[MemorySummary] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def active_agents(self) -> list[AgentProfile]:
        return [agent for agent in self.agents if agent.status == AgentStatus.ACTIVE]


class ConversationEvent(BaseModel):
    type: str
    conversation_id: str
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any]
