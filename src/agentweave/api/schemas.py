from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agentweave.core.enums import (
    AgentRole,
    AgentStatus,
    ConversationStatus,
    DiversityGapType,
    EvaluationRecommendation,
    ReplacementPressure,
)
from agentweave.core.models import AgentState, RuntimeConfig


class CreateConversationRequest(BaseModel):
    topic: str = Field(min_length=3)
    scene: str | None = None
    constraints: list[str] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    auto_start: bool = False


class AgentProfileResponse(BaseModel):
    id: str
    role: AgentRole
    personality: str
    perspective: str | None
    state: AgentState
    status: AgentStatus
    confidence: float
    priority: float
    expertise_weight: float
    contribution_score: float
    repetition_score: float
    novelty_score: float
    token_usage: int
    replacement_eligibility: float


class ExchangeResponse(BaseModel):
    id: str
    round_number: int
    agent_id: str
    role: AgentRole
    content: str
    contribution_score: float
    novelty_score: float
    repetition_score: float
    created_at: datetime


class EvaluationResponse(BaseModel):
    id: str
    round_number: int
    progress_score: float
    novelty_score: float
    coherence_score: float
    redundancy_score: float
    goal_alignment_score: float
    depth_score: float
    conflict_utility_score: float
    recommendation: EvaluationRecommendation
    rationale: str
    created_at: datetime


class ReplacementResponse(BaseModel):
    id: str
    round_number: int
    removed_agent_id: str
    removed_role: AgentRole
    added_agent_id: str
    added_role: AgentRole
    failure_mode: str
    reason: str
    pressure: ReplacementPressure
    perspective: str | None
    created_at: datetime


class PerspectiveDimensionResponse(BaseModel):
    id: str
    name: str
    theory: str
    agent_id: str | None


class DiversityAssessmentResponse(BaseModel):
    id: str
    round_number: int
    gap: DiversityGapType
    rationale: str
    dimension_name: str
    proposed_theory: str
    converged_agent_ids: list[str]
    created_at: datetime


class UsageBucketResponse(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int


class ConversationUsageResponse(BaseModel):
    by_purpose: dict[str, UsageBucketResponse]
    total: UsageBucketResponse


class MemorySummaryResponse(BaseModel):
    id: str
    round_number: int
    summary: str
    created_at: datetime


class SharedContextResponse(BaseModel):
    goal: str
    scene: str | None
    constraints: list[str]
    decisions: list[str]
    active_conflicts: list[str]
    history: list[str]


class ConversationSummaryResponse(BaseModel):
    id: str
    topic: str
    scene: str | None
    status: ConversationStatus
    current_round: int
    active_agent_count: int
    exchange_count: int
    evaluation_count: int
    replacement_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(BaseModel):
    id: str
    topic: str
    scene: str | None
    constraints: list[str]
    status: ConversationStatus
    runtime: RuntimeConfig
    current_round: int
    stall_count: int
    final_summary: str | None
    shared_context: SharedContextResponse
    agents: list[AgentProfileResponse]
    exchanges: list[ExchangeResponse]
    evaluations: list[EvaluationResponse]
    replacements: list[ReplacementResponse]
    summaries: list[MemorySummaryResponse]
    perspective_map: list[PerspectiveDimensionResponse]
    diversity_assessments: list[DiversityAssessmentResponse]
    usage: ConversationUsageResponse
    created_at: datetime
    updated_at: datetime


class ConversationEventResponse(BaseModel):
    type: str
    conversation_id: str
    created_at: datetime
    payload: dict


class FutureAGIDatasetRowResponse(BaseModel):
    conversation_id: str
    topic: str
    scene: str | None
    constraints: list[str]
    status: str
    current_round: int
    exchange_count: int
    evaluation_count: int
    replacement_count: int
    provider_mode: str
    transcript: str
    final_summary: str | None


class FutureAGIUploadRequest(BaseModel):
    dataset_name: str = Field(min_length=1)


class FutureAGIUploadResponse(BaseModel):
    dataset_name: str
    status: str
    message: str
