from __future__ import annotations

from dataclasses import dataclass

from agentweave.core.enums import AgentRole, EvaluationRecommendation
from agentweave.core.models import AgentProfile, Conversation, EvaluationSnapshot


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class AgentResponse:
    content: str
    contribution_score: float
    novelty_score: float
    repetition_score: float


class AgentProvider:
    """Provider abstraction for future LLM-backed role execution."""

    async def respond(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        active_roles = ", ".join(role.role.value for role in conversation.active_agents())
        prior_signals = len(conversation.evaluations)
        prior_replacements = len(conversation.replacements)
        goal = conversation.topic
        constraints = (
            "; ".join(conversation.constraints)
            if conversation.constraints
            else "no explicit constraints yet"
        )
        content = (
            f"[{agent.role.value}] Round {conversation.current_round + 1}: "
            f"advance '{goal}' while respecting {constraints}. "
            f"Active team: {active_roles}. "
            f"Account for {prior_signals} evaluations and {prior_replacements} replacements."
        )

        novelty_bias = {
            AgentRole.CHATTER: 0.82,
            AgentRole.LISTENER: 0.66,
            AgentRole.ORDER: 0.48,
            AgentRole.CRITIC: 0.59,
            AgentRole.SYNTHESIZER: 0.61,
            AgentRole.MODERATOR: 0.46,
            AgentRole.DOMAIN_EXPERT: 0.69,
            AgentRole.PRACTICAL_ENGINEER: 0.63,
            AgentRole.RATIONAL_ANALYST: 0.52,
            AgentRole.MEDIATOR: 0.51,
            AgentRole.VISIONARY: 0.88,
            AgentRole.CONSTRAINT_PLANNER: 0.57,
            AgentRole.CONTRARIAN: 0.71,
        }.get(agent.role, 0.55)
        repetition_penalty = min(0.55, conversation.current_round * 0.03 + prior_signals * 0.02)

        novelty_score = _clamp(novelty_bias - repetition_penalty + agent.expertise_weight * 0.1)
        repetition_score = _clamp(1.0 - novelty_score - 0.1)
        contribution_score = _clamp(
            (agent.confidence * 0.25)
            + (agent.priority * 0.2)
            + (agent.expertise_weight * 0.25)
            + (novelty_score * 0.3)
        )

        return AgentResponse(
            content=content,
            contribution_score=contribution_score,
            novelty_score=novelty_score,
            repetition_score=repetition_score,
        )

    async def evaluate(self, conversation: Conversation) -> EvaluationSnapshot:
        recent = conversation.exchanges[-conversation.runtime.evaluation_interval :]
        if not recent:
            progress = novelty = coherence = 0.0
            redundancy = 1.0
            alignment = depth = conflict_utility = 0.0
        else:
            novelty = sum(item.novelty_score for item in recent) / len(recent)
            redundancy = sum(item.repetition_score for item in recent) / len(recent)
            contribution = sum(item.contribution_score for item in recent) / len(recent)
            coherence = _clamp(1.0 - redundancy * 0.7)
            alignment = _clamp(0.7 + min(0.25, len(conversation.constraints) * 0.03))
            depth = _clamp((contribution * 0.6) + (novelty * 0.4))
            conflict_utility = _clamp(0.55 + (novelty - redundancy) * 0.35)
            progress = _clamp(
                (novelty * 0.25)
                + (coherence * 0.2)
                + (alignment * 0.2)
                + (depth * 0.2)
                + (conflict_utility * 0.15)
            )

        recommendation = EvaluationRecommendation.CONTINUE
        rationale = "Discussion remains productive."
        if redundancy > 0.62 or novelty < 0.35:
            recommendation = EvaluationRecommendation.REPLACE
            rationale = "Discussion is getting repetitive or too flat."
        if conversation.current_round >= conversation.runtime.restructuring_interval:
            recommendation = EvaluationRecommendation.RESTRUCTURE
            rationale = "Long-running discussion benefits from a team restructure."
        if progress >= 0.9:
            recommendation = EvaluationRecommendation.STOP
            rationale = "Discussion has converged sufficiently."

        return EvaluationSnapshot(
            round_number=conversation.current_round,
            progress_score=progress,
            novelty_score=novelty,
            coherence_score=coherence,
            redundancy_score=redundancy,
            goal_alignment_score=alignment,
            depth_score=depth,
            conflict_utility_score=conflict_utility,
            recommendation=recommendation,
            rationale=rationale,
        )
