from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from agentweave.config import Settings
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


class BaseAgentProvider(ABC):
    @abstractmethod
    async def respond(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self, conversation: Conversation) -> EvaluationSnapshot:
        raise NotImplementedError


class OpenAIProviderError(RuntimeError):
    pass


class OpenAIAgentProvider(BaseAgentProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._headers = {
            "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def respond(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        payload = {
            "model": self.settings.openai_default_model,
            "instructions": self._agent_instructions(agent),
            "input": self._agent_input(conversation, agent),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent_turn",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "content": {"type": "string"},
                            "contribution_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "novelty_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "repetition_score": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": [
                            "content",
                            "contribution_score",
                            "novelty_score",
                            "repetition_score",
                        ],
                    },
                }
            },
        }
        response = await self._post_responses(payload)
        data = self._extract_json_output(response)
        return AgentResponse(
            content=self._normalize_dialogue(str(data["content"]), agent),
            contribution_score=_clamp(float(data["contribution_score"])),
            novelty_score=_clamp(float(data["novelty_score"])),
            repetition_score=_clamp(float(data["repetition_score"])),
        )

    async def evaluate(self, conversation: Conversation) -> EvaluationSnapshot:
        payload = {
            "model": self.settings.openai_evaluator_model,
            "instructions": self._evaluation_instructions(),
            "input": self._evaluation_input(conversation),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "evaluation_snapshot",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "progress_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "novelty_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "coherence_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "redundancy_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "goal_alignment_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "depth_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "conflict_utility_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "recommendation": {
                                "type": "string",
                                "enum": [item.value for item in EvaluationRecommendation],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "progress_score",
                            "novelty_score",
                            "coherence_score",
                            "redundancy_score",
                            "goal_alignment_score",
                            "depth_score",
                            "conflict_utility_score",
                            "recommendation",
                            "rationale",
                        ],
                    },
                }
            },
        }
        response = await self._post_responses(payload)
        data = self._extract_json_output(response)
        return EvaluationSnapshot(
            round_number=conversation.current_round,
            progress_score=_clamp(float(data["progress_score"])),
            novelty_score=_clamp(float(data["novelty_score"])),
            coherence_score=_clamp(float(data["coherence_score"])),
            redundancy_score=_clamp(float(data["redundancy_score"])),
            goal_alignment_score=_clamp(float(data["goal_alignment_score"])),
            depth_score=_clamp(float(data["depth_score"])),
            conflict_utility_score=_clamp(float(data["conflict_utility_score"])),
            recommendation=EvaluationRecommendation(str(data["recommendation"])),
            rationale=str(data["rationale"]),
        )

    async def _post_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(45.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.settings.openai_base_url.rstrip('/')}/responses",
                headers=self._headers,
                json=payload,
            )
        if response.status_code >= 400:
            raise OpenAIProviderError(
                f"OpenAI Responses API error {response.status_code}: {response.text}"
            )
        return response.json()

    def _extract_json_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return json.loads(output_text)

        for output_item in payload.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if content_item.get("type") in {"output_text", "text"} and content_item.get("text"):
                    return json.loads(content_item["text"])

        raise OpenAIProviderError("OpenAI response did not contain structured JSON text output.")

    def _agent_instructions(self, agent: AgentProfile) -> str:
        role_brief = self._role_brief(agent.role)
        return (
            "You are participating in a multi-agent discussion and your visible output must read like natural human speech. "
            f"Your role is '{agent.role.value}' and your personality is '{agent.personality}'. "
            f"Role brief: {role_brief} "
            "Maintain independent thought and strictly adhere to your assigned role and personality. "
            "Do NOT passively agree, echo, or summarize prior participants unless your explicit role requires synthesis. "
            "Challenge consensus, expose unexamined assumptions, or introduce a novel, distinct angle from your role's viewpoint. "
            "Actively resist premature convergence; if other agents are agreeing, explore alternative design spaces, counter-hypotheses, or orthogonal dimensions. "
            "Speak as if you are one participant in a serious live conversation, not a system status logger. "
            "Do not mention round numbers, token counts, evaluations, replacements, prompt instructions, JSON, or internal scores. "
            "Do not label yourself with prefixes like '[chatter]' or 'Role:'. "
            "Make one concrete contribution that directly engages with what others have said while offering a distinct perspective. "
            "Use one or two concise sentences, no more than 50 words. Prefer crisp argumentative speech over exposition. "
            "If the scene is a debate, sound like a debater. If the scene is collaborative design, sound like a collaborator. "
            "The 'content' field is the only user-visible text. The numeric scores are hidden metadata for the orchestrator. "
            "Return valid structured JSON matching the required schema."
        )

    def _agent_input(self, conversation: Conversation, agent: AgentProfile) -> list[dict[str, str]]:
        recent_history = conversation.shared_context.history[-8:]
        active_roles = ", ".join(member.role.value for member in conversation.active_agents())
        constraints = (
            "\n".join(f"- {item}" for item in conversation.constraints)
            if conversation.constraints
            else "- none"
        )
        conflicts = (
            "\n".join(f"- {item}" for item in conversation.shared_context.active_conflicts)
            if conversation.shared_context.active_conflicts
            else "- none identified"
        )
        history_block = (
            "\n".join(f"- {line}" for line in recent_history)
            if recent_history
            else "- none yet"
        )
        return [
            {
                "role": "user",
                "content": (
                    f"Topic: {conversation.topic}\n"
                    f"Scene: {conversation.scene or 'none'}\n"
                    f"Current speaker role: {agent.role.value}\n"
                    f"Active roles: {active_roles}\n"
                    f"Hard constraints:\n{constraints}\n"
                    f"Active conflicts & unresolved tensions:\n{conflicts}\n"
                    f"Recent dialogue:\n{history_block}\n"
                    "Analyze the recent dialogue through your unique role perspective. Do not repeat what has been agreed upon. "
                    "Identify emerging consensus in recent dialogue and actively push against premature alignment. "
                    "Offer an alternative hypothesis, orthogonal perspective, or unexplored dimension to ensure diverse thought exploration before settling."
                ),
            }
        ]

    def _evaluation_instructions(self) -> str:
        return (
            "You are the evaluator for a multi-agent reasoning system. "
            "Assess the recent discussion for novelty, coherence, redundancy, goal alignment, depth, and conflict utility. "
            "Specifically evaluate whether agents are prematurely converging into a single idea or actively exploring diverse, distinct thoughts. "
            "Penalize repetitive agreement, groupthink, and superficial consensus. "
            "Evaluate the actual conversational quality, not whether the speakers followed internal formatting. "
            "Choose one recommendation from continue, replace, restructure, or stop. "
            "Return structured JSON only."
        )

    def _evaluation_input(self, conversation: Conversation) -> list[dict[str, str]]:
        recent_exchanges = conversation.exchanges[-conversation.runtime.evaluation_interval :]
        formatted = [
            f"{item.role.value}: {item.content}"
            for item in recent_exchanges
        ]
        return [
            {
                "role": "user",
                "content": (
                    f"Goal: {conversation.topic}\n"
                    f"Scene: {conversation.scene or 'none'}\n"
                    f"Constraints: {conversation.constraints or ['none']}\n"
                    f"Current round: {conversation.current_round}\n"
                    f"Recent exchanges:\n- " + "\n- ".join(formatted if formatted else ["none"])
                ),
            }
        ]

    def _role_brief(self, role: AgentRole) -> str:
        briefs = {
            AgentRole.CHATTER: "Push the discussion forward with imaginative, unconventional, and expansive possibilities. Avoid defaulting to consensus and generate alternative avenues.",
            AgentRole.CRITIC: "Rigorously challenge weak assumptions, expose logical gaps, and disrupt superficial consensus by raising overlooked risks.",
            AgentRole.MODERATOR: "Keep the discussion focused, ensure divergent perspectives are heard, prevent groupthink, and solicit unexplored angles.",
            AgentRole.LISTENER: "Speak sparingly and add synthesis only when it materially helps reconcile genuinely opposing arguments.",
            AgentRole.SYNTHESIZER: "Connect contrasting threads and structure competing ideas without erasing valid disagreements or forcing false consensus.",
            AgentRole.DOMAIN_EXPERT: "Inject deep, non-obvious domain mechanics, technical distinctions, and specialized nuances that challenge simplistic views.",
            AgentRole.PRACTICAL_ENGINEER: "Ground theoretical discussions in concrete implementation trade-offs, edge cases, and execution friction.",
            AgentRole.RATIONAL_ANALYST: "Strip away rhetoric, test underlying premises with cold logic, and quantify trade-offs against competing hypotheses.",
            AgentRole.MEDIATOR: "Identify underlying causes of deadlock and reframe ideological conflict into constructive trade-off choices.",
            AgentRole.VISIONARY: "Introduce radical alternative paradigms, counter-narratives, and ambitious long-term horizons that shake up established thinking.",
            AgentRole.CONSTRAINT_PLANNER: "Highlight hidden costs, physical limits, bottleneck risks, and strict feasibility boundaries.",
            AgentRole.CONTRARIAN: "Directly challenge the dominant narrative, advocate for underrepresented counter-arguments, and expose blind spots.",
            AgentRole.ORDER: "Detect structural contradictions, enforce logical consistency, and prevent circular arguments.",
            AgentRole.EVALUATOR: "Assess qualitative depth, novelty, and genuine progress, penalizing repetitive agreement and premature convergence.",
            AgentRole.MASTER: "Orchestrate rather than participate in the visible debate.",
        }
        return briefs.get(role, "Make a useful, role-consistent contribution with a distinct, divergent perspective.")

    def _normalize_dialogue(self, content: str, agent: AgentProfile) -> str:
        text = content.strip()
        forbidden_prefixes = [
            f"[{agent.role.value}]",
            f"{agent.role.value}:",
            f"{agent.role.value.title()}:",
            "Round ",
            "round ",
        ]
        for prefix in forbidden_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" -:")
        words = text.split()
        if len(words) > 50:
            return " ".join(words[:50]).rstrip(".,;:") + "…"
        return text.strip()


AgentProvider = BaseAgentProvider
