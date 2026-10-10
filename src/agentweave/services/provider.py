from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, DiversityGapType, EvaluationRecommendation, SpeakerProvider
from agentweave.core.models import (
    AgentProfile,
    AgentState,
    Conversation,
    ConversationUsage,
    DiversityAssessment,
    EvaluationSnapshot,
    PerspectiveDimension,
)
from agentweave.services.text import strip_reasoning_block
from agentweave.tuning import SamplingProfile

logger = logging.getLogger(__name__)

# Tries per turn when a speaking model returns nothing. Retries show up as extra calls in the usage table.
SPEAKER_ATTEMPTS = 4


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class AgentResponse:
    content: str
    contribution_score: float
    novelty_score: float
    repetition_score: float
    state_update: AgentState | None = None


_STATE_FIELDS = ("assumptions", "causal_model", "claims", "concessions", "unresolved_attacks")


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "core_thesis": {"type": "string"},
        **{name: _string_array() for name in _STATE_FIELDS},
    },
    "required": ["core_thesis", *_STATE_FIELDS],
}


class BaseAgentProvider(ABC):
    @abstractmethod
    async def respond(self, conversation: Conversation, agent: AgentProfile) -> AgentResponse:
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self, conversation: Conversation) -> EvaluationSnapshot:
        raise NotImplementedError

    async def map_perspectives(
        self, conversation: Conversation, count: int
    ) -> list[PerspectiveDimension]:
        """Discover distinct theories for the topic. Providers without support return none."""
        return []

    async def assess_diversity(self, conversation: Conversation) -> DiversityAssessment | None:
        """Detect gaps in the reasoning space. Providers without support return None."""
        return None


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
        profile = conversation.tuning.for_role(agent.role)
        external = self.settings.speaker_provider(agent.role) != SpeakerProvider.OPENAI
        if profile.endpoint or profile.plain_text or external:
            return await self._respond_plain(conversation, agent, profile)
        payload = {
            "model": self.settings.openai_default_model,
            "temperature": conversation.runtime.temperature,
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
                            "state_update": _STATE_SCHEMA,
                        },
                        "required": [
                            "content",
                            "contribution_score",
                            "novelty_score",
                            "repetition_score",
                            "state_update",
                        ],
                    },
                }
            },
        }
        self._apply_sampling(payload, conversation.tuning.for_role(agent.role))
        response = await self._post_responses(payload, conversation, "agent_turn")
        data = self._extract_json_output(response)
        return AgentResponse(
            content=self._normalize_dialogue(str(data["content"]), agent),
            contribution_score=_clamp(float(data["contribution_score"])),
            novelty_score=_clamp(float(data["novelty_score"])),
            repetition_score=_clamp(float(data["repetition_score"])),
            state_update=self._parse_state(data.get("state_update")),
        )

    async def _respond_plain(
        self, conversation: Conversation, agent: AgentProfile, profile: SamplingProfile
    ) -> AgentResponse:
        """The speaker (hosted or a local endpoint) writes only the reply; a hosted call does the bookkeeping."""
        system = self._agent_instructions(agent, plain=True)
        user = self._agent_input(conversation, agent, plain=True)[0]["content"]
        raw = ""
        for _ in range(SPEAKER_ATTEMPTS):  # a local model can occasionally stop at once
            vendor = self.settings.speaker_provider(agent.role)
            if profile.endpoint:
                raw = await self._local_chat(profile, system, user, conversation)
            elif vendor != SpeakerProvider.OPENAI:
                raw = await self._vendor_chat(vendor, profile, system, user, conversation)
            else:
                payload: dict[str, Any] = {
                    "model": self.settings.openai_default_model,
                    "instructions": system,
                    "input": [{"role": "user", "content": user}],
                }
                self._apply_sampling(payload, profile)
                raw = self._plain_output(await self._post_responses(payload, conversation, "agent_turn"))
            if strip_reasoning_block(raw):
                break
        else:
            raise OpenAIProviderError(f"The speaking model returned an empty reply {SPEAKER_ATTEMPTS} times.")
        reply = self._normalize_dialogue(strip_reasoning_block(raw), agent)
        contribution, novelty, repetition, state = await self._bookkeep(conversation, agent, reply)
        return AgentResponse(
            content=reply,
            contribution_score=contribution,
            novelty_score=novelty,
            repetition_score=repetition,
            state_update=state,
        )

    @staticmethod
    def _plain_output(payload: dict[str, Any]) -> str:
        text = payload.get("output_text")
        if isinstance(text, str) and text.strip():
            return text
        for item in payload.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") in {"output_text", "text"} and part.get("text"):
                        return str(part["text"])
        return ""

    async def _vendor_chat(
        self,
        vendor: SpeakerProvider,
        profile: SamplingProfile,
        system: str,
        user: str,
        conversation: Conversation,
    ) -> str:
        """One spoken turn from Claude or Perplexity. profile.model names an OpenAI model, so it is not sent."""
        settings = self.settings
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            if vendor == SpeakerProvider.CLAUDE:
                payload: dict[str, Any] = {
                    "model": settings.claude_model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "max_tokens": 200,
                }
                if profile.temperature is not None:
                    payload["temperature"] = min(profile.temperature, 1.0)
                response = await client.post(
                    f"{settings.claude_base_url.rstrip('/')}/messages",
                    headers={
                        "x-api-key": settings.claude_api_key.get_secret_value(),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
            else:
                payload = {
                    "model": settings.perplexity_model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": 200,
                }
                if profile.temperature is not None:
                    payload["temperature"] = min(profile.temperature, 1.99)
                response = await client.post(
                    f"{settings.perplexity_base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.perplexity_api_key.get_secret_value()}"},
                    json=payload,
                )
        if response.status_code >= 400:
            raise OpenAIProviderError(f"{vendor.value} API error {response.status_code}: {response.text}")
        body = response.json()
        usage = body.get("usage") or {}
        if vendor == SpeakerProvider.CLAUDE:
            conversation.usage.record(
                "agent_turn_claude", int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
            )
            return "".join(part.get("text", "") for part in body.get("content") or [] if part.get("type") == "text")
        conversation.usage.record(
            "agent_turn_perplexity", int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        )
        text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return re.sub(r"\s*\[\d+\]", "", text)  # drop search-citation markers; this is spoken dialogue

    async def _local_chat(
        self, profile: SamplingProfile, system: str, user: str, conversation: Conversation
    ) -> str:
        payload: dict[str, Any] = {
            "model": profile.model or "default_model",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 200,
        }
        if profile.temperature is not None:
            payload["temperature"] = profile.temperature
        if profile.top_p is not None:
            payload["top_p"] = profile.top_p
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            response = await client.post(
                f"{(profile.endpoint or '').rstrip('/')}/chat/completions", headers={}, json=payload
            )
        if response.status_code >= 400:
            raise OpenAIProviderError(f"Local model error {response.status_code}: {response.text}")
        body = response.json()
        usage = body.get("usage") or {}
        conversation.usage.record(
            "agent_turn_local", int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        )
        return str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

    async def _bookkeep(
        self, conversation: Conversation, agent: AgentProfile, reply: str
    ) -> tuple[float, float, float, AgentState | None]:
        """Scores and the agent's revised state, written by a hosted model that is not the speaker."""
        history = conversation.shared_context.history[-conversation.runtime.history_window :]
        state_text = self._state_block(agent, plain=True) or "No private state yet.\n"
        recent = "\n".join(f"- {line}" for line in history or ["none yet"])
        payload: dict[str, Any] = {
            "model": self.settings.openai_evaluator_model,
            "instructions": (
                "You keep the records for one participant in a multi-agent discussion. Given their private "
                "state, the recent dialogue and the reply they just gave, return: contribution_score (0-1, how "
                "much the reply advances the discussion), novelty_score (0-1, how new it is compared with "
                "earlier turns), repetition_score (0-1, how much it restates earlier turns by anyone), and the "
                "participant's revised state. Keep core_thesis stable. Keep each list to at most 6 short items. "
                "Return structured JSON only."
            ),
            "input": [
                {
                    "role": "user",
                    "content": (
                        f"Topic: {conversation.topic}\nParticipant role: {agent.role.value}\n"
                        f"{state_text}Recent dialogue:\n{recent}\nThe reply just given: {reply}"
                    ),
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "turn_bookkeeping",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "contribution_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "novelty_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "repetition_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "state_update": _STATE_SCHEMA,
                        },
                        "required": ["contribution_score", "novelty_score", "repetition_score", "state_update"],
                    },
                }
            },
        }
        self._apply_sampling(payload, conversation.tuning.evaluator)
        try:
            data = self._extract_json_output(await self._post_responses(payload, conversation, "bookkeeping"))
            return (
                _clamp(float(data["contribution_score"])),
                _clamp(float(data["novelty_score"])),
                _clamp(float(data["repetition_score"])),
                self._parse_state(data.get("state_update")),
            )
        except Exception:
            # Records are secondary to the debate itself; the usage table shows how often this call succeeded.
            logger.warning("Bookkeeping failed for %s; using neutral scores.", agent.role.value, exc_info=True)
            return 0.5, 0.5, 0.0, None

    @staticmethod
    def _parse_state(raw: Any) -> AgentState | None:
        if not isinstance(raw, dict):
            return None
        return AgentState(
            core_thesis=str(raw.get("core_thesis", "")),
            **{
                name: [str(item) for item in raw.get(name, [])]
                for name in _STATE_FIELDS
            },
        )

    async def map_perspectives(
        self, conversation: Conversation, count: int
    ) -> list[PerspectiveDimension]:
        payload = {
            "model": self.settings.openai_evaluator_model,
            "instructions": (
                "You map the reasoning space of a topic for a multi-agent discussion. "
                "Identify the major, mutually distinct dimensions along which this topic can be understood, "
                "and for each give one concrete causal theory that a committed advocate could defend. "
                "Theories must be plausible, none obviously false, and ideally incompatible with one another "
                "in what they treat as the primary cause or value. "
                "Do not assume any particular domain; derive the dimensions from the topic itself. "
                "Order them from most fundamental to most peripheral. Return structured JSON only."
            ),
            "input": [
                {
                    "role": "user",
                    "content": (
                        f"Topic: {conversation.topic}\n"
                        f"Scene: {conversation.scene or 'none'}\n"
                        f"Constraints: {conversation.constraints or ['none']}\n"
                        f"Return exactly {count} dimensions."
                    ),
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "perspective_map",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "dimensions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "theory": {"type": "string"},
                                    },
                                    "required": ["name", "theory"],
                                },
                            }
                        },
                        "required": ["dimensions"],
                    },
                }
            },
        }
        self._apply_sampling(payload, conversation.tuning.evaluator)
        data = self._extract_json_output(
            await self._post_responses(payload, conversation, "perspective_map")
        )
        return [
            PerspectiveDimension(name=str(item["name"]).strip(), theory=str(item["theory"]).strip())
            for item in data.get("dimensions", [])
            if str(item.get("name", "")).strip() and str(item.get("theory", "")).strip()
        ][:count]

    async def assess_diversity(self, conversation: Conversation) -> DiversityAssessment | None:
        payload = {
            "model": self.settings.openai_evaluator_model,
            "instructions": (
                "You audit the diversity of a multi-agent discussion. Quality is not your concern; "
                "look for missing reasoning space. Decide whether exactly one gap exists: "
                "'missing_dimension' (an important dimension nobody has argued from), "
                "'shared_assumption' (every agent silently assumes the same thing), "
                "'converged_models' (two or more agents now defend the same causal explanation), or 'none'. "
                "If a gap exists, propose one concrete theory for a new agent that would fill it, "
                "and for converged_models list the ids of the agents that converged. "
                "Prefer unclaimed dimensions from the perspective map when they fit. "
                "Return structured JSON only."
            ),
            "input": [{"role": "user", "content": self._diversity_input(conversation)}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "diversity_assessment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "gap": {"type": "string", "enum": [item.value for item in DiversityGapType]},
                            "rationale": {"type": "string"},
                            "dimension_name": {"type": "string"},
                            "proposed_theory": {"type": "string"},
                            "converged_agent_ids": _string_array(),
                        },
                        "required": [
                            "gap",
                            "rationale",
                            "dimension_name",
                            "proposed_theory",
                            "converged_agent_ids",
                        ],
                    },
                }
            },
        }
        self._apply_sampling(payload, conversation.tuning.evaluator)
        data = self._extract_json_output(
            await self._post_responses(payload, conversation, "diversity")
        )
        return DiversityAssessment(
            round_number=max(conversation.current_round, 1),
            gap=DiversityGapType(str(data["gap"])),
            rationale=str(data["rationale"]),
            dimension_name=str(data["dimension_name"]),
            proposed_theory=str(data["proposed_theory"]),
            converged_agent_ids=[str(item) for item in data["converged_agent_ids"]],
        )

    def _diversity_input(self, conversation: Conversation) -> str:
        by_id = {agent.id: agent for agent in conversation.agents}
        dimensions = [
            f"- {dim.name}: {dim.theory} "
            f"[{'claimed by ' + by_id[dim.agent_id].role.value if dim.agent_id in by_id else 'unclaimed'}]"
            for dim in conversation.perspective_map
        ] or ["- none mapped"]
        agents = [
            f"- id={agent.id} role={agent.role.value} thesis={agent.state.core_thesis or agent.perspective or 'none'} "
            f"assumptions={agent.state.assumptions} causal_model={agent.state.causal_model}"
            for agent in conversation.active_agents()
            if agent.role != AgentRole.EVALUATOR
        ]
        recent = [
            f"{item.role.value}: {item.content}"
            for item in conversation.exchanges[-conversation.runtime.evaluation_interval * 3 :]
        ]
        return (
            f"Topic: {conversation.topic}\n"
            f"Scene: {conversation.scene or 'none'}\n"
            f"Perspective map:\n" + "\n".join(dimensions) + "\n"
            "Active agents:\n" + "\n".join(agents) + "\n"
            "Recent exchanges:\n- " + "\n- ".join(recent or ["none"])
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
        self._apply_sampling(payload, conversation.tuning.evaluator)
        response = await self._post_responses(payload, conversation, "evaluation")
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

    @staticmethod
    def _apply_sampling(payload: dict[str, Any], profile: SamplingProfile) -> None:
        """Overlay tuning onto a request. Unset fields are left out so API defaults apply."""
        if profile.model:
            payload["model"] = profile.model
        if profile.temperature is not None:
            payload["temperature"] = profile.temperature
        if profile.top_p is not None:
            payload["top_p"] = profile.top_p
        if profile.reasoning_effort:
            payload["reasoning"] = {"effort": profile.reasoning_effort}

    async def structured_output(
        self,
        *,
        instructions: str,
        user_input: str,
        schema_name: str,
        schema: dict[str, Any],
        sampling: SamplingProfile | None = None,
        usage: ConversationUsage | None = None,
    ) -> dict[str, Any]:
        """One strict-JSON call outside any conversation, e.g. an independent judge."""
        payload: dict[str, Any] = {
            "model": self.settings.openai_evaluator_model,
            "instructions": instructions,
            "input": [{"role": "user", "content": user_input}],
            "text": {
                "format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}
            },
        }
        if sampling is not None:
            self._apply_sampling(payload, sampling)
        body = await self._post_responses(payload, None, schema_name)
        if usage is not None:
            counts = body.get("usage") or {}
            usage.record(
                schema_name, int(counts.get("input_tokens") or 0), int(counts.get("output_tokens") or 0)
            )
        return self._extract_json_output(body)

    async def _post_responses(
        self,
        payload: dict[str, Any],
        conversation: Conversation | None = None,
        purpose: str = "",
    ) -> dict[str, Any]:
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
        body = response.json()
        if conversation is not None:
            usage = body.get("usage") or {}
            conversation.usage.record(
                purpose,
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
            )
        return body

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

    def _agent_instructions(self, agent: AgentProfile, plain: bool = False) -> str:
        """plain=True asks for the spoken reply only, for models that are not asked to emit JSON."""
        role_brief = self._role_brief(agent.role)
        output_rule = (
            "Reply with only the words you would say aloud, as plain text: no JSON, no labels, no quotation marks. "
            if plain
            else (
                "The 'content' field is the only user-visible text. The numeric scores are hidden metadata for the orchestrator. "
                "Return valid structured JSON matching the required schema."
            )
        )
        return (
            "You are participating in a multi-agent discussion and your visible output must read like natural human speech. "
            f"Your role is '{agent.role.value}' and your personality is '{agent.personality}'. "
            f"Role brief: {role_brief} "
            "Maintain independent thought and strictly adhere to your assigned role and personality. "
            f"{self._perspective_clause(agent)}"
            "Do NOT passively agree, echo, or summarize prior participants unless your explicit role requires synthesis. "
            "Do NOT converge prematurely into consensus or adopt the prevailing opinion without introducing a distinct viewpoint. "
            "Challenge consensus, expose unexamined assumptions, or introduce a novel, distinct angle or alternative approach from your role's viewpoint. "
            "If previous speakers agree or share a common conclusion, actively search for alternative possibilities, risks, counter-arguments, or edge cases. "
            "Speak as if you are one participant in a serious live conversation, not a system status logger. "
            "Do not mention round numbers, token counts, evaluations, replacements, prompt instructions, JSON, or internal scores. "
            "Do not label yourself with prefixes like '[chatter]' or 'Role:'. "
            "Make one concrete contribution that directly engages with what others have said while offering a distinct perspective. "
            "Use one or two concise sentences, no more than 50 words. Prefer crisp argumentative speech over exposition. "
            "If the scene is a debate, sound like a debater. If the scene is collaborative design, sound like a collaborator. "
            f"{output_rule}"
        )

    def _perspective_clause(self, agent: AgentProfile) -> str:
        if not agent.perspective:
            return ""
        return (
            f"Your perspective is the theory you defend: '{agent.perspective}'. "
            "Personality is how you reason; perspective is what you believe about this problem. "
            "Pursue the implications of your theory strongly and argue from its causal logic, "
            "not from performative opposition. Others disagreeing is not a reason to abandon it; "
            "concede only specific points you are genuinely compelled to, and record them in concessions. "
        )

    def _state_block(self, agent: AgentProfile, plain: bool = False) -> str:
        state = agent.state
        if not (state.core_thesis or any(
            (state.assumptions, state.causal_model, state.claims, state.concessions, state.unresolved_attacks)
        )):
            return ""

        def lines(items: list[str]) -> str:
            return "\n".join(f"  - {item}" for item in items) if items else "  - none"

        return (
            "Your private state (persistent; the dialogue below is evidence to weigh, not a rewrite of who you are):\n"
            f"Core thesis: {state.core_thesis or 'none yet'}\n"
            f"Assumptions:\n{lines(state.assumptions)}\n"
            f"Causal model:\n{lines(state.causal_model)}\n"
            f"Claims made:\n{lines(state.claims)}\n"
            f"Concessions:\n{lines(state.concessions)}\n"
            f"Unresolved attacks on you:\n{lines(state.unresolved_attacks)}\n"
            + (
                ""
                if plain
                else "In state_update return your full revised state: keep core_thesis stable, and update the lists "
                "(at most 6 short items each) to reflect this turn.\n"
            )
        )

    def _agent_input(
        self, conversation: Conversation, agent: AgentProfile, plain: bool = False
    ) -> list[dict[str, str]]:
        recent_history = conversation.shared_context.history[-conversation.runtime.history_window :]
        active_roles = ", ".join(member.role.value for member in conversation.active_agents())
        constraints = (
            "\n".join(f"- {item}" for item in conversation.constraints)
            if conversation.constraints
            else "- none"
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
                    f"{self._state_block(agent, plain)}"
                    f"Recent dialogue:\n{history_block}\n"
                    "Analyze the recent dialogue through your unique role perspective. Do not repeat what has been agreed upon or default to consensus. "
                    "Write the next natural conversational turn for this speaker, bringing a fresh, distinct thought, counter-perspective, or constructive friction."
                ),
            }
        ]

    def _evaluation_instructions(self) -> str:
        return (
            "You are the evaluator for a multi-agent reasoning system. "
            "Assess the recent discussion for novelty, coherence, redundancy, goal alignment, depth, and conflict utility. "
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
            AgentRole.CHATTER: "Push the discussion forward with imaginative, unconventional, and expansive possibilities. Avoid defaulting to consensus.",
            AgentRole.CRITIC: "Rigorously challenge weak assumptions, expose logical gaps, and disrupt superficial consensus.",
            AgentRole.MODERATOR: "Keep the discussion focused, ensure divergent perspectives are heard, and prevent groupthink.",
            AgentRole.LISTENER: "Speak sparingly and add synthesis only when it materially helps reconcile genuinely opposing arguments.",
            AgentRole.SYNTHESIZER: "Connect contrasting threads and structure competing ideas without erasing valid disagreements.",
            AgentRole.DOMAIN_EXPERT: "Inject deep, non-obvious domain mechanics, technical distinctions, and specialized nuances.",
            AgentRole.PRACTICAL_ENGINEER: "Ground theoretical discussions in concrete implementation trade-offs, edge cases, and execution friction.",
            AgentRole.RATIONAL_ANALYST: "Strip away rhetoric, test underlying premises with cold logic, and quantify trade-offs.",
            AgentRole.MEDIATOR: "Identify underlying causes of deadlock and reframe ideological conflict into constructive trade-off choices.",
            AgentRole.VISIONARY: "Introduce radical alternative paradigms and ambitious long-term horizons that shake up established thinking.",
            AgentRole.CONSTRAINT_PLANNER: "Highlight hidden costs, physical limits, bottleneck risks, and strict feasibility boundaries.",
            AgentRole.CONTRARIAN: "Directly challenge the dominant narrative, advocate for underrepresented counter-arguments, and expose blind spots.",
            AgentRole.EXPLORER: "Argue from a specific causal theory that the room has not yet represented, and hold it until it is seriously answered.",
            AgentRole.ORDER: "Detect structural contradictions, enforce logical consistency, and prevent circular arguments.",
            AgentRole.EVALUATOR: "Assess qualitative depth, novelty, and genuine progress, penalizing repetitive agreement.",
            AgentRole.MASTER: "Orchestrate rather than participate in the visible debate.",
        }
        return briefs.get(role, "Make a useful, role-consistent contribution with a distinct perspective.")

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
