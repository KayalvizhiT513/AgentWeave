"""Independent transcript judge. Separate from the in-loop evaluator so the system does not grade itself."""

from __future__ import annotations

from typing import Any, Protocol

from agentweave.core.models import Conversation, ConversationUsage
from agentweave.services.provider import OpenAIAgentProvider
from agentweave.tuning import SamplingProfile

# Bump when the rubric or prompt changes: scores from different versions are not comparable.
JUDGE_VERSION = "v1"

CRITERIA = ("novelty", "depth", "disagreement_preserved")

_INSTRUCTIONS = (
    "You are a blind judge of a multi-agent discussion transcript. Score each criterion as an integer 1-5.\n"
    "novelty: 1 = participants restate the obvious or each other; 3 = a few non-obvious points; "
    "5 = several ideas a thoughtful reader would not have predicted from the topic.\n"
    "depth: 1 = assertions without reasons; 3 = some causal reasoning or evidence; "
    "5 = mechanisms, edge cases and tradeoffs worked through.\n"
    "disagreement_preserved: 1 = participants converge on one view or fake agreement; "
    "3 = some real disagreements remain visible; 5 = distinct positions are kept, sharpened and their "
    "points of conflict are explicit.\n"
    "Judge only the text. Ignore style, length and speaker names. Return structured JSON only."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{name: {"type": "integer", "minimum": 1, "maximum": 5} for name in CRITERIA},
        "rationale": {"type": "string"},
    },
    "required": [*CRITERIA, "rationale"],
}


class Judge(Protocol):
    version: str
    model: str

    async def score(self, conversation: Conversation, usage: ConversationUsage) -> dict: ...


def render_transcript(conversation: Conversation) -> str:
    speakers: dict[str, str] = {}
    lines = []
    for turn in conversation.exchanges:
        label = speakers.setdefault(turn.agent_id, f"Speaker {len(speakers) + 1}")
        lines.append(f"{label}: {turn.content}")
    return f"Topic: {conversation.topic}\n\nTranscript:\n" + "\n".join(lines)


class OpenAIJudge:
    version = JUDGE_VERSION

    def __init__(self, provider: OpenAIAgentProvider, sampling: SamplingProfile | None = None) -> None:
        self._provider = provider
        self._sampling = sampling or SamplingProfile()
        self.model = self._sampling.model or provider.settings.openai_evaluator_model

    async def score(self, conversation: Conversation, usage: ConversationUsage) -> dict:
        data = await self._provider.structured_output(
            instructions=_INSTRUCTIONS,
            user_input=render_transcript(conversation),
            schema_name="judge_transcript",
            schema=_SCHEMA,
            sampling=self._sampling,
            usage=usage,
        )
        scores = {f"judge_{name}": int(data[name]) for name in CRITERIA}
        scores["judge_rationale"] = str(data["rationale"])
        return scores
