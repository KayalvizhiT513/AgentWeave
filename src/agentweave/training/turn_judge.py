"""Pairwise judge for one debate turn: which of two candidate replies is the better next move?"""

from __future__ import annotations

from dataclasses import dataclass

from agentweave.core.models import ConversationUsage
from agentweave.services.provider import OpenAIAgentProvider

# Bump when the rubric changes: preferences from different versions are not comparable.
TURN_JUDGE_VERSION = "t1"

_INSTRUCTIONS = (
    "You compare two candidate next turns for the same participant in a multi-agent discussion. "
    "Pick the reply that is the better move for that participant's job. A better reply: (1) answers something "
    "specific that was just said, not a generic version of it; (2) exposes a real gap, assumption or "
    "consequence, or adds a point nobody has made; (3) holds a coherent position instead of opposing for its "
    "own sake; (4) sounds like natural speech from one person. Penalize restating earlier points, vague "
    "contrarianism and filler. Do not reward length. Give a one-sentence rationale first, then the winner."
)
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rationale": {"type": "string"}, "winner": {"type": "string", "enum": ["first", "second"]}},
    "required": ["rationale", "winner"],
}


@dataclass(slots=True)
class TurnContext:
    topic: str
    scene: str | None
    role: str
    role_brief: str
    history: list[str]


def _render(context: TurnContext, first: str, second: str) -> str:
    recent = "\n".join(f"- {line}" for line in context.history[-6:]) or "- (start of discussion)"
    return (
        f"Topic: {context.topic}\nScene: {context.scene or 'none'}\n"
        f"Participant's role: {context.role} ({context.role_brief})\n"
        f"Recent dialogue:\n{recent}\n\n"
        f"First candidate reply:\n{first}\n\nSecond candidate reply:\n{second}"
    )


class TurnJudge:
    version = TURN_JUDGE_VERSION

    def __init__(self, provider: OpenAIAgentProvider, usage: ConversationUsage) -> None:
        self._provider, self._usage = provider, usage
        self.model = provider.settings.openai_evaluator_model

    async def _once(self, context: TurnContext, first: str, second: str) -> str:
        data = await self._provider.structured_output(
            instructions=_INSTRUCTIONS,
            user_input=_render(context, first, second),
            schema_name="turn_judge",
            schema=_SCHEMA,
            usage=self._usage,
        )
        return str(data["winner"])

    async def prefer(self, context: TurnContext, a: str, b: str) -> int | None:
        """0 if `a` wins, 1 if `b` wins, None if the two orders disagree (position noise, not preference)."""
        forward = await self._once(context, a, b)
        backward = await self._once(context, b, a)
        picked_a_forward = forward == "first"
        picked_a_backward = backward == "second"
        if picked_a_forward != picked_a_backward:
            return None
        return 0 if picked_a_forward else 1
