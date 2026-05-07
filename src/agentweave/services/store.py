from __future__ import annotations

import asyncio

from agentweave.core.models import Conversation


class ConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def create(self, conversation: Conversation) -> Conversation:
        async with self._lock:
            self._conversations[conversation.id] = conversation
            return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    async def list(self) -> list[Conversation]:
        return sorted(
            self._conversations.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )

    async def update(self, conversation: Conversation) -> Conversation:
        async with self._lock:
            self._conversations[conversation.id] = conversation
            return conversation
