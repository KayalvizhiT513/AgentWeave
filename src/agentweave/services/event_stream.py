from __future__ import annotations

import asyncio
from collections import defaultdict

from agentweave.core.models import ConversationEvent


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[ConversationEvent]]] = defaultdict(list)

    def subscribe(self, conversation_id: str) -> asyncio.Queue[ConversationEvent]:
        queue: asyncio.Queue[ConversationEvent] = asyncio.Queue()
        self._queues[conversation_id].append(queue)
        return queue

    def unsubscribe(
        self,
        conversation_id: str,
        queue: asyncio.Queue[ConversationEvent],
    ) -> None:
        subscribers = self._queues.get(conversation_id, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers and conversation_id in self._queues:
            del self._queues[conversation_id]

    async def publish(self, event: ConversationEvent) -> None:
        for queue in self._queues.get(event.conversation_id, []):
            await queue.put(event)
