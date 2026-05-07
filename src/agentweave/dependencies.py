from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentProvider
from agentweave.services.store import ConversationStore


store = ConversationStore()
event_bus = EventBus()
provider = AgentProvider()
orchestrator = ConversationOrchestrator(store=store, event_bus=event_bus, provider=provider)


def get_orchestrator() -> ConversationOrchestrator:
    return orchestrator
