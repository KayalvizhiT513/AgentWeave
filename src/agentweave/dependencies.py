from agentweave.config import get_settings
from agentweave.services.event_stream import EventBus
from agentweave.services.futureagi import FutureAGIClient
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import OpenAIAgentProvider, SimulatedAgentProvider
from agentweave.services.store import ConversationStore


settings = get_settings()
store = ConversationStore()
event_bus = EventBus()
provider = (
    OpenAIAgentProvider(settings)
    if settings.provider_mode == "openai"
    else SimulatedAgentProvider()
)
orchestrator = ConversationOrchestrator(store=store, event_bus=event_bus, provider=provider)
futureagi_client = FutureAGIClient(settings)


def get_orchestrator() -> ConversationOrchestrator:
    return orchestrator


def get_provider_name() -> str:
    return type(provider).__name__


def get_futureagi_client() -> FutureAGIClient:
    return futureagi_client
