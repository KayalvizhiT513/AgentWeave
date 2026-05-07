from enum import Enum


class ConversationStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class AgentRole(str, Enum):
    MASTER = "master"
    CHATTER = "chatter"
    LISTENER = "listener"
    ORDER = "order"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    MODERATOR = "moderator"
    EVALUATOR = "evaluator"
    DOMAIN_EXPERT = "domain_expert"
    PRACTICAL_ENGINEER = "practical_engineer"
    RATIONAL_ANALYST = "rational_analyst"
    MEDIATOR = "mediator"
    VISIONARY = "visionary"
    CONSTRAINT_PLANNER = "constraint_planner"
    CONTRARIAN = "contrarian"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    REPLACED = "replaced"
    INACTIVE = "inactive"


class EvaluationRecommendation(str, Enum):
    CONTINUE = "continue"
    REPLACE = "replace"
    RESTRUCTURE = "restructure"
    STOP = "stop"


class EventType(str, Enum):
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_STARTED = "conversation.started"
    ROUND_COMPLETED = "conversation.round_completed"
    EVALUATION_CREATED = "conversation.evaluation_created"
    AGENT_REPLACED = "conversation.agent_replaced"
    SUMMARY_UPDATED = "conversation.summary_updated"
    CONVERSATION_COMPLETED = "conversation.completed"
    CONVERSATION_STOPPED = "conversation.stopped"
