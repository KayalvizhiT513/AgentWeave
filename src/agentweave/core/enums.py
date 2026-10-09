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
    EXPLORER = "explorer"


class SpeakerProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    PERPLEXITY = "perplexity"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    REPLACED = "replaced"
    INACTIVE = "inactive"


class EvaluationRecommendation(str, Enum):
    CONTINUE = "continue"
    REPLACE = "replace"
    RESTRUCTURE = "restructure"
    STOP = "stop"


class DiversityGapType(str, Enum):
    NONE = "none"
    MISSING_DIMENSION = "missing_dimension"
    SHARED_ASSUMPTION = "shared_assumption"
    CONVERGED_MODELS = "converged_models"


class ReplacementPressure(str, Enum):
    QUALITY = "quality"
    EXPLORATION = "exploration"


class EventType(str, Enum):
    CONVERSATION_CREATED = "conversation.created"
    PERSPECTIVE_MAP_CREATED = "conversation.perspective_map_created"
    CONVERSATION_STARTED = "conversation.started"
    AGENT_RESPONDED = "conversation.agent_responded"
    ROUND_COMPLETED = "conversation.round_completed"
    EVALUATION_CREATED = "conversation.evaluation_created"
    DIVERSITY_ASSESSED = "conversation.diversity_assessed"
    AGENT_REPLACED = "conversation.agent_replaced"
    SUMMARY_UPDATED = "conversation.summary_updated"
    CONVERSATION_COMPLETED = "conversation.completed"
    CONVERSATION_STOPPED = "conversation.stopped"
