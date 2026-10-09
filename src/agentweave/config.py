from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentweave.core.enums import AgentRole, SpeakerProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class RuntimeDefaults(BaseModel):
    evaluation_interval: int = Field(default=2, ge=1)
    restructuring_interval: int = Field(default=25, ge=1)
    max_rounds: int = Field(default=5, ge=1, le=500)
    max_history_entries: int = Field(default=120, ge=10)
    summary_window: int = Field(default=8, ge=2)
    stagnation_threshold: int = Field(default=3, ge=1)


DEFAULT_RUNTIME = RuntimeDefaults()

# Which vendor voices each persona, by each model family's general reputation, so personas differ in
# training lineage and not only in prompt. A role missing here, or whose key is unset, uses OpenAI.
ROLE_PROVIDERS: dict[AgentRole, SpeakerProvider] = {
    # OpenAI: deliberate, step-by-step reasoning and depth.
    AgentRole.CRITIC: SpeakerProvider.OPENAI,
    AgentRole.RATIONAL_ANALYST: SpeakerProvider.OPENAI,
    AgentRole.DOMAIN_EXPERT: SpeakerProvider.OPENAI,
    AgentRole.CONSTRAINT_PLANNER: SpeakerProvider.OPENAI,
    AgentRole.ORDER: SpeakerProvider.OPENAI,
    # Claude: natural, nuanced conversation and synthesis.
    AgentRole.CHATTER: SpeakerProvider.CLAUDE,
    AgentRole.VISIONARY: SpeakerProvider.CLAUDE,
    AgentRole.MEDIATOR: SpeakerProvider.CLAUDE,
    AgentRole.SYNTHESIZER: SpeakerProvider.CLAUDE,
    AgentRole.LISTENER: SpeakerProvider.CLAUDE,
    AgentRole.MODERATOR: SpeakerProvider.CLAUDE,
    # Perplexity: search-grounded, real-world evidence.
    AgentRole.EXPLORER: SpeakerProvider.PERPLEXITY,
    AgentRole.CONTRARIAN: SpeakerProvider.PERPLEXITY,
    AgentRole.PRACTICAL_ENGINEER: SpeakerProvider.PERPLEXITY,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    provider_mode: str = "openai"
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-5.4-mini"
    openai_evaluator_model: str = "gpt-5.4-mini"
    claude_api_key: SecretStr | None = None
    claude_base_url: str = "https://api.anthropic.com/v1"
    claude_model: str = "claude-sonnet-5-5"
    perplexity_api_key: SecretStr | None = None
    perplexity_base_url: str = "https://api.perplexity.ai"
    perplexity_model: str = "sonar-pro"
    fi_api_key: SecretStr | None = None
    fi_secret_key: SecretStr | None = None
    fi_base_url: str = "https://api.futureagi.com"

    def speaker_provider(self, role: AgentRole) -> SpeakerProvider:
        """The vendor that voices this role, falling back to OpenAI when its key is not configured."""
        choice = ROLE_PROVIDERS.get(role, SpeakerProvider.OPENAI)
        keys = {SpeakerProvider.CLAUDE: self.claude_api_key, SpeakerProvider.PERPLEXITY: self.perplexity_api_key}
        key = keys.get(choice)
        if choice != SpeakerProvider.OPENAI and (key is None or not key.get_secret_value().strip()):
            return SpeakerProvider.OPENAI
        return choice

    def validate_provider_configuration(self) -> None:
        if self.provider_mode != "openai":
            raise ValueError("Only PROVIDER_MODE=openai is supported.")
        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Add it to the local .env file or your shell environment."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_provider_configuration()
    return settings
