from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class RuntimeDefaults(BaseModel):
    evaluation_interval: int = Field(default=5, ge=1)
    restructuring_interval: int = Field(default=25, ge=1)
    max_rounds: int = Field(default=20, ge=1, le=500)
    max_history_entries: int = Field(default=120, ge=10)
    summary_window: int = Field(default=8, ge=2)
    stagnation_threshold: int = Field(default=3, ge=1)


DEFAULT_RUNTIME = RuntimeDefaults()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    provider_mode: Literal["simulated", "openai"] = "simulated"
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4.1-mini"
    openai_evaluator_model: str = "gpt-4.1-mini"
    fi_api_key: SecretStr | None = None
    fi_secret_key: SecretStr | None = None
    fi_base_url: str = "https://api.futureagi.com"

    def validate_provider_configuration(self) -> None:
        if self.provider_mode == "openai" and self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY is required when PROVIDER_MODE=openai. "
                "Add it to the local .env file or your shell environment."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_provider_configuration()
    return settings
