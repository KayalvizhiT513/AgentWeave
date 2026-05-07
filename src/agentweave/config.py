from pydantic import BaseModel, Field


class RuntimeDefaults(BaseModel):
    evaluation_interval: int = Field(default=5, ge=1)
    restructuring_interval: int = Field(default=25, ge=1)
    max_rounds: int = Field(default=20, ge=1, le=500)
    max_history_entries: int = Field(default=120, ge=10)
    summary_window: int = Field(default=8, ge=2)
    stagnation_threshold: int = Field(default=3, ge=1)


DEFAULT_RUNTIME = RuntimeDefaults()
