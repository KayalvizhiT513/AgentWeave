from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentweave.core.models import RuntimeConfig
from agentweave.tuning import TuningConfig


class Topic(BaseModel):
    id: str
    topic: str
    scene: str | None = None
    constraints: list[str] = Field(default_factory=list)


class Variant(BaseModel):
    """Partial overrides, deep-merged over the experiment's shared settings and then the defaults."""

    model_config = ConfigDict(extra="forbid")

    runtime: dict[str, Any] = Field(default_factory=dict)
    tuning: dict[str, Any] = Field(default_factory=dict)


class Experiment(BaseModel):
    """
    One pre-registered question. Write the hypothesis and the primary metrics BEFORE running,
    so the ledger shows what was predicted, not what was found.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis: str
    primary_metrics: list[str]
    topics: str = "../topics.json"  # path relative to this file
    reps: int = Field(default=1, ge=1)
    runtime: dict[str, Any] = Field(default_factory=dict)  # shared by all variants
    tuning: dict[str, Any] = Field(default_factory=dict)
    baseline: str
    variants: dict[str, Variant]

    @model_validator(mode="after")
    def _check(self) -> Experiment:
        if self.baseline not in self.variants:
            raise ValueError(f"baseline '{self.baseline}' is not one of the variants")
        # Fail on typos now: RuntimeConfig would silently ignore unknown keys.
        for source in (self.runtime, *(v.runtime for v in self.variants.values())):
            unknown = set(source) - set(RuntimeConfig.model_fields)
            if unknown:
                raise ValueError(f"unknown runtime settings: {sorted(unknown)}")
        for name in self.variants:
            self.resolve(name)  # validates the tuning overrides too
        return self

    def resolve(self, variant: str) -> tuple[RuntimeConfig, TuningConfig]:
        spec = self.variants[variant]
        runtime = {**self.runtime, **spec.runtime}
        tuning = _deep_merge(self.tuning, spec.tuning)
        return RuntimeConfig(**runtime), TuningConfig(**tuning)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment(path: Path) -> tuple[Experiment, list[Topic]]:
    experiment = Experiment.model_validate_json(path.read_text())
    topics_path = (path.parent / experiment.topics).resolve()
    topics = [Topic.model_validate(item) for item in json.loads(topics_path.read_text())]
    return experiment, topics
