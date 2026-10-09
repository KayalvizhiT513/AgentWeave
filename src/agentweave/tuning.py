from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from agentweave.core.enums import AgentRole


class SamplingProfile(BaseModel):
    """Per-call model settings. None means "do not send", so the API default applies."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    reasoning_effort: str | None = None
    # Plain-text speaking: the model writes only the reply; a separate hosted call does the bookkeeping.
    plain_text: bool | None = None
    # An OpenAI-compatible chat-completions server (e.g. a local MLX model). Implies plain-text speaking.
    endpoint: str | None = None

    def over(self, base: SamplingProfile) -> SamplingProfile:
        """This profile's explicit fields win; unset fields fall through to base."""
        return SamplingProfile(
            **{name: getattr(self, name) if getattr(self, name) is not None else getattr(base, name)
               for name in SamplingProfile.model_fields}
        )


class Thresholds(BaseModel):
    """Orchestration cutoffs. Defaults reproduce the values that were previously hardcoded."""

    model_config = ConfigDict(extra="forbid")

    redundancy_cutoff: float = Field(default=0.62, ge=0.0, le=1.0)
    novelty_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    depth_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    # Weight on repetition in replacement eligibility; contribution gets 1 - this.
    repetition_weight: float = Field(default=0.55, ge=0.0, le=1.0)


class TuningConfig(BaseModel):
    """Everything tunable without training: sampling per role, and orchestration thresholds."""

    model_config = ConfigDict(extra="forbid")

    default: SamplingProfile = Field(default_factory=SamplingProfile)
    roles: dict[AgentRole, SamplingProfile] = Field(default_factory=dict)
    # Used for evaluation, perspective mapping and diversity assessment.
    evaluator: SamplingProfile = Field(default_factory=SamplingProfile)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    def for_role(self, role: AgentRole) -> SamplingProfile:
        profile = self.roles.get(role)
        return profile.over(self.default) if profile else self.default

    def fingerprint(self) -> str:
        """Stable short hash of the effective config, for matching ledger rows to settings."""
        return config_hash(self.model_dump(mode="json"))


def config_hash(config: object) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
