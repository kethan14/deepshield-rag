"""Frozen generation-model configuration for reproducible C0/C1/C2 experiments."""

import json
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator


class ExperimentConfigurationError(RuntimeError):
    """Raised when the frozen generation configuration cannot be loaded."""


class FixedGenerationConfiguration(BaseModel):
    """Versioned model identity and inference settings shared by all configurations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    configuration_id: Literal["generation-model-v1"] = "generation-model-v1"
    selected_on: date
    provider: Literal["openai"]
    base_url: HttpUrl
    endpoint: Literal["chat-completions"]
    model: Literal["gpt-5.4-mini-2026-03-17"]
    reasoning_effort: Literal["none"]
    temperature: Literal[0]
    max_completion_tokens: int = Field(ge=1, le=4_096)
    configurations: list[Literal["C0", "C1", "C2"]]
    pricing_as_of: date
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)

    @model_validator(mode="after")
    def require_all_comparison_configurations(self) -> Self:
        if self.configurations != ["C0", "C1", "C2"]:
            raise ValueError("configurations must contain C0, C1, and C2 in comparison order")
        return self


def load_fixed_generation_configuration(path: Path) -> FixedGenerationConfiguration:
    """Load one strict UTF-8 generation configuration."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentConfigurationError(
            f"unable to read generation configuration: {path}"
        ) from exc
    try:
        return FixedGenerationConfiguration.model_validate_json(content)
    except ValidationError as exc:
        raise ExperimentConfigurationError(
            f"generation configuration validation failed:\n{exc}"
        ) from exc


def render_fixed_generation_configuration(config: FixedGenerationConfiguration) -> str:
    """Render canonical deterministic configuration JSON."""

    return json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
