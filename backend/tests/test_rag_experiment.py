from pathlib import Path

import pytest
from pydantic import ValidationError

from deepshield_api.rag.experiment import (
    FixedGenerationConfiguration,
    load_fixed_generation_configuration,
    render_fixed_generation_configuration,
)


def test_committed_generation_configuration_is_canonical_and_frozen() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / "config/experiments/generation-model.v1.json"
    configuration = load_fixed_generation_configuration(path)

    assert configuration.model == "gpt-5.4-mini-2026-03-17"
    assert str(configuration.base_url) == "https://api.openai.com/v1"
    assert configuration.configurations == ["C0", "C1", "C2"]
    assert configuration.reasoning_effort == "none"
    assert configuration.temperature == 0
    assert configuration.max_completion_tokens == 700
    assert configuration.input_usd_per_million_tokens == 0.75
    assert configuration.output_usd_per_million_tokens == 4.5
    assert path.read_text(encoding="utf-8") == render_fixed_generation_configuration(configuration)


def test_generation_configuration_rejects_unpinned_alias() -> None:
    with pytest.raises(ValidationError, match="gpt-5.4-mini-2026-03-17"):
        FixedGenerationConfiguration.model_validate(
            {
                "selected_on": "2026-10-02",
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "chat-completions",
                "model": "gpt-5.4-mini",
                "reasoning_effort": "none",
                "temperature": 0,
                "max_completion_tokens": 700,
                "configurations": ["C0", "C1", "C2"],
                "pricing_as_of": "2026-10-02",
                "input_usd_per_million_tokens": 0.75,
                "output_usd_per_million_tokens": 4.5,
            }
        )


def test_generation_configuration_requires_comparison_order() -> None:
    with pytest.raises(ValidationError, match="comparison order"):
        FixedGenerationConfiguration.model_validate(
            {
                "selected_on": "2026-10-02",
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "chat-completions",
                "model": "gpt-5.4-mini-2026-03-17",
                "reasoning_effort": "none",
                "temperature": 0,
                "max_completion_tokens": 700,
                "configurations": ["C0", "C2", "C1"],
                "pricing_as_of": "2026-10-02",
                "input_usd_per_million_tokens": 0.75,
                "output_usd_per_million_tokens": 4.5,
            }
        )
