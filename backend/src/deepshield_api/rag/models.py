"""Validated request, response, and evidence models for comparable RAG configurations."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from deepshield_api.corpus.models import Slug, SourcePublisher


class BaselineQuestion(BaseModel):
    """One user question submitted to the baseline policy assistant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedPassage(BaseModel):
    """One ranked official passage supplied to the answer generator."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    rank: int = Field(ge=1)
    score: float = Field(ge=-1.000001, le=1.000001)
    chunk_id: str = Field(min_length=1)
    source_id: Slug
    title: str = Field(min_length=1)
    publisher: SourcePublisher
    canonical_url: HttpUrl
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)


class GenerationUsage(BaseModel):
    """Provider-reported token usage when the endpoint supplies it."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class BaselineAnswer(BaseModel):
    """C0 answer plus complete retrieval and citation evidence."""

    model_config = ConfigDict(extra="forbid")

    configuration: Literal["C0"] = "C0"
    prompt_version: Literal["c0-v1"] = "c0-v1"
    question: str
    answer: str = Field(min_length=1)
    cited_reference_ids: list[str]
    invalid_reference_ids: list[str]
    retrieved_passages: list[RetrievedPassage] = Field(min_length=1)
    generator_model: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: GenerationUsage


class GuardrailStage(StrEnum):
    """Layered checkpoints that can affect one request."""

    INPUT = "input"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"
    AUTHORIZATION = "authorization"


class GuardrailAction(StrEnum):
    """Observable action taken by a deterministic guardrail."""

    ALLOW = "allow"
    WARN = "warn"
    EXCLUDE = "exclude"
    BLOCK = "block"


class GuardrailCategory(StrEnum):
    """Sanitized risk categories emitted by deterministic guardrails."""

    NONE = "none"
    CONTROL_CHARACTERS = "control-characters"
    DIRECT_PROMPT_INJECTION = "direct-prompt-injection"
    ROLE_MANIPULATION = "role-manipulation"
    SYSTEM_PROMPT_EXTRACTION = "system-prompt-extraction"
    SENSITIVE_DATA_REQUEST = "sensitive-data-request"
    UNAUTHORIZED_ACTION_REQUEST = "unauthorized-action-request"
    MISSING_CITATION = "missing-citation"
    INVALID_CITATION = "invalid-citation"
    SENSITIVE_OUTPUT = "sensitive-output"
    SYNTHETIC_SECRET_LEAKAGE = "synthetic-secret-leakage"
    FICTIONAL_PII_LEAKAGE = "fictional-pii-leakage"
    UNKNOWN_MOCK_TOOL = "unknown-mock-tool"
    UNKNOWN_SYNTHETIC_FIXTURE = "unknown-synthetic-fixture"
    MISSING_TOOL_SCOPE = "missing-tool-scope"
    UNAUTHORIZED_SYNTHETIC_FIXTURE = "unauthorized-synthetic-fixture"
    SIDE_EFFECT_CONFIRMATION_REQUIRED = "side-effect-confirmation-required"
    AUTHORIZED_MOCK_TOOL = "authorized-mock-tool"
    CLASSIFIER_INPUT_RISK = "classifier-input-risk"
    CLASSIFIER_RETRIEVAL_RISK = "classifier-retrieval-risk"


class GuardrailEvent(BaseModel):
    """Sanitized evidence explaining one guardrail decision."""

    model_config = ConfigDict(extra="forbid")

    stage: GuardrailStage
    action: GuardrailAction
    category: GuardrailCategory
    code: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    detail: str = Field(min_length=1, max_length=300)
    chunk_id: str | None = None


class HardenedQuestion(BaselineQuestion):
    """One question submitted to the standard-defense C1 pipeline."""


class HardenedAnswer(BaseModel):
    """C1 result with security decisions and complete safe evidence."""

    model_config = ConfigDict(extra="forbid")

    configuration: Literal["C1"] = "C1"
    prompt_version: Literal["c1-v1"] = "c1-v1"
    guardrail_version: Literal["c1-v1"] = "c1-v1"
    outcome: Literal["answered", "refused"]
    question: str
    answer: str = Field(min_length=1)
    cited_reference_ids: list[str]
    invalid_reference_ids: list[str]
    retrieved_passages: list[RetrievedPassage]
    excluded_chunk_ids: list[str]
    guardrail_events: list[GuardrailEvent] = Field(min_length=1)
    generated_output_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    generator_model: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: GenerationUsage


class ClassifierRiskAssessment(BaseModel):
    """Sanitized C2 classifier evidence without retaining the assessed text."""

    model_config = ConfigDict(extra="forbid")

    stage: GuardrailStage
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_id: str | None = None
    risk_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    risky: bool
    model_id: Literal["deepshield-c2-risk-classifier"] = "deepshield-c2-risk-classifier"
    model_version: int = Field(ge=1)

    @model_validator(mode="after")
    def require_supported_stage_and_consistent_decision(self) -> Self:
        if self.stage not in {GuardrailStage.INPUT, GuardrailStage.RETRIEVAL}:
            raise ValueError("classifier assessments support only input and retrieval stages")
        if (self.stage is GuardrailStage.RETRIEVAL) != (self.chunk_id is not None):
            raise ValueError("only retrieval assessments require a chunk_id")
        if self.risky != (self.risk_score >= self.threshold):
            raise ValueError("classifier decision must match score and threshold")
        return self


class DeepShieldQuestion(BaselineQuestion):
    """One question submitted to the complete C2 assistant pipeline."""


class DeepShieldAnswer(BaseModel):
    """C2 result with classifier, deterministic guardrail, and citation evidence."""

    model_config = ConfigDict(extra="forbid")

    configuration: Literal["C2"] = "C2"
    prompt_version: Literal["c2-v1"] = "c2-v1"
    guardrail_version: Literal["c2-v1"] = "c2-v1"
    outcome: Literal["answered", "refused"]
    question: str
    answer: str = Field(min_length=1)
    cited_reference_ids: list[str]
    invalid_reference_ids: list[str]
    retrieved_passages: list[RetrievedPassage]
    excluded_chunk_ids: list[str]
    guardrail_events: list[GuardrailEvent] = Field(min_length=1)
    risk_assessments: list[ClassifierRiskAssessment] = Field(min_length=1)
    generated_output_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    generator_model: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: GenerationUsage
