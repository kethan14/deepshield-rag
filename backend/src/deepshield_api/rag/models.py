"""Validated request, response, and evidence models for the C0 RAG pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

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
