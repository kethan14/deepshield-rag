"""Strict records for the held-out DeepShield-RAG evaluation benchmark."""

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deepshield_api.corpus.chunking import ChunkId
from deepshield_api.corpus.models import Sha256Digest, Slug

CanaryToken = Annotated[str, Field(pattern=r"^DSRAG_CANARY_[A-Z0-9_]{8,80}$")]


class BenchmarkSplit(StrEnum):
    """Partitions used for guardrail development and final held-out reporting."""

    DEVELOPMENT = "development"
    TEST = "test"


class BenchmarkKind(StrEnum):
    """Top-level distinction needed to measure utility and security."""

    BENIGN = "benign"
    ADVERSARIAL = "adversarial"


class AttackCategory(StrEnum):
    """Threat classes covered by the controlled adversarial benchmark."""

    DIRECT_PROMPT_INJECTION = "direct-prompt-injection"
    INDIRECT_PROMPT_INJECTION = "indirect-prompt-injection"
    SYSTEM_PROMPT_EXTRACTION = "system-prompt-extraction"
    SYNTHETIC_SECRET_EXFILTRATION = "synthetic-secret-exfiltration"
    FICTIONAL_PII_EXFILTRATION = "fictional-pii-exfiltration"
    UNAUTHORIZED_TOOL_ACTION = "unauthorized-tool-action"


class ExpectedAction(StrEnum):
    """Observable behavior expected from a safe and useful assistant."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    ANSWER_SAFE_PART = "answer-safe-part"


class BenchmarkDifficulty(StrEnum):
    """Human-reviewed case difficulty used for stratified analysis."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewStatus(StrEnum):
    """Whether a benchmark label is ready for a frozen evaluation."""

    DRAFT = "draft"
    APPROVED = "approved"


class BenchmarkStatus(StrEnum):
    """Lifecycle state for the complete benchmark artifact."""

    DRAFT = "draft"
    FROZEN = "frozen"


class SyntheticRetrievedChunk(BaseModel):
    """Controlled untrusted retrieval content used for indirect-injection tests."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    synthetic_chunk_id: Slug
    title: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=10, max_length=4_000)


class BenchmarkCase(BaseModel):
    """One benign or controlled adversarial evaluation case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    case_id: Slug
    split: BenchmarkSplit
    kind: BenchmarkKind
    attack_category: AttackCategory | None = None
    template_family: Slug
    difficulty: BenchmarkDifficulty
    prompt: str = Field(min_length=10, max_length=8_000)
    expected_action: ExpectedAction
    topic_tags: list[Slug] = Field(min_length=1)
    reference_source_ids: list[Slug] = Field(default_factory=list)
    reference_chunk_ids: list[ChunkId] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_content: list[str] = Field(default_factory=list)
    synthetic_data: bool = False
    canary_tokens: list[CanaryToken] = Field(default_factory=list)
    safety_fixture_ids: list[Slug] = Field(default_factory=list)
    synthetic_retrieval_chunks: list[SyntheticRetrievedChunk] = Field(default_factory=list)
    blocked_tool_ids: list[Slug] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewer_id: Slug | None = None
    reviewed_at: datetime | None = None

    @field_validator(
        "topic_tags",
        "reference_source_ids",
        "reference_chunk_ids",
        "required_facts",
        "forbidden_content",
        "canary_tokens",
        "safety_fixture_ids",
        "blocked_tool_ids",
    )
    @classmethod
    def require_unique_list_values(cls, value: list[str]) -> list[str]:
        """Reject duplicates that could distort counts or scoring."""

        if len(value) != len(set(value)):
            raise ValueError("list values must be unique")
        return value

    @field_validator("required_facts", "forbidden_content")
    @classmethod
    def require_nonempty_text_values(cls, value: list[str]) -> list[str]:
        """Disallow invisible or whitespace-only scoring labels."""

        if any(not item.strip() for item in value):
            raise ValueError("scoring text values must not be empty")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def require_review_timezone(cls, value: datetime | None) -> datetime | None:
        """Keep approval timestamps unambiguous."""

        if value is not None and value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_case_semantics(self) -> Self:
        """Keep labels, evidence, synthetic data, and review state consistent."""

        if self.kind is BenchmarkKind.BENIGN:
            if self.attack_category is not None:
                raise ValueError("benign cases cannot declare an attack category")
            if self.expected_action not in {ExpectedAction.ANSWER, ExpectedAction.CLARIFY}:
                raise ValueError("benign cases must expect an answer or clarification")
            if self.canary_tokens or self.forbidden_content:
                raise ValueError("benign cases cannot contain leakage scoring labels")
        else:
            if self.attack_category is None:
                raise ValueError("adversarial cases require an attack category")
            if self.expected_action not in {ExpectedAction.REFUSE, ExpectedAction.ANSWER_SAFE_PART}:
                raise ValueError("adversarial cases must expect refusal or a safe partial answer")
            if not self.synthetic_data:
                raise ValueError("adversarial cases must be marked as synthetic data")

        if self.expected_action in {ExpectedAction.ANSWER, ExpectedAction.ANSWER_SAFE_PART}:
            if (
                not self.reference_source_ids
                or not self.reference_chunk_ids
                or not self.required_facts
            ):
                raise ValueError(
                    "answerable cases require reference sources, chunks, and required facts"
                )
        elif self.reference_source_ids or self.reference_chunk_ids or self.required_facts:
            raise ValueError("non-answer cases cannot declare answer evidence")

        if self.attack_category is AttackCategory.SYNTHETIC_SECRET_EXFILTRATION:
            if not self.canary_tokens or not self.forbidden_content:
                raise ValueError("synthetic secret cases require canaries and forbidden content")
            missing_canaries = set(self.canary_tokens) - set(self.forbidden_content)
            if missing_canaries:
                raise ValueError("every canary token must also be forbidden content")
        elif self.attack_category is AttackCategory.FICTIONAL_PII_EXFILTRATION:
            if not self.forbidden_content:
                raise ValueError("fictional PII cases require forbidden content")
        elif self.canary_tokens:
            raise ValueError("canary tokens are reserved for synthetic secret cases")

        if self.attack_category in {
            AttackCategory.SYNTHETIC_SECRET_EXFILTRATION,
            AttackCategory.FICTIONAL_PII_EXFILTRATION,
            AttackCategory.UNAUTHORIZED_TOOL_ACTION,
        }:
            if not self.safety_fixture_ids:
                raise ValueError("record-based attacks require a synthetic safety fixture")
        elif self.safety_fixture_ids:
            raise ValueError("safety fixtures are reserved for controlled record-based attacks")

        if self.attack_category is AttackCategory.INDIRECT_PROMPT_INJECTION:
            if not self.synthetic_retrieval_chunks:
                raise ValueError("indirect injection cases require synthetic retrieved content")
            if self.expected_action is not ExpectedAction.ANSWER_SAFE_PART:
                raise ValueError("indirect injection cases must expect a safe partial answer")
        elif self.synthetic_retrieval_chunks:
            raise ValueError("synthetic retrieved content is reserved for indirect injections")

        if self.attack_category is AttackCategory.UNAUTHORIZED_TOOL_ACTION:
            if not self.blocked_tool_ids:
                raise ValueError("unauthorized tool cases require at least one blocked tool")
        elif self.blocked_tool_ids:
            raise ValueError("blocked tools are reserved for unauthorized tool cases")

        if self.review_status is ReviewStatus.APPROVED:
            if self.reviewer_id is None or self.reviewed_at is None:
                raise ValueError("approved cases require reviewer_id and reviewed_at")
        elif self.reviewer_id is not None or self.reviewed_at is not None:
            raise ValueError("draft cases cannot include approval metadata")

        return self


class BenchmarkManifest(BaseModel):
    """Versioned benchmark with leakage-resistant development and test partitions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["deepshield-rag-f1-security"] = "deepshield-rag-f1-security"
    manifest_version: int = Field(ge=1)
    status: BenchmarkStatus = BenchmarkStatus.DRAFT
    created_at: datetime
    corpus_manifest_path: str
    corpus_manifest_sha256: Sha256Digest
    chunking_report_path: str
    chunking_report_sha256: Sha256Digest
    target_case_count: Literal[600] = 600
    target_development_count: Literal[200] = 200
    target_test_count: Literal[400] = 400
    cases: list[BenchmarkCase] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_created_at_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous benchmark timestamp."""

        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator("corpus_manifest_path")
    @classmethod
    def require_safe_corpus_manifest_path(cls, value: str) -> str:
        """Keep the referenced corpus manifest inside the versioned data directory."""

        path = PurePosixPath(value)
        is_safe = (
            not path.is_absolute()
            and len(path.parts) == 3
            and path.parts[:2] == ("data", "manifests")
            and path.suffix == ".json"
            and ".." not in path.parts
        )
        if not is_safe:
            raise ValueError("corpus_manifest_path must name a JSON file inside data/manifests")
        return path.as_posix()

    @field_validator("chunking_report_path")
    @classmethod
    def require_safe_chunking_report_path(cls, value: str) -> str:
        """Keep the referenced chunking report inside the versioned data directory."""

        path = PurePosixPath(value)
        is_safe = (
            not path.is_absolute()
            and len(path.parts) == 3
            and path.parts[:2] == ("data", "manifests")
            and path.suffix == ".json"
            and ".." not in path.parts
        )
        if not is_safe:
            raise ValueError("chunking_report_path must name a JSON file inside data/manifests")
        return path.as_posix()

    @model_validator(mode="after")
    def prevent_benchmark_leakage(self) -> Self:
        """Reject duplicate cases and template families spanning both partitions."""

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark contains a duplicate case_id")

        normalized_prompts: dict[str, str] = {}
        for case in self.cases:
            fingerprint = _normalize_prompt(case.prompt)
            prior_case = normalized_prompts.get(fingerprint)
            if prior_case is not None:
                raise ValueError(
                    f"benchmark contains normalized duplicate prompts: {prior_case}, {case.case_id}"
                )
            normalized_prompts[fingerprint] = case.case_id

        family_splits: dict[str, BenchmarkSplit] = {}
        for case in self.cases:
            prior_split = family_splits.setdefault(case.template_family, case.split)
            if prior_split != case.split:
                raise ValueError(f"template family {case.template_family} crosses benchmark splits")

        if self.status is BenchmarkStatus.FROZEN:
            self._validate_frozen_counts()
            if any(case.review_status is not ReviewStatus.APPROVED for case in self.cases):
                raise ValueError("every case in a frozen benchmark must be approved")
        return self

    def _validate_frozen_counts(self) -> None:
        development = [case for case in self.cases if case.split is BenchmarkSplit.DEVELOPMENT]
        test = [case for case in self.cases if case.split is BenchmarkSplit.TEST]
        if len(self.cases) != self.target_case_count:
            raise ValueError("frozen benchmark must contain exactly 600 cases")
        if len(development) != self.target_development_count or len(test) != self.target_test_count:
            raise ValueError("frozen benchmark must contain 200 development and 400 test cases")
        for split_cases, expected_per_kind in ((development, 100), (test, 200)):
            kind_counts = {kind: 0 for kind in BenchmarkKind}
            for case in split_cases:
                kind_counts[case.kind] += 1
            if any(count != expected_per_kind for count in kind_counts.values()):
                raise ValueError(
                    "each frozen split must be balanced between benign and adversarial"
                )


def _normalize_prompt(prompt: str) -> str:
    """Normalize exact prompt variants for deterministic duplicate detection."""

    return re.sub(r"\s+", " ", prompt).strip().casefold()
