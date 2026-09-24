"""Validated metadata models for the versioned F-1 policy corpus."""

from datetime import date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationInfo,
    field_validator,
    model_validator,
)

Slug = Annotated[
    str,
    Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
Sha256Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class SourcePublisher(StrEnum):
    """Approved publishers for the primary F-1 policy corpus."""

    USCIS = "uscis"
    DHS = "dhs"
    ICE_SEVP = "ice-sevp"
    STUDY_IN_THE_STATES = "study-in-the-states"
    ECFR = "ecfr"
    DEPARTMENT_OF_STATE = "department-of-state"
    CBP = "cbp"


class SourceDocumentType(StrEnum):
    """Semantic document categories used during collection and evaluation."""

    GUIDANCE_PAGE = "guidance-page"
    REGULATION = "regulation"
    FORM_INSTRUCTIONS = "form-instructions"
    POLICY_MANUAL = "policy-manual"
    PDF_GUIDE = "pdf-guide"


class SourceMediaType(StrEnum):
    """Supported source-file media types."""

    HTML = "text/html"
    PDF = "application/pdf"
    PLAIN_TEXT = "text/plain"


class CollectionStatus(StrEnum):
    """Lifecycle states permitted in a source manifest."""

    PLANNED = "planned"
    COLLECTED = "collected"


PUBLISHER_DOMAINS: dict[SourcePublisher, frozenset[str]] = {
    SourcePublisher.USCIS: frozenset({"uscis.gov"}),
    SourcePublisher.DHS: frozenset({"dhs.gov"}),
    SourcePublisher.ICE_SEVP: frozenset({"ice.gov"}),
    SourcePublisher.STUDY_IN_THE_STATES: frozenset({"studyinthestates.dhs.gov"}),
    SourcePublisher.ECFR: frozenset({"ecfr.gov"}),
    SourcePublisher.DEPARTMENT_OF_STATE: frozenset({"state.gov"}),
    SourcePublisher.CBP: frozenset({"cbp.gov"}),
}


class SourceRecord(BaseModel):
    """Metadata for one immutable version of an authoritative source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    source_id: Slug
    document_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=300)
    publisher: SourcePublisher
    canonical_url: HttpUrl
    document_type: SourceDocumentType
    media_type: SourceMediaType
    scope_tags: list[Slug] = Field(min_length=1)
    rights_notes: str = Field(min_length=1, max_length=1000)
    language: Literal["en"] = "en"
    collection_status: CollectionStatus = CollectionStatus.PLANNED
    published_on: date | None = None
    last_updated_on: date | None = None
    retrieved_at: datetime | None = None
    raw_path: str | None = None
    sha256: Sha256Digest | None = None

    @field_validator("canonical_url")
    @classmethod
    def require_https_url(cls, value: HttpUrl, info: ValidationInfo) -> HttpUrl:
        """Require HTTPS URLs on the selected publisher's official domain."""

        if value.scheme != "https":
            raise ValueError("canonical_url must use HTTPS")
        if value.username is not None or value.password is not None:
            raise ValueError("canonical_url must not contain credentials")
        if value.port != 443:
            raise ValueError("canonical_url must use the standard HTTPS port")
        if value.fragment is not None:
            raise ValueError("canonical_url must not contain a URL fragment")

        publisher = info.data.get("publisher")
        if isinstance(publisher, SourcePublisher):
            allowed_domains = PUBLISHER_DOMAINS[publisher]
            hostname = value.host
            is_allowed = hostname is not None and any(
                hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains
            )
            if not is_allowed:
                raise ValueError(
                    f"canonical_url host is not approved for publisher {publisher.value}"
                )
        return value

    @field_validator("scope_tags")
    @classmethod
    def require_unique_scope_tags(cls, value: list[str]) -> list[str]:
        """Reject duplicated tags and store them deterministically."""

        if len(value) != len(set(value)):
            raise ValueError("scope_tags must not contain duplicates")
        return sorted(value)

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Require an unambiguous retrieval timestamp."""

        if value is not None and value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @field_validator("raw_path")
    @classmethod
    def require_safe_raw_path(cls, value: str | None) -> str | None:
        """Keep portable raw paths inside the intended data directory."""

        if value is None:
            return None

        path = PurePosixPath(value)
        is_safe = (
            not path.is_absolute()
            and len(path.parts) >= 3
            and path.parts[:2] == ("data", "raw")
            and ".." not in path.parts
        )
        if not is_safe:
            raise ValueError("raw_path must be a relative POSIX path inside data/raw")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_record_state(self) -> Self:
        """Keep lifecycle state and collection artifacts consistent."""

        if (
            self.published_on is not None
            and self.last_updated_on is not None
            and self.last_updated_on < self.published_on
        ):
            raise ValueError("last_updated_on cannot be earlier than published_on")

        artifacts = {
            "raw_path": self.raw_path,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
        }

        if self.collection_status is CollectionStatus.COLLECTED:
            missing = sorted(name for name, value in artifacts.items() if value is None)
            if missing:
                raise ValueError("collected sources require these fields: " + ", ".join(missing))
        elif any(value is not None for value in artifacts.values()):
            raise ValueError("planned sources cannot include collection artifacts")

        return self


class SourceManifest(BaseModel):
    """Versioned collection of authoritative corpus source records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: Literal["deepshield-rag-f1-policy"] = "deepshield-rag-f1-policy"
    manifest_version: int = Field(ge=1)
    created_at: datetime
    sources: list[SourceRecord] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_created_at_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous manifest timestamp."""

        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_unique_source_versions(self) -> Self:
        """Prevent duplicate source-version pairs within one manifest."""

        source_versions = [(source.source_id, source.document_version) for source in self.sources]
        if len(source_versions) != len(set(source_versions)):
            raise ValueError("manifest contains a duplicate source version")
        return self
