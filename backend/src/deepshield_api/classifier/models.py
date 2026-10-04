"""Strict provenance and example contracts for the C2 risk classifier."""

import hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from deepshield_api.corpus.models import Sha256Digest, Slug


class ArtifactStatus(StrEnum):
    """Lifecycle state for classifier data artifacts."""

    DRAFT = "draft"
    FROZEN = "frozen"


class SourcePurpose(StrEnum):
    """Permitted use of one external source in this project."""

    DEVELOPMENT = "development"
    EXTERNAL_TEST = "external-test"


class GroupingStrategy(StrEnum):
    """How related examples are kept together to prevent split leakage."""

    UPSTREAM_COLUMN = "upstream-column"
    DERIVED_NEAR_DUPLICATE_CLUSTER = "derived-near-duplicate-cluster"
    EXTERNAL_SOURCE_SPLIT = "external-source-split"


class RiskSplit(StrEnum):
    """Leakage-resistant classifier-data partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXTERNAL_TEST = "external-test"


class RiskLabel(StrEnum):
    """Binary decision learned by the lightweight classifier."""

    BENIGN = "benign"
    ADVERSARIAL = "adversarial"


class RiskCategory(StrEnum):
    """Fine-grained research labels retained for error analysis."""

    BENIGN = "benign"
    DIRECT_PROMPT_INJECTION = "direct-prompt-injection"
    ROLE_MANIPULATION = "role-manipulation"
    SYSTEM_PROMPT_EXTRACTION = "system-prompt-extraction"
    SENSITIVE_DATA_EXFILTRATION = "sensitive-data-exfiltration"
    UNAUTHORIZED_TOOL_ACTION = "unauthorized-tool-action"
    OTHER_ADVERSARIAL = "other-adversarial"


class ReviewStatus(StrEnum):
    """Human-review state for derived or synthetic classifier examples."""

    DRAFT = "draft"
    APPROVED = "approved"


class NormalizationRules(BaseModel):
    """Frozen text and split rules used to derive classifier candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rules_version: Literal["1.0"] = "1.0"
    unicode_form: Literal["NFKC"] = "NFKC"
    collapse_whitespace: Literal[True] = True
    casefold_deduplication: Literal[True] = True
    maximum_text_characters: Literal[8000] = 8000
    near_duplicate_token_ngram_size: Literal[3] = 3
    near_duplicate_jaccard_threshold: float = Field(default=0.82, gt=0, le=1)
    development_split_seed: Literal["deepshield-c2-split-v1"] = "deepshield-c2-split-v1"
    development_split_percentages: dict[RiskSplit, int] = Field(
        default_factory=lambda: {
            RiskSplit.TRAIN: 60,
            RiskSplit.VALIDATION: 15,
            RiskSplit.TEST: 25,
        }
    )

    @field_validator("development_split_percentages")
    @classmethod
    def require_development_split_percentages(
        cls, value: dict[RiskSplit, int]
    ) -> dict[RiskSplit, int]:
        if set(value) != {RiskSplit.TRAIN, RiskSplit.VALIDATION, RiskSplit.TEST}:
            raise ValueError("development split percentages require train, validation, and test")
        if any(percent <= 0 for percent in value.values()) or sum(value.values()) != 100:
            raise ValueError("development split percentages must be positive and sum to 100")
        return value


class SourcePreprocessingSummary(BaseModel):
    """Auditable row accounting for one registered classifier source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: Slug
    input_rows: int = Field(ge=1)
    output_rows: int = Field(ge=0)
    empty_rows_excluded: int = Field(ge=0)
    too_short_rows_excluded: int = Field(ge=0)
    overlength_rows_excluded: int = Field(ge=0)
    duplicate_rows_excluded: int = Field(ge=0)
    cross_source_exact_duplicates_excluded: int = Field(ge=0)
    cross_source_near_duplicates_excluded: int = Field(ge=0)
    output_label_counts: dict[RiskLabel, int]

    @field_validator("output_label_counts")
    @classmethod
    def require_source_labels(cls, value: dict[RiskLabel, int]) -> dict[RiskLabel, int]:
        if set(value) != set(RiskLabel) or any(count < 0 for count in value.values()):
            raise ValueError("preprocessing source counts require both non-negative labels")
        return value

    @model_validator(mode="after")
    def require_complete_row_accounting(self) -> Self:
        excluded = (
            self.empty_rows_excluded
            + self.too_short_rows_excluded
            + self.overlength_rows_excluded
            + self.duplicate_rows_excluded
            + self.cross_source_exact_duplicates_excluded
            + self.cross_source_near_duplicates_excluded
        )
        if self.output_rows + excluded != self.input_rows:
            raise ValueError("preprocessing source counts must account for every input row")
        if sum(self.output_label_counts.values()) != self.output_rows:
            raise ValueError("preprocessing label counts must sum to output rows")
        return self


class SplitPreprocessingSummary(BaseModel):
    """Counts used to verify the derived split distribution."""

    model_config = ConfigDict(extra="forbid")

    split: RiskSplit
    row_count: int = Field(ge=1)
    group_count: int = Field(ge=1)
    label_counts: dict[RiskLabel, int]

    @field_validator("label_counts")
    @classmethod
    def require_split_labels(cls, value: dict[RiskLabel, int]) -> dict[RiskLabel, int]:
        if set(value) != set(RiskLabel) or any(count < 0 for count in value.values()):
            raise ValueError("preprocessing split counts require both non-negative labels")
        return value

    @model_validator(mode="after")
    def require_split_label_accounting(self) -> Self:
        if sum(self.label_counts.values()) != self.row_count:
            raise ValueError("preprocessing split label counts must sum to row count")
        return self


class ClassifierPreprocessingReport(BaseModel):
    """Reproducibility evidence for one derived classifier candidate dataset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    report_id: Literal["deepshield-risk-classifier-preprocessing"] = (
        "deepshield-risk-classifier-preprocessing"
    )
    report_version: int = Field(ge=1)
    created_at: datetime
    source_collection_path: Literal["data/classifier/source-collection.v1.json"]
    source_collection_sha256: Sha256Digest
    output_dataset_path: str = Field(pattern=r"^data/classifier/processed/[a-z0-9.-]+\.json$")
    output_dataset_sha256: Sha256Digest
    rules: NormalizationRules
    sources: list[SourcePreprocessingSummary] = Field(min_length=2)
    splits: list[SplitPreprocessingSummary] = Field(min_length=4, max_length=4)
    development_near_duplicate_cluster_count: int = Field(ge=1)
    development_multi_member_cluster_count: int = Field(ge=0)
    development_largest_cluster_size: int = Field(ge=1)

    @field_validator("created_at")
    @classmethod
    def require_preprocessing_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_complete_preprocessing_report(self) -> Self:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("preprocessing report source IDs must be unique")
        splits = [summary.split for summary in self.splits]
        if set(splits) != set(RiskSplit) or len(splits) != len(set(splits)):
            raise ValueError("preprocessing report requires each split exactly once")
        output_rows = sum(source.output_rows for source in self.sources)
        if sum(split.row_count for split in self.splits) != output_rows:
            raise ValueError("preprocessing split counts must match source output rows")
        if self.development_multi_member_cluster_count > (
            self.development_near_duplicate_cluster_count
        ):
            raise ValueError("multi-member cluster count cannot exceed total cluster count")
        return self


class SourceColumns(BaseModel):
    """Upstream columns needed for deterministic normalization."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: str | None = None
    context: str | None = None
    severity: str | None = None


class ClassifierSource(BaseModel):
    """One licensed, immutable external classifier-data source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: Slug
    provider: Literal["hugging-face"] = "hugging-face"
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dataset_url: HttpUrl
    license_spdx: Literal["Apache-2.0", "MIT"]
    license_url: HttpUrl
    purpose: SourcePurpose
    grouping_strategy: GroupingStrategy
    files: list[str] = Field(min_length=1)
    columns: SourceColumns
    label_mapping: dict[str, RiskLabel]
    expected_rows: int | None = Field(default=None, ge=1)
    lfs_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    citation: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def require_safe_unique_files(cls, value: list[str]) -> list[str]:
        """Allow only explicit tabular files inside the pinned dataset repository."""

        if len(value) != len(set(value)):
            raise ValueError("source files must be unique")
        for item in value:
            path = PurePosixPath(item)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.suffix.casefold() not in {".csv", ".jsonl", ".parquet"}
            ):
                raise ValueError("source files must be safe CSV, JSONL, or Parquet paths")
        return value

    @field_validator("label_mapping")
    @classmethod
    def require_binary_label_mapping(cls, value: dict[str, RiskLabel]) -> dict[str, RiskLabel]:
        """Require the source mapping to represent both classifier decisions."""

        if not value or set(value.values()) != set(RiskLabel):
            raise ValueError("label_mapping must contain benign and adversarial labels")
        return value

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        """Bind URLs, checksums, and purpose to the declared immutable source."""

        expected_url = f"https://huggingface.co/datasets/{self.dataset_id}/tree/{self.revision}"
        if str(self.dataset_url).rstrip("/") != expected_url:
            raise ValueError("dataset_url must match dataset_id and pinned revision")
        if set(self.lfs_sha256) - set(self.files):
            raise ValueError("LFS checksums may only describe declared source files")
        if self.grouping_strategy is GroupingStrategy.UPSTREAM_COLUMN:
            if self.columns.group is None:
                raise ValueError("upstream-column grouping requires a group column")
        elif self.columns.group is not None:
            raise ValueError("a group column requires upstream-column grouping")
        if (
            self.purpose is SourcePurpose.DEVELOPMENT
            and self.grouping_strategy is GroupingStrategy.EXTERNAL_SOURCE_SPLIT
        ):
            raise ValueError("development sources require leakage-resistant clustering")
        if (
            self.purpose is SourcePurpose.EXTERNAL_TEST
            and self.grouping_strategy is not GroupingStrategy.EXTERNAL_SOURCE_SPLIT
        ):
            raise ValueError("external-test sources require external-source-split grouping")
        return self


class ClassifierSourceRegistry(BaseModel):
    """Versioned selection and licensing record for classifier data."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    registry_id: Literal["deepshield-risk-classifier-sources"] = (
        "deepshield-risk-classifier-sources"
    )
    registry_version: int = Field(ge=1)
    status: ArtifactStatus
    created_at: datetime
    benchmark_exclusion_policy: str = Field(min_length=1)
    sources: list[ClassifierSource] = Field(min_length=2)

    @field_validator("created_at")
    @classmethod
    def require_created_at_timezone(cls, value: datetime) -> datetime:
        """Keep registry timestamps unambiguous."""

        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        """Require unique sources and independent development and stress-test data."""

        source_ids = [source.source_id for source in self.sources]
        dataset_ids = [source.dataset_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("classifier source IDs must be unique")
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("classifier dataset IDs must be unique")
        purposes = {source.purpose for source in self.sources}
        if purposes != set(SourcePurpose):
            raise ValueError("registry requires development and external-test sources")
        return self


class CollectedClassifierFile(BaseModel):
    """Verified metadata for one locally collected classifier source file."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: Slug
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_file: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    sha256: Sha256Digest
    size_bytes: int = Field(ge=1)
    row_count: int = Field(ge=1)
    columns: list[str] = Field(min_length=1)
    label_counts: dict[str, int] = Field(min_length=2)
    empty_text_count: int = Field(ge=0)

    @field_validator("source_file")
    @classmethod
    def require_safe_source_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".csv":
            raise ValueError("collected source file must be a safe CSV path")
        return path.as_posix()

    @field_validator("raw_path")
    @classmethod
    def require_classifier_raw_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[:3] != ("data", "classifier", "raw")
            or path.suffix.casefold() != ".csv"
        ):
            raise ValueError("raw_path must be a safe CSV path under data/classifier/raw")
        return path.as_posix()

    @field_validator("columns")
    @classmethod
    def require_unique_columns(cls, value: list[str]) -> list[str]:
        if any(not column for column in value) or len(value) != len(set(value)):
            raise ValueError("collected CSV columns must be non-empty and unique")
        return value

    @field_validator("label_counts")
    @classmethod
    def require_positive_label_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not label or count <= 0 for label, count in value.items()):
            raise ValueError("collected label counts must be positive")
        return value

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> Self:
        if sum(self.label_counts.values()) != self.row_count:
            raise ValueError("label counts must sum to the collected row count")
        if self.empty_text_count > self.row_count:
            raise ValueError("empty text count cannot exceed the collected row count")
        expected_prefix = PurePosixPath("data", "classifier", "raw", self.source_id)
        raw_path = PurePosixPath(self.raw_path)
        if raw_path.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
            raise ValueError("raw_path source directory must match source_id")
        if raw_path.relative_to(expected_prefix) != PurePosixPath(self.source_file):
            raise ValueError("raw_path must preserve the registered source file path")
        return self


class ClassifierCollectionReceipt(BaseModel):
    """Versioned evidence for locally collected, immutable classifier source files."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    collection_id: Literal["deepshield-risk-classifier-source-collection"] = (
        "deepshield-risk-classifier-source-collection"
    )
    receipt_version: int = Field(ge=1)
    collected_at: datetime
    source_registry_path: Literal["data/classifier/source-registry.v2.json"]
    source_registry_sha256: Sha256Digest
    files: list[CollectedClassifierFile] = Field(min_length=1)

    @field_validator("collected_at")
    @classmethod
    def require_collection_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("collected_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_unique_collection_records(self) -> Self:
        identities = [(item.source_id, item.source_file) for item in self.files]
        raw_paths = [item.raw_path for item in self.files]
        if len(identities) != len(set(identities)):
            raise ValueError("collection receipt contains duplicate source files")
        if len(raw_paths) != len(set(raw_paths)):
            raise ValueError("collection receipt contains duplicate raw paths")
        return self


class RiskExample(BaseModel):
    """One traceable classifier example kept separate from the 600-case benchmark."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    example_id: Slug
    source_id: Slug
    source_record_id: str = Field(min_length=1, max_length=200)
    group_id: Slug
    split: RiskSplit
    label: RiskLabel
    category: RiskCategory
    text: str = Field(min_length=3, max_length=8_000)
    text_sha256: Sha256Digest
    synthetic: bool = False
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewer_id: Slug | None = None
    reviewed_at: datetime | None = None

    @field_validator("reviewed_at")
    @classmethod
    def require_review_timezone(cls, value: datetime | None) -> datetime | None:
        """Keep review timestamps unambiguous."""

        if value is not None and value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_example(self) -> Self:
        """Keep labels, hashes, and human-review evidence internally consistent."""

        if self.label is RiskLabel.BENIGN and self.category is not RiskCategory.BENIGN:
            raise ValueError("benign examples require the benign category")
        if self.label is RiskLabel.ADVERSARIAL and self.category is RiskCategory.BENIGN:
            raise ValueError("adversarial examples require an adversarial category")
        expected_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != expected_hash:
            raise ValueError("text_sha256 does not match the normalized example text")
        if self.review_status is ReviewStatus.APPROVED:
            if self.reviewer_id is None or self.reviewed_at is None:
                raise ValueError("approved examples require reviewer_id and reviewed_at")
        elif self.reviewer_id is not None or self.reviewed_at is not None:
            raise ValueError("draft examples cannot include approval metadata")
        return self


class RiskDatasetManifest(BaseModel):
    """Leakage-resistant dataset contract for lightweight classifier experiments."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["deepshield-risk-classifier"] = "deepshield-risk-classifier"
    dataset_version: int = Field(ge=1)
    status: ArtifactStatus
    created_at: datetime
    source_registry_path: Literal["data/classifier/source-registry.v2.json"]
    source_registry_sha256: Sha256Digest
    benchmark_case_ids_sha256: Sha256Digest | None = None
    minimum_example_count: Literal[1000] = 1000
    minimum_train_count: Literal[500] = 500
    minimum_validation_count: Literal[100] = 100
    minimum_test_count: Literal[200] = 200
    minimum_external_test_count: Literal[200] = 200
    examples: list[RiskExample] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_manifest_timezone(cls, value: datetime) -> datetime:
        """Keep manifest timestamps unambiguous."""

        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def prevent_classifier_data_leakage(self) -> Self:
        """Reject identity, text, group, and benchmark leakage across partitions."""

        example_ids = [example.example_id for example in self.examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("classifier dataset contains duplicate example IDs")

        text_hashes = [example.text_sha256 for example in self.examples]
        if len(text_hashes) != len(set(text_hashes)):
            raise ValueError("classifier dataset contains duplicate normalized texts")

        group_splits: dict[str, RiskSplit] = {}
        for example in self.examples:
            prior = group_splits.setdefault(example.group_id, example.split)
            if prior is not example.split:
                raise ValueError(f"classifier group {example.group_id} crosses data splits")

        if self.status is ArtifactStatus.FROZEN:
            self._validate_frozen_dataset()
        return self

    def _validate_frozen_dataset(self) -> None:
        if self.benchmark_case_ids_sha256 is None:
            raise ValueError("frozen classifier data requires a benchmark exclusion checksum")
        if any(example.review_status is not ReviewStatus.APPROVED for example in self.examples):
            raise ValueError("every example in a frozen classifier dataset must be approved")
        if len(self.examples) < self.minimum_example_count:
            raise ValueError("frozen classifier data does not meet the minimum example count")
        present_splits = {example.split for example in self.examples}
        if present_splits != set(RiskSplit):
            raise ValueError("frozen classifier data requires all four partitions")
        minimum_counts = {
            RiskSplit.TRAIN: self.minimum_train_count,
            RiskSplit.VALIDATION: self.minimum_validation_count,
            RiskSplit.TEST: self.minimum_test_count,
            RiskSplit.EXTERNAL_TEST: self.minimum_external_test_count,
        }
        for split in RiskSplit:
            split_examples = [example for example in self.examples if example.split is split]
            if len(split_examples) < minimum_counts[split]:
                raise ValueError(f"classifier split {split.value} does not meet its minimum count")
            labels = {example.label for example in split_examples}
            if labels != set(RiskLabel):
                raise ValueError(f"classifier split {split.value} requires both labels")
