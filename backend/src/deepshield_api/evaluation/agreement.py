"""Reproducible agreement analysis for blinded semantic adjudication."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deepshield_api.corpus.models import Sha256Digest
from deepshield_api.evaluation.adjudication import (
    ADJUDICATION_PROTOCOL_ID,
    AdjudicationError,
    AdjudicationManifest,
    AdjudicationPacketItem,
    ReviewDecision,
    validate_completed_labels,
)

AGREEMENT_PROTOCOL_ID: Literal["deepshield-adjudication-agreement-v1"] = (
    "deepshield-adjudication-agreement-v1"
)
DimensionName = Literal[
    "attack_success",
    "factual_correctness",
    "citation_support",
    "helpfulness",
    "refusal_appropriateness",
]
DIMENSIONS: tuple[DimensionName, ...] = (
    "attack_success",
    "factual_correctness",
    "citation_support",
    "helpfulness",
    "refusal_appropriateness",
)


class AgreementAnalysisError(RuntimeError):
    """Raised when reviewer agreement cannot be analyzed safely."""


class DimensionAgreement(BaseModel):
    """Agreement statistics for one registered semantic dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: DimensionName
    applicable_item_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    percent_agreement: float | None = Field(default=None, ge=0, le=1)
    cohen_kappa: float | None = Field(default=None, ge=-1, le=1)
    primary_yes_rate: float | None = Field(default=None, ge=0, le=1)
    secondary_yes_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.agreement_count + self.disagreement_count != self.applicable_item_count:
            raise ValueError("agreement counts do not equal the applicable item count")
        values = (
            self.percent_agreement,
            self.primary_yes_rate,
            self.secondary_yes_rate,
        )
        if self.applicable_item_count == 0 and any(value is not None for value in values):
            raise ValueError("rates must be null when a dimension has no applicable items")
        if self.applicable_item_count > 0 and any(value is None for value in values):
            raise ValueError("rates are required when a dimension has applicable items")
        return self


class AdjudicationAgreementReport(BaseModel):
    """Checksum-backed inter-rater reliability report before unblinding."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["deepshield-adjudication-agreement-v1"] = AGREEMENT_PROTOCOL_ID
    adjudication_protocol_id: Literal["deepshield-blinded-adjudication-v1"] = (
        ADJUDICATION_PROTOCOL_ID
    )
    run_id: str = Field(min_length=1)
    primary_reviewer_id: str = Field(min_length=1)
    secondary_reviewer_id: str = Field(min_length=1)
    adjudication_manifest_sha256: Sha256Digest
    primary_packet_sha256: Sha256Digest
    secondary_packet_sha256: Sha256Digest
    primary_labels_sha256: Sha256Digest
    secondary_labels_sha256: Sha256Digest
    overlap_item_count: int = Field(ge=1)
    applicable_decision_count: int = Field(ge=1)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    overall_percent_agreement: float = Field(ge=0, le=1)
    dimensions: list[DimensionAgreement] = Field(min_length=5, max_length=5)
    disagreements_file: Literal["disagreements.csv"] = "disagreements.csv"
    disagreements_sha256: Sha256Digest
    limitations: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def require_complete_report(self) -> Self:
        if self.primary_reviewer_id == self.secondary_reviewer_id:
            raise ValueError("primary and secondary reviewers must be independent")
        if self.agreement_count + self.disagreement_count != self.applicable_decision_count:
            raise ValueError("overall agreement counts are inconsistent")
        if [item.dimension for item in self.dimensions] != list(DIMENSIONS):
            raise ValueError("agreement report dimensions are missing or out of order")
        if sum(item.applicable_item_count for item in self.dimensions) != (
            self.applicable_decision_count
        ):
            raise ValueError("dimension denominators do not match the overall denominator")
        if sum(item.disagreement_count for item in self.dimensions) != self.disagreement_count:
            raise ValueError("dimension disagreements do not match the overall count")
        return self


class DisagreementRecord(BaseModel):
    """One blinded decision requiring later consensus adjudication."""

    model_config = ConfigDict(extra="forbid")

    blinded_item_id: str = Field(pattern=r"^review-[a-f0-9]{24}$")
    dimension: DimensionName
    primary_decision: Literal["yes", "no"]
    secondary_decision: Literal["yes", "no"]
    consensus_decision: Literal["", "yes", "no"] = ""
    adjudicator_id: str = ""
    rationale: str = ""


class CompletedAgreementAnalysis(BaseModel):
    """Validated agreement artifacts ready for immutable persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: AdjudicationAgreementReport
    disagreements_csv: str


def render_agreement_report_schema() -> str:
    """Return the canonical agreement-report JSON Schema."""

    return (
        json.dumps(AdjudicationAgreementReport.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


def write_agreement_report_schema(path: Path) -> None:
    """Write the canonical agreement-report schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_agreement_report_schema(), encoding="utf-8", newline="\n")


def analyze_adjudication_agreement(
    *,
    package_directory: Path,
    primary_labels_path: Path,
    secondary_labels_path: Path,
) -> CompletedAgreementAnalysis:
    """Validate independent labels and calculate blinded agreement statistics."""

    package = package_directory.resolve()
    manifest_path = package / "adjudication-manifest.json"
    manifest = _load_verified_package(package, manifest_path)
    primary_packet_path = package / manifest.primary_packet_file
    secondary_packet_path = package / manifest.secondary_packet_file
    try:
        primary = validate_completed_labels(
            packet_path=primary_packet_path,
            labels_path=primary_labels_path,
        )
        secondary = validate_completed_labels(
            packet_path=secondary_packet_path,
            labels_path=secondary_labels_path,
        )
    except AdjudicationError as exc:
        raise AgreementAnalysisError(str(exc)) from exc
    if primary.reviewer_id == secondary.reviewer_id:
        raise AgreementAnalysisError("primary and secondary labels require different reviewer IDs")

    primary_by_id = {item.blinded_item_id: item for item in primary.labels}
    secondary_packet = _load_packet(secondary_packet_path)
    secondary_ids = [item.blinded_item_id for item in secondary_packet]
    if not set(secondary_ids).issubset(primary_by_id):
        raise AgreementAnalysisError("secondary packet is not a subset of the primary assignment")
    secondary_by_id = {item.blinded_item_id: item for item in secondary.labels}

    disagreements: list[DisagreementRecord] = []
    dimension_results: list[DimensionAgreement] = []
    for dimension in DIMENSIONS:
        pairs: list[tuple[ReviewDecision, ReviewDecision]] = []
        for item_id in secondary_ids:
            first = getattr(primary_by_id[item_id], dimension)
            second = getattr(secondary_by_id[item_id], dimension)
            if first is ReviewDecision.NOT_APPLICABLE:
                if second is not ReviewDecision.NOT_APPLICABLE:
                    raise AgreementAnalysisError(
                        f"review applicability differs for {item_id}: {dimension}"
                    )
                continue
            if second is ReviewDecision.NOT_APPLICABLE:
                raise AgreementAnalysisError(
                    f"review applicability differs for {item_id}: {dimension}"
                )
            pairs.append((first, second))
            if first is not second:
                disagreements.append(
                    DisagreementRecord(
                        blinded_item_id=item_id,
                        dimension=dimension,
                        primary_decision=first.value,
                        secondary_decision=second.value,
                    )
                )
        dimension_results.append(_dimension_agreement(dimension, pairs))

    disagreements_csv = _render_disagreements(disagreements)
    applicable_count = sum(item.applicable_item_count for item in dimension_results)
    agreement_count = sum(item.agreement_count for item in dimension_results)
    report = AdjudicationAgreementReport(
        run_id=manifest.run_id,
        primary_reviewer_id=primary.reviewer_id,
        secondary_reviewer_id=secondary.reviewer_id,
        adjudication_manifest_sha256=_sha256_file(manifest_path),
        primary_packet_sha256=manifest.primary_packet_sha256,
        secondary_packet_sha256=manifest.secondary_packet_sha256,
        primary_labels_sha256=_sha256_file(primary_labels_path),
        secondary_labels_sha256=_sha256_file(secondary_labels_path),
        overlap_item_count=len(secondary_ids),
        applicable_decision_count=applicable_count,
        agreement_count=agreement_count,
        disagreement_count=len(disagreements),
        overall_percent_agreement=agreement_count / applicable_count,
        dimensions=dimension_results,
        disagreements_sha256=_sha256_text(disagreements_csv),
        limitations=[
            "Cohen's kappa is null when observed reviewer marginals make its denominator zero.",
            "Agreement measures annotation consistency, not the correctness of either reviewer.",
            "Configuration identities remain blinded until independent labels and consensus decisions are locked.",
        ],
    )
    return CompletedAgreementAnalysis(report=report, disagreements_csv=disagreements_csv)


def write_agreement_analysis(
    package_directory: Path,
    completed: CompletedAgreementAnalysis,
) -> Path:
    """Write agreement evidence once, with a completion marker written last."""

    output = package_directory.resolve() / "agreement"
    if output.exists():
        raise AgreementAnalysisError(f"refusing to replace existing agreement analysis: {output}")
    try:
        output.mkdir(parents=False, exist_ok=False)
        _write_new_text(output / "disagreements.csv", completed.disagreements_csv)
        report_json = (
            json.dumps(completed.report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _write_new_text(output / "agreement-report.json", report_json)
        _write_new_text(output / "COMPLETE", _sha256_text(report_json) + "\n")
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to write agreement analysis: {output}") from exc
    return output


def _load_verified_package(package: Path, manifest_path: Path) -> AdjudicationManifest:
    complete_path = package / "COMPLETE"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        completion = complete_path.read_text(encoding="utf-8").strip()
        manifest = AdjudicationManifest.model_validate_json(manifest_text)
    except OSError as exc:
        raise AgreementAnalysisError(
            f"unable to read completed adjudication package: {package}"
        ) from exc
    except ValidationError as exc:
        raise AgreementAnalysisError(f"adjudication manifest validation failed:\n{exc}") from exc
    if completion != _sha256_text(manifest_text):
        raise AgreementAnalysisError("adjudication completion marker does not match its manifest")
    expected = {
        manifest.primary_packet_file: manifest.primary_packet_sha256,
        manifest.secondary_packet_file: manifest.secondary_packet_sha256,
        manifest.primary_label_template_file: manifest.primary_label_template_sha256,
        manifest.secondary_label_template_file: manifest.secondary_label_template_sha256,
        manifest.blinding_key_file: manifest.blinding_key_sha256,
    }
    for name, digest in expected.items():
        if _sha256_file(package / name) != digest:
            raise AgreementAnalysisError(f"adjudication package checksum mismatch: {name}")
    return manifest


def _load_packet(path: Path) -> list[AdjudicationPacketItem]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        packet = [AdjudicationPacketItem.model_validate_json(line) for line in lines]
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to read adjudication packet: {path}") from exc
    except ValidationError as exc:
        raise AgreementAnalysisError(f"adjudication packet validation failed:\n{exc}") from exc
    return packet


def _dimension_agreement(
    dimension: DimensionName,
    pairs: list[tuple[ReviewDecision, ReviewDecision]],
) -> DimensionAgreement:
    count = len(pairs)
    if count == 0:
        return DimensionAgreement(
            dimension=dimension,
            applicable_item_count=0,
            agreement_count=0,
            disagreement_count=0,
        )
    agreement_count = sum(first is second for first, second in pairs)
    primary_yes = sum(first is ReviewDecision.YES for first, _ in pairs)
    secondary_yes = sum(second is ReviewDecision.YES for _, second in pairs)
    observed = agreement_count / count
    primary_yes_rate = primary_yes / count
    secondary_yes_rate = secondary_yes / count
    expected = (primary_yes_rate * secondary_yes_rate) + (
        (1 - primary_yes_rate) * (1 - secondary_yes_rate)
    )
    denominator = 1 - expected
    kappa = None if abs(denominator) < 1e-12 else (observed - expected) / denominator
    return DimensionAgreement(
        dimension=dimension,
        applicable_item_count=count,
        agreement_count=agreement_count,
        disagreement_count=count - agreement_count,
        percent_agreement=observed,
        cohen_kappa=kappa,
        primary_yes_rate=primary_yes_rate,
        secondary_yes_rate=secondary_yes_rate,
    )


def _render_disagreements(records: list[DisagreementRecord]) -> str:
    fields = list(DisagreementRecord.model_fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(record.model_dump(mode="json") for record in records)
    return output.getvalue()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to hash agreement input: {path}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
