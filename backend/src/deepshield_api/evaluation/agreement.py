"""Reproducible agreement analysis for blinded semantic adjudication."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from deepshield_api.corpus.models import Sha256Digest, Slug
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
CONSENSUS_PROTOCOL_ID: Literal["deepshield-adjudication-consensus-v1"] = (
    "deepshield-adjudication-consensus-v1"
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


class ConsensusRecord(BaseModel):
    """One completed, still-blinded resolution of a registered disagreement."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    blinded_item_id: str = Field(pattern=r"^review-[a-f0-9]{24}$")
    dimension: DimensionName
    primary_decision: Literal["yes", "no"]
    secondary_decision: Literal["yes", "no"]
    consensus_decision: Literal["yes", "no"]
    adjudicator_id: Slug
    rationale: str = Field(min_length=1, max_length=2_000)

    @field_validator("rationale")
    @classmethod
    def reject_spreadsheet_formula_rationale(cls, value: str) -> str:
        if value.lstrip().startswith(("=", "+", "-", "@")):
            raise ValueError("consensus rationale cannot begin with a spreadsheet formula marker")
        return value


class ResolvedAdjudicationItem(BaseModel):
    """Final blinded semantic decisions after registered consensus resolution."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    blinded_item_id: str = Field(pattern=r"^review-[a-f0-9]{24}$")
    attack_success: ReviewDecision
    factual_correctness: ReviewDecision
    citation_support: ReviewDecision
    helpfulness: ReviewDecision
    refusal_appropriateness: ReviewDecision
    consensus_dimensions: list[DimensionName]

    @field_validator("consensus_dimensions")
    @classmethod
    def require_unique_ordered_dimensions(cls, value: list[DimensionName]) -> list[DimensionName]:
        if value != [dimension for dimension in DIMENSIONS if dimension in value]:
            raise ValueError("consensus dimensions must be unique and registered order")
        return value


class AdjudicationConsensusReport(BaseModel):
    """Checksum-backed proof that every blinded disagreement was resolved."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["deepshield-adjudication-consensus-v1"] = CONSENSUS_PROTOCOL_ID
    agreement_protocol_id: Literal["deepshield-adjudication-agreement-v1"] = AGREEMENT_PROTOCOL_ID
    run_id: str = Field(min_length=1)
    agreement_report_sha256: Sha256Digest
    disagreements_template_sha256: Sha256Digest
    primary_labels_sha256: Sha256Digest
    secondary_labels_sha256: Sha256Digest
    completed_consensus_sha256: Sha256Digest
    adjudicator_id: Slug | None
    item_count: int = Field(ge=1)
    disagreement_count: int = Field(ge=0)
    resolved_disagreement_count: int = Field(ge=0)
    resolved_labels_file: Literal["resolved-labels.jsonl"] = "resolved-labels.jsonl"
    resolved_labels_sha256: Sha256Digest
    limitations: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def require_complete_resolution(self) -> Self:
        if self.resolved_disagreement_count != self.disagreement_count:
            raise ValueError("every registered disagreement must be resolved")
        if (self.disagreement_count == 0) != (self.adjudicator_id is None):
            raise ValueError("adjudicator ID is required exactly when disagreements exist")
        return self


class CompletedConsensusAnalysis(BaseModel):
    """Validated consensus artifacts ready for immutable persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: AdjudicationConsensusReport
    resolved_labels_jsonl: str


def render_agreement_report_schema() -> str:
    """Return the canonical agreement-report JSON Schema."""

    return (
        json.dumps(AdjudicationAgreementReport.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


def write_agreement_report_schema(path: Path) -> None:
    """Write the canonical agreement-report schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_agreement_report_schema(), encoding="utf-8", newline="\n")


def render_consensus_report_schema() -> str:
    """Return the canonical consensus-report JSON Schema."""

    return (
        json.dumps(AdjudicationConsensusReport.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


def write_consensus_report_schema(path: Path) -> None:
    """Write the canonical consensus-report schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_consensus_report_schema(), encoding="utf-8", newline="\n")


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


def resolve_adjudication_consensus(
    *,
    package_directory: Path,
    primary_labels_path: Path,
    secondary_labels_path: Path,
    completed_consensus_path: Path,
) -> CompletedConsensusAnalysis:
    """Validate blinded consensus decisions and produce one resolved label per item."""

    package = package_directory.resolve()
    manifest = _load_verified_package(package, package / "adjudication-manifest.json")
    agreement, frozen_disagreements = _load_verified_agreement(package)
    if agreement.adjudication_manifest_sha256 != _sha256_file(
        package / "adjudication-manifest.json"
    ):
        raise AgreementAnalysisError("agreement report does not match the adjudication package")
    if _sha256_file(primary_labels_path) != agreement.primary_labels_sha256:
        raise AgreementAnalysisError("primary labels do not match the agreement report")
    if _sha256_file(secondary_labels_path) != agreement.secondary_labels_sha256:
        raise AgreementAnalysisError("secondary labels do not match the agreement report")
    try:
        primary = validate_completed_labels(
            packet_path=package / manifest.primary_packet_file,
            labels_path=primary_labels_path,
        )
        secondary = validate_completed_labels(
            packet_path=package / manifest.secondary_packet_file,
            labels_path=secondary_labels_path,
        )
    except AdjudicationError as exc:
        raise AgreementAnalysisError(str(exc)) from exc
    if (
        primary.reviewer_id != agreement.primary_reviewer_id
        or secondary.reviewer_id != agreement.secondary_reviewer_id
    ):
        raise AgreementAnalysisError("reviewer IDs do not match the agreement report")

    consensus = _load_completed_consensus(completed_consensus_path, frozen_disagreements)
    adjudicator_ids = {record.adjudicator_id for record in consensus}
    if len(adjudicator_ids) > 1:
        raise AgreementAnalysisError("one consensus file must contain one adjudicator ID")
    adjudicator_id = next(iter(adjudicator_ids), None)
    if adjudicator_id in {primary.reviewer_id, secondary.reviewer_id}:
        raise AgreementAnalysisError(
            "consensus adjudicator ID must differ from both independent reviewer IDs"
        )

    consensus_by_key = {(record.blinded_item_id, record.dimension): record for record in consensus}
    primary_by_id = {item.blinded_item_id: item for item in primary.labels}
    secondary_by_id = {item.blinded_item_id: item for item in secondary.labels}
    primary_packet = _load_packet(package / manifest.primary_packet_file)
    resolved: list[ResolvedAdjudicationItem] = []
    for packet_item in primary_packet:
        item_id = packet_item.blinded_item_id
        primary_label = primary_by_id[item_id]
        secondary_label = secondary_by_id.get(item_id)
        decisions: dict[str, ReviewDecision] = {}
        consensus_dimensions: list[DimensionName] = []
        for dimension in DIMENSIONS:
            first = getattr(primary_label, dimension)
            if secondary_label is None:
                decisions[dimension] = first
                continue
            second = getattr(secondary_label, dimension)
            if first is second:
                decisions[dimension] = first
                continue
            record = consensus_by_key.get((item_id, dimension))
            if record is None:
                raise AgreementAnalysisError(
                    f"missing registered consensus decision for {item_id}: {dimension}"
                )
            decisions[dimension] = ReviewDecision(record.consensus_decision)
            consensus_dimensions.append(dimension)
        resolved.append(
            ResolvedAdjudicationItem(
                blinded_item_id=item_id,
                attack_success=decisions["attack_success"],
                factual_correctness=decisions["factual_correctness"],
                citation_support=decisions["citation_support"],
                helpfulness=decisions["helpfulness"],
                refusal_appropriateness=decisions["refusal_appropriateness"],
                consensus_dimensions=consensus_dimensions,
            )
        )

    resolved_jsonl = "".join(item.model_dump_json() + "\n" for item in resolved)
    report = AdjudicationConsensusReport(
        run_id=agreement.run_id,
        agreement_report_sha256=_sha256_file(package / "agreement/agreement-report.json"),
        disagreements_template_sha256=agreement.disagreements_sha256,
        primary_labels_sha256=agreement.primary_labels_sha256,
        secondary_labels_sha256=agreement.secondary_labels_sha256,
        completed_consensus_sha256=_sha256_file(completed_consensus_path),
        adjudicator_id=adjudicator_id,
        item_count=len(resolved),
        disagreement_count=agreement.disagreement_count,
        resolved_disagreement_count=len(consensus),
        resolved_labels_sha256=_sha256_text(resolved_jsonl),
        limitations=[
            "Consensus decisions remain configuration-blinded and are not configuration-level results.",
            "Primary-only items retain the primary reviewer's decision because the registered secondary sample covers 20% of cases.",
            "Unblinding and configuration-level semantic summaries require a later immutable analysis step.",
        ],
    )
    return CompletedConsensusAnalysis(report=report, resolved_labels_jsonl=resolved_jsonl)


def write_consensus_analysis(
    package_directory: Path,
    completed: CompletedConsensusAnalysis,
) -> Path:
    """Write resolved blinded labels once, with the completion marker last."""

    output = package_directory.resolve() / "consensus"
    if output.exists():
        raise AgreementAnalysisError(f"refusing to replace existing consensus analysis: {output}")
    try:
        output.mkdir(parents=False, exist_ok=False)
        _write_new_text(output / "resolved-labels.jsonl", completed.resolved_labels_jsonl)
        report_json = (
            json.dumps(completed.report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _write_new_text(output / "consensus-report.json", report_json)
        _write_new_text(output / "COMPLETE", _sha256_text(report_json) + "\n")
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to write consensus analysis: {output}") from exc
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


def _load_verified_agreement(
    package: Path,
) -> tuple[AdjudicationAgreementReport, list[DisagreementRecord]]:
    agreement_directory = package / "agreement"
    report_path = agreement_directory / "agreement-report.json"
    disagreements_path = agreement_directory / "disagreements.csv"
    try:
        report_text = report_path.read_text(encoding="utf-8")
        completion = (agreement_directory / "COMPLETE").read_text(encoding="utf-8").strip()
        report = AdjudicationAgreementReport.model_validate_json(report_text)
    except OSError as exc:
        raise AgreementAnalysisError(
            f"unable to read completed agreement analysis: {agreement_directory}"
        ) from exc
    except ValidationError as exc:
        raise AgreementAnalysisError(f"agreement report validation failed:\n{exc}") from exc
    if completion != _sha256_text(report_text):
        raise AgreementAnalysisError("agreement completion marker does not match its report")
    if _sha256_file(disagreements_path) != report.disagreements_sha256:
        raise AgreementAnalysisError("frozen disagreement worksheet checksum does not match")
    disagreements = _load_disagreement_template(disagreements_path)
    keys = [(record.blinded_item_id, record.dimension) for record in disagreements]
    if len(keys) != len(set(keys)):
        raise AgreementAnalysisError("frozen disagreement worksheet contains duplicate decisions")
    if len(disagreements) != report.disagreement_count:
        raise AgreementAnalysisError("frozen disagreement count does not match the report")
    return report, disagreements


def _load_disagreement_template(path: Path) -> list[DisagreementRecord]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = reader.fieldnames
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to read disagreement worksheet: {path}") from exc
    expected_fields = list(DisagreementRecord.model_fields)
    if fields is None or set(fields) != set(expected_fields):
        raise AgreementAnalysisError("disagreement worksheet columns do not match the protocol")
    try:
        records = [DisagreementRecord.model_validate(row) for row in rows]
    except ValidationError as exc:
        raise AgreementAnalysisError(f"disagreement worksheet validation failed:\n{exc}") from exc
    return records


def _load_completed_consensus(
    path: Path,
    frozen: list[DisagreementRecord],
) -> list[ConsensusRecord]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = reader.fieldnames
    except OSError as exc:
        raise AgreementAnalysisError(f"unable to read completed consensus: {path}") from exc
    expected_fields = list(DisagreementRecord.model_fields)
    if fields is None or set(fields) != set(expected_fields):
        raise AgreementAnalysisError("completed consensus columns do not match the protocol")
    if len(rows) != len(frozen):
        raise AgreementAnalysisError(
            "completed consensus must resolve every registered disagreement"
        )
    try:
        records = [ConsensusRecord.model_validate(row) for row in rows]
    except ValidationError as exc:
        raise AgreementAnalysisError(f"completed consensus validation failed:\n{exc}") from exc
    keys = [(record.blinded_item_id, record.dimension) for record in records]
    if len(keys) != len(set(keys)):
        raise AgreementAnalysisError("completed consensus contains duplicate decisions")
    frozen_by_key = {(record.blinded_item_id, record.dimension): record for record in frozen}
    if set(keys) != set(frozen_by_key):
        raise AgreementAnalysisError("completed consensus does not match registered disagreements")
    for record in records:
        expected = frozen_by_key[(record.blinded_item_id, record.dimension)]
        if (
            record.primary_decision != expected.primary_decision
            or record.secondary_decision != expected.secondary_decision
        ):
            raise AgreementAnalysisError(
                f"reviewer decisions changed for {record.blinded_item_id}: {record.dimension}"
            )
    return records


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
