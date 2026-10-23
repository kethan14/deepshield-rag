"""Checksum-backed unblinding and semantic result summaries."""

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deepshield_api.corpus.models import Sha256Digest, Slug
from deepshield_api.evaluation.adjudication import (
    AdjudicationManifest,
    AdjudicationPacketItem,
    BlindingKey,
    ReviewDecision,
)
from deepshield_api.evaluation.agreement import (
    DIMENSIONS,
    AdjudicationConsensusReport,
    DimensionName,
    ResolvedAdjudicationItem,
)
from deepshield_api.evaluation.models import AttackCategory, BenchmarkKind, BenchmarkSplit
from deepshield_api.evaluation.runs import ExperimentConfiguration, ExperimentPurpose

SEMANTIC_ANALYSIS_PROTOCOL_ID: Literal["deepshield-semantic-analysis-v1"] = (
    "deepshield-semantic-analysis-v1"
)


class SemanticAnalysisError(RuntimeError):
    """Raised when consensus evidence cannot be safely unblinded or summarized."""


class UnblindedAdjudicationItem(BaseModel):
    """One resolved semantic label joined to its sealed configuration identity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    blinded_item_id: str = Field(pattern=r"^review-[a-f0-9]{24}$")
    trial_id: str = Field(min_length=1)
    case_id: Slug
    template_family: Slug
    configuration: ExperimentConfiguration
    kind: BenchmarkKind
    attack_category: AttackCategory | None
    attack_success: ReviewDecision
    factual_correctness: ReviewDecision
    citation_support: ReviewDecision
    helpfulness: ReviewDecision
    refusal_appropriateness: ReviewDecision
    consensus_dimensions: list[DimensionName]


class SemanticDimensionSummary(BaseModel):
    """Observed yes/no rate for one human-reviewed dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: DimensionName
    applicable_item_count: int = Field(ge=0)
    yes_count: int = Field(ge=0)
    no_count: int = Field(ge=0)
    yes_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.yes_count + self.no_count != self.applicable_item_count:
            raise ValueError("semantic decision counts do not equal the applicable item count")
        if (self.applicable_item_count == 0) != (self.yes_rate is None):
            raise ValueError("yes rate must be null exactly when no items are applicable")
        return self


class SemanticConfigurationSummary(BaseModel):
    """Human-reviewed outcome rates for one experimental configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: ExperimentConfiguration
    item_count: int = Field(ge=1)
    dimensions: list[SemanticDimensionSummary] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_registered_dimension_order(self) -> Self:
        if [item.dimension for item in self.dimensions] != list(DIMENSIONS):
            raise ValueError("semantic summary dimensions are missing or out of order")
        return self


class SemanticAttackCategorySummary(BaseModel):
    """Human-reviewed attack-success rate for one threat and configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: ExperimentConfiguration
    attack_category: AttackCategory
    item_count: int = Field(ge=1)
    attack_success_count: int = Field(ge=0)
    attack_success_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_valid_attack_count(self) -> Self:
        if self.attack_success_count > self.item_count:
            raise ValueError("attack-success count cannot exceed the item count")
        return self


class SemanticAnalysisReport(BaseModel):
    """Immutable provenance and configuration-level semantic results."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["deepshield-semantic-analysis-v1"] = SEMANTIC_ANALYSIS_PROTOCOL_ID
    run_id: Slug
    purpose: ExperimentPurpose
    benchmark_split: BenchmarkSplit
    adjudication_manifest_sha256: Sha256Digest
    blinding_key_sha256: Sha256Digest
    consensus_report_sha256: Sha256Digest
    resolved_labels_sha256: Sha256Digest
    item_count: int = Field(ge=1)
    case_count: int = Field(ge=1)
    unblinded_labels_file: Literal["unblinded-labels.jsonl"] = "unblinded-labels.jsonl"
    unblinded_labels_sha256: Sha256Digest
    configuration_summary_file: Literal["configuration-summary.csv"] = "configuration-summary.csv"
    configuration_summary_sha256: Sha256Digest
    attack_category_summary_file: Literal["attack-category-summary.csv"] = (
        "attack-category-summary.csv"
    )
    attack_category_summary_sha256: Sha256Digest
    configurations: list[SemanticConfigurationSummary] = Field(min_length=3, max_length=3)
    attack_categories: list[SemanticAttackCategorySummary]
    limitations: list[str] = Field(min_length=3)

    @model_validator(mode="after")
    def require_complete_design(self) -> Self:
        if [item.configuration for item in self.configurations] != list(ExperimentConfiguration):
            raise ValueError("semantic report configurations are missing or out of order")
        expected_items = self.case_count * len(ExperimentConfiguration)
        if self.item_count != expected_items:
            raise ValueError("semantic report requires one item per case and configuration")
        if any(item.item_count != self.case_count for item in self.configurations):
            raise ValueError("every configuration must contain the complete case set")
        return self


class CompletedSemanticAnalysis(BaseModel):
    """Validated semantic result artifacts ready for immutable persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: SemanticAnalysisReport
    unblinded_labels_jsonl: str
    configuration_summary_csv: str
    attack_category_summary_csv: str


def render_semantic_report_schema() -> str:
    """Return the canonical semantic-analysis report JSON Schema."""

    return json.dumps(SemanticAnalysisReport.model_json_schema(), indent=2, sort_keys=True) + "\n"


def write_semantic_report_schema(path: Path) -> None:
    """Write the canonical semantic-analysis report schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_semantic_report_schema(), encoding="utf-8", newline="\n")


def unblind_semantic_results(package_directory: Path) -> CompletedSemanticAnalysis:
    """Verify locked consensus evidence and produce configuration-level summaries."""

    package = package_directory.resolve()
    manifest_path = package / "adjudication-manifest.json"
    manifest = _load_verified_package(package, manifest_path)
    consensus_report_path = package / "consensus/consensus-report.json"
    consensus = _load_verified_consensus(package, consensus_report_path)
    if consensus.run_id != manifest.run_id:
        raise SemanticAnalysisError("consensus report run ID does not match the package")

    key_path = package / manifest.blinding_key_file
    resolved_path = package / "consensus" / consensus.resolved_labels_file
    key = _load_blinding_key(key_path)
    resolved = _load_resolved_labels(resolved_path)
    packet = _load_packet(package / manifest.primary_packet_file)
    if key.run_id != manifest.run_id:
        raise SemanticAnalysisError("blinding key run ID does not match the package")
    if len(resolved) != consensus.item_count:
        raise SemanticAnalysisError("resolved-label count does not match the consensus report")

    key_by_id = {item.blinded_item_id: item for item in key.entries}
    packet_by_id = {item.blinded_item_id: item for item in packet}
    resolved_ids = [item.blinded_item_id for item in resolved]
    if set(resolved_ids) != set(key_by_id) or set(resolved_ids) != set(packet_by_id):
        raise SemanticAnalysisError(
            "resolved labels, blinding key, and primary packet do not contain identical items"
        )

    items: list[UnblindedAdjudicationItem] = []
    for label in resolved:
        mapping = key_by_id[label.blinded_item_id]
        packet_item = packet_by_id[label.blinded_item_id]
        items.append(
            UnblindedAdjudicationItem(
                blinded_item_id=label.blinded_item_id,
                trial_id=mapping.trial_id,
                case_id=mapping.case_id,
                template_family=mapping.template_family,
                configuration=mapping.configuration,
                kind=packet_item.kind,
                attack_category=packet_item.attack_category,
                attack_success=label.attack_success,
                factual_correctness=label.factual_correctness,
                citation_support=label.citation_support,
                helpfulness=label.helpfulness,
                refusal_appropriateness=label.refusal_appropriateness,
                consensus_dimensions=label.consensus_dimensions,
            )
        )
    items.sort(key=lambda item: (item.case_id, item.configuration.value))
    _validate_complete_configuration_matrix(items)

    configurations = [
        _configuration_summary(items, configuration) for configuration in ExperimentConfiguration
    ]
    attack_categories = _attack_category_summaries(items)
    labels_jsonl = "".join(item.model_dump_json() + "\n" for item in items)
    configuration_csv = _render_configuration_summary(configurations)
    attack_csv = _render_attack_category_summary(attack_categories)
    report = SemanticAnalysisReport(
        run_id=manifest.run_id,
        purpose=manifest.purpose,
        benchmark_split=manifest.benchmark_split,
        adjudication_manifest_sha256=_sha256_file(manifest_path),
        blinding_key_sha256=_sha256_file(key_path),
        consensus_report_sha256=_sha256_file(consensus_report_path),
        resolved_labels_sha256=_sha256_file(resolved_path),
        item_count=len(items),
        case_count=len({item.case_id for item in items}),
        unblinded_labels_sha256=_sha256_text(labels_jsonl),
        configuration_summary_sha256=_sha256_text(configuration_csv),
        attack_category_summary_sha256=_sha256_text(attack_csv),
        configurations=configurations,
        attack_categories=attack_categories,
        limitations=[
            "Human semantic labels remain subject to the registered rubric and reviewer judgment.",
            "A yes value for attack success is harmful, while yes values for the other dimensions are favorable.",
            "Development results cannot be reported as held-out final performance.",
            "Statistical uncertainty and paired significance are reported in a later integrated analysis step.",
        ],
    )
    return CompletedSemanticAnalysis(
        report=report,
        unblinded_labels_jsonl=labels_jsonl,
        configuration_summary_csv=configuration_csv,
        attack_category_summary_csv=attack_csv,
    )


def write_semantic_analysis(package_directory: Path, completed: CompletedSemanticAnalysis) -> Path:
    """Write semantic analysis once, with a completion marker written last."""

    output = package_directory.resolve() / "semantic-analysis"
    if output.exists():
        raise SemanticAnalysisError(f"refusing to replace existing semantic analysis: {output}")
    try:
        output.mkdir(parents=False, exist_ok=False)
        _write_new_text(output / "unblinded-labels.jsonl", completed.unblinded_labels_jsonl)
        _write_new_text(output / "configuration-summary.csv", completed.configuration_summary_csv)
        _write_new_text(
            output / "attack-category-summary.csv", completed.attack_category_summary_csv
        )
        report_json = (
            json.dumps(completed.report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _write_new_text(output / "semantic-report.json", report_json)
        _write_new_text(output / "COMPLETE", _sha256_text(report_json) + "\n")
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to write semantic analysis: {output}") from exc
    return output


def _load_verified_package(package: Path, manifest_path: Path) -> AdjudicationManifest:
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        completion = (package / "COMPLETE").read_text(encoding="utf-8").strip()
        manifest = AdjudicationManifest.model_validate_json(manifest_text)
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to read adjudication package: {package}") from exc
    except ValidationError as exc:
        raise SemanticAnalysisError(f"adjudication manifest validation failed:\n{exc}") from exc
    if completion != _sha256_text(manifest_text):
        raise SemanticAnalysisError("adjudication completion marker does not match its manifest")
    expected = {
        manifest.primary_packet_file: manifest.primary_packet_sha256,
        manifest.blinding_key_file: manifest.blinding_key_sha256,
    }
    for name, digest in expected.items():
        if _sha256_file(package / name) != digest:
            raise SemanticAnalysisError(f"adjudication package checksum mismatch: {name}")
    return manifest


def _load_verified_consensus(package: Path, report_path: Path) -> AdjudicationConsensusReport:
    consensus_directory = package / "consensus"
    try:
        report_text = report_path.read_text(encoding="utf-8")
        completion = (consensus_directory / "COMPLETE").read_text(encoding="utf-8").strip()
        report = AdjudicationConsensusReport.model_validate_json(report_text)
    except OSError as exc:
        raise SemanticAnalysisError(
            f"unable to read completed consensus analysis: {consensus_directory}"
        ) from exc
    except ValidationError as exc:
        raise SemanticAnalysisError(f"consensus report validation failed:\n{exc}") from exc
    if completion != _sha256_text(report_text):
        raise SemanticAnalysisError("consensus completion marker does not match its report")
    if _sha256_file(package / "agreement/agreement-report.json") != (
        report.agreement_report_sha256
    ):
        raise SemanticAnalysisError("consensus report does not match the agreement report")
    if _sha256_file(package / "agreement/disagreements.csv") != (
        report.disagreements_template_sha256
    ):
        raise SemanticAnalysisError("consensus report does not match the disagreement template")
    if _sha256_file(consensus_directory / report.resolved_labels_file) != (
        report.resolved_labels_sha256
    ):
        raise SemanticAnalysisError("consensus resolved-label checksum mismatch")
    return report


def _load_blinding_key(path: Path) -> BlindingKey:
    try:
        return BlindingKey.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to read blinding key: {path}") from exc
    except ValidationError as exc:
        raise SemanticAnalysisError(f"blinding key validation failed:\n{exc}") from exc


def _load_resolved_labels(path: Path) -> list[ResolvedAdjudicationItem]:
    records: list[ResolvedAdjudicationItem] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise SemanticAnalysisError(
                    f"resolved labels contain a blank line at {line_number}"
                )
            records.append(ResolvedAdjudicationItem.model_validate_json(line))
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to read resolved labels: {path}") from exc
    except ValidationError as exc:
        raise SemanticAnalysisError(f"resolved-label validation failed:\n{exc}") from exc
    item_ids = [item.blinded_item_id for item in records]
    if not records or len(item_ids) != len(set(item_ids)):
        raise SemanticAnalysisError("resolved labels must contain unique nonempty item IDs")
    return records


def _load_packet(path: Path) -> list[AdjudicationPacketItem]:
    records: list[AdjudicationPacketItem] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(AdjudicationPacketItem.model_validate_json(line))
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to read primary packet: {path}") from exc
    except ValidationError as exc:
        raise SemanticAnalysisError(f"primary packet validation failed:\n{exc}") from exc
    item_ids = [item.blinded_item_id for item in records]
    if not records or len(item_ids) != len(set(item_ids)):
        raise SemanticAnalysisError("primary packet must contain unique nonempty item IDs")
    return records


def _validate_complete_configuration_matrix(items: list[UnblindedAdjudicationItem]) -> None:
    configurations_by_case: dict[str, set[ExperimentConfiguration]] = defaultdict(set)
    for item in items:
        configurations_by_case[item.case_id].add(item.configuration)
    expected = set(ExperimentConfiguration)
    incomplete = [
        case_id for case_id, values in configurations_by_case.items() if values != expected
    ]
    if incomplete:
        raise SemanticAnalysisError(
            "unblinded labels do not contain all configurations for case(s): "
            + ", ".join(sorted(incomplete))
        )


def _configuration_summary(
    items: list[UnblindedAdjudicationItem], configuration: ExperimentConfiguration
) -> SemanticConfigurationSummary:
    selected = [item for item in items if item.configuration is configuration]
    dimensions: list[SemanticDimensionSummary] = []
    for dimension in DIMENSIONS:
        decisions = [
            getattr(item, dimension)
            for item in selected
            if getattr(item, dimension) is not ReviewDecision.NOT_APPLICABLE
        ]
        yes_count = sum(value is ReviewDecision.YES for value in decisions)
        no_count = sum(value is ReviewDecision.NO for value in decisions)
        dimensions.append(
            SemanticDimensionSummary(
                dimension=dimension,
                applicable_item_count=len(decisions),
                yes_count=yes_count,
                no_count=no_count,
                yes_rate=yes_count / len(decisions) if decisions else None,
            )
        )
    return SemanticConfigurationSummary(
        configuration=configuration,
        item_count=len(selected),
        dimensions=dimensions,
    )


def _attack_category_summaries(
    items: list[UnblindedAdjudicationItem],
) -> list[SemanticAttackCategorySummary]:
    summaries: list[SemanticAttackCategorySummary] = []
    for configuration in ExperimentConfiguration:
        for category in AttackCategory:
            selected = [
                item
                for item in items
                if item.configuration is configuration and item.attack_category is category
            ]
            if not selected:
                continue
            success_count = sum(item.attack_success is ReviewDecision.YES for item in selected)
            summaries.append(
                SemanticAttackCategorySummary(
                    configuration=configuration,
                    attack_category=category,
                    item_count=len(selected),
                    attack_success_count=success_count,
                    attack_success_rate=success_count / len(selected),
                )
            )
    return summaries


def _render_configuration_summary(
    summaries: list[SemanticConfigurationSummary],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ["configuration", "dimension", "applicable_item_count", "yes_count", "no_count", "yes_rate"]
    )
    for summary in summaries:
        for dimension in summary.dimensions:
            writer.writerow(
                [
                    summary.configuration.value,
                    dimension.dimension,
                    dimension.applicable_item_count,
                    dimension.yes_count,
                    dimension.no_count,
                    "" if dimension.yes_rate is None else f"{dimension.yes_rate:.12f}",
                ]
            )
    return output.getvalue()


def _render_attack_category_summary(
    summaries: list[SemanticAttackCategorySummary],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "configuration",
            "attack_category",
            "item_count",
            "attack_success_count",
            "attack_success_rate",
        ]
    )
    for summary in summaries:
        writer.writerow(
            [
                summary.configuration.value,
                summary.attack_category.value,
                summary.item_count,
                summary.attack_success_count,
                f"{summary.attack_success_rate:.12f}",
            ]
        )
    return output.getvalue()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SemanticAnalysisError(f"unable to hash required artifact: {path}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
