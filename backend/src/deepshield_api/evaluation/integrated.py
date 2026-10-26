"""Integrated automated and human-semantic experiment analysis."""

import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deepshield_api.corpus.models import Sha256Digest, Slug
from deepshield_api.evaluation.adjudication import ReviewDecision
from deepshield_api.evaluation.agreement import DIMENSIONS, DimensionName
from deepshield_api.evaluation.models import AttackCategory, BenchmarkKind, BenchmarkSplit
from deepshield_api.evaluation.runs import ExperimentConfiguration, ExperimentPurpose
from deepshield_api.evaluation.scoring import ExperimentScoreReport, MetricEstimate, TrialScore
from deepshield_api.evaluation.semantic import (
    SemanticAnalysisReport,
    UnblindedAdjudicationItem,
)

INTEGRATED_ANALYSIS_PROTOCOL_ID: Literal["deepshield-integrated-analysis-v1"] = (
    "deepshield-integrated-analysis-v1"
)
BOOTSTRAP_ITERATIONS = 2_000
CONFIDENCE_LEVEL = 0.95
REGISTERED_PAIRS: tuple[tuple[ExperimentConfiguration, ExperimentConfiguration], ...] = (
    (ExperimentConfiguration.BASELINE, ExperimentConfiguration.HARDENED),
    (ExperimentConfiguration.HARDENED, ExperimentConfiguration.DEEPSHIELD),
    (ExperimentConfiguration.BASELINE, ExperimentConfiguration.DEEPSHIELD),
)
ReportModel = TypeVar("ReportModel", bound=BaseModel)


class IntegratedAnalysisError(RuntimeError):
    """Raised when automated and semantic evidence cannot be combined safely."""


class IntegratedTrialResult(BaseModel):
    """Sanitized automated and human labels for one trial."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    trial_id: str = Field(min_length=1)
    case_id: Slug
    template_family: Slug
    configuration: ExperimentConfiguration
    kind: BenchmarkKind
    attack_category: AttackCategory | None
    coarse_action_alignment: bool | None
    exact_forbidden_leakage: bool | None
    retrieval_recall_at_5: float | None = Field(default=None, ge=0, le=1)
    retrieval_reciprocal_rank_at_5: float | None = Field(default=None, ge=0, le=1)
    has_valid_citation: bool
    guardrail_blocked: bool
    duration_ms: int = Field(ge=0)
    attack_success: ReviewDecision
    factual_correctness: ReviewDecision
    citation_support: ReviewDecision
    helpfulness: ReviewDecision
    refusal_appropriateness: ReviewDecision


class IntegratedConfigurationSummary(BaseModel):
    """Primary automated and human metrics for one configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: ExperimentConfiguration
    case_count: int = Field(ge=1)
    coarse_action_alignment: MetricEstimate
    exact_sensitive_data_leakage: MetricEstimate
    retrieval_recall_at_5: MetricEstimate
    attack_success: MetricEstimate
    factual_correctness: MetricEstimate
    citation_support: MetricEstimate
    helpfulness: MetricEstimate
    refusal_appropriateness: MetricEstimate


class SemanticPairedComparison(BaseModel):
    """Paired human-label comparison with clustered uncertainty."""

    model_config = ConfigDict(extra="forbid")

    dimension: DimensionName
    direction: Literal["higher-is-better", "lower-is-better"]
    reference_configuration: ExperimentConfiguration
    candidate_configuration: ExperimentConfiguration
    reference_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    candidate_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    difference: float = Field(ge=-1, le=1, allow_inf_nan=False)
    confidence_lower: float = Field(ge=-1, le=1, allow_inf_nan=False)
    confidence_upper: float = Field(ge=-1, le=1, allow_inf_nan=False)
    sample_count: int = Field(ge=1)
    cluster_count: int = Field(ge=1)
    reference_only_yes: int = Field(ge=0)
    candidate_only_yes: int = Field(ge=0)
    mcnemar_exact_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    holm_adjusted_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    bootstrap_iterations: int = Field(default=BOOTSTRAP_ITERATIONS, ge=1)
    confidence_level: float = Field(default=CONFIDENCE_LEVEL, gt=0, lt=1)

    @model_validator(mode="after")
    def require_consistent_difference(self) -> Self:
        expected = self.candidate_rate - self.reference_rate
        if not math.isclose(self.difference, expected, abs_tol=1e-12):
            raise ValueError("paired difference must be candidate rate minus reference rate")
        if not self.confidence_lower <= self.difference <= self.confidence_upper:
            raise ValueError("paired difference must lie inside its confidence interval")
        return self


class SecurityUtilityPoint(BaseModel):
    """Figure-ready point showing the primary safety-utility trade-off."""

    model_config = ConfigDict(extra="forbid")

    configuration: ExperimentConfiguration
    attack_success_rate: float = Field(ge=0, le=1)
    factual_correctness_rate: float = Field(ge=0, le=1)
    helpfulness_rate: float = Field(ge=0, le=1)
    exact_sensitive_data_leakage_rate: float = Field(ge=0, le=1)


class ThreatHeatmapCell(BaseModel):
    """Figure-ready human attack-success cell for one threat."""

    model_config = ConfigDict(extra="forbid")

    configuration: ExperimentConfiguration
    attack_category: AttackCategory
    item_count: int = Field(ge=1)
    attack_success_rate: float = Field(ge=0, le=1)


class IntegratedAnalysisReport(BaseModel):
    """Checksum-backed integrated evidence contract for tables and figures."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["deepshield-integrated-analysis-v1"] = INTEGRATED_ANALYSIS_PROTOCOL_ID
    run_id: Slug
    purpose: ExperimentPurpose
    benchmark_split: BenchmarkSplit
    score_report_sha256: Sha256Digest
    semantic_report_sha256: Sha256Digest
    automated_trial_scores_sha256: Sha256Digest
    unblinded_labels_sha256: Sha256Digest
    item_count: int = Field(ge=1)
    case_count: int = Field(ge=1)
    integrated_trials_file: Literal["integrated-trials.jsonl"] = "integrated-trials.jsonl"
    integrated_trials_sha256: Sha256Digest
    configuration_summary_file: Literal["configuration-summary.csv"] = "configuration-summary.csv"
    configuration_summary_sha256: Sha256Digest
    paired_comparisons_file: Literal["semantic-paired-comparisons.csv"] = (
        "semantic-paired-comparisons.csv"
    )
    paired_comparisons_sha256: Sha256Digest
    security_utility_figure_file: Literal["figure-security-utility.csv"] = (
        "figure-security-utility.csv"
    )
    security_utility_figure_sha256: Sha256Digest
    threat_heatmap_figure_file: Literal["figure-threat-heatmap.csv"] = "figure-threat-heatmap.csv"
    threat_heatmap_figure_sha256: Sha256Digest
    configurations: list[IntegratedConfigurationSummary] = Field(min_length=3, max_length=3)
    semantic_paired_comparisons: list[SemanticPairedComparison] = Field(
        min_length=15, max_length=15
    )
    security_utility_points: list[SecurityUtilityPoint] = Field(min_length=3, max_length=3)
    threat_heatmap: list[ThreatHeatmapCell]
    limitations: list[str] = Field(min_length=4)

    @model_validator(mode="after")
    def require_registered_design(self) -> Self:
        if [item.configuration for item in self.configurations] != list(ExperimentConfiguration):
            raise ValueError("integrated configuration summaries must be ordered C0, C1, C2")
        expected = [
            (reference, candidate, dimension)
            for reference, candidate in REGISTERED_PAIRS
            for dimension in DIMENSIONS
        ]
        received = [
            (
                item.reference_configuration,
                item.candidate_configuration,
                item.dimension,
            )
            for item in self.semantic_paired_comparisons
        ]
        if received != expected:
            raise ValueError("semantic paired comparisons do not match the registered design")
        if [item.configuration for item in self.security_utility_points] != list(
            ExperimentConfiguration
        ):
            raise ValueError("security-utility points must be ordered C0, C1, C2")
        if self.item_count != self.case_count * len(ExperimentConfiguration):
            raise ValueError("integrated evidence requires one item per case and configuration")
        return self


class CompletedIntegratedAnalysis(BaseModel):
    """Validated integrated artifacts ready for immutable persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: IntegratedAnalysisReport
    integrated_trials_jsonl: str
    configuration_summary_csv: str
    paired_comparisons_csv: str
    security_utility_csv: str
    threat_heatmap_csv: str


def render_integrated_report_schema() -> str:
    """Return the canonical integrated-analysis report JSON Schema."""

    return json.dumps(IntegratedAnalysisReport.model_json_schema(), indent=2, sort_keys=True) + "\n"


def write_integrated_report_schema(path: Path) -> None:
    """Write the canonical integrated-analysis report schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_integrated_report_schema(), encoding="utf-8", newline="\n")


def integrate_completed_analysis(run_directory: Path) -> CompletedIntegratedAnalysis:
    """Verify and combine completed automated and semantic experiment evidence."""

    run = run_directory.resolve()
    score_path = run / "analysis/score-report.json"
    semantic_path = run / "adjudication/semantic-analysis/semantic-report.json"
    score = _load_score_analysis(run / "analysis", score_path)
    semantic = _load_semantic_analysis(run / "adjudication/semantic-analysis", semantic_path)
    if (
        score.run_id != semantic.run_id
        or score.purpose is not semantic.purpose
        or score.benchmark_split is not semantic.benchmark_split
    ):
        raise IntegratedAnalysisError(
            "automated and semantic reports do not describe the same experiment"
        )

    score_file = run / "analysis" / score.trial_scores_file
    semantic_file = run / "adjudication/semantic-analysis" / semantic.unblinded_labels_file
    automated = _load_jsonl(score_file, TrialScore, "automated trial scores")
    human = _load_jsonl(semantic_file, UnblindedAdjudicationItem, "unblinded semantic labels")
    combined = _join_trials(automated, human)
    configurations = [
        _configuration_summary(combined, score, configuration)
        for configuration in ExperimentConfiguration
    ]
    comparisons = _semantic_paired_comparisons(combined)
    points = _security_utility_points(configurations)
    heatmap = [
        ThreatHeatmapCell(
            configuration=item.configuration,
            attack_category=item.attack_category,
            item_count=item.item_count,
            attack_success_rate=item.attack_success_rate,
        )
        for item in semantic.attack_categories
    ]
    trials_jsonl = "".join(item.model_dump_json() + "\n" for item in combined)
    configuration_csv = _render_configuration_summary(configurations)
    comparisons_csv = _render_paired_comparisons(comparisons)
    security_utility_csv = _render_security_utility(points)
    heatmap_csv = _render_threat_heatmap(heatmap)
    report = IntegratedAnalysisReport(
        run_id=score.run_id,
        purpose=score.purpose,
        benchmark_split=score.benchmark_split,
        score_report_sha256=_sha256_file(score_path),
        semantic_report_sha256=_sha256_file(semantic_path),
        automated_trial_scores_sha256=_sha256_file(score_file),
        unblinded_labels_sha256=_sha256_file(semantic_file),
        item_count=len(combined),
        case_count=len({item.case_id for item in combined}),
        integrated_trials_sha256=_sha256_text(trials_jsonl),
        configuration_summary_sha256=_sha256_text(configuration_csv),
        paired_comparisons_sha256=_sha256_text(comparisons_csv),
        security_utility_figure_sha256=_sha256_text(security_utility_csv),
        threat_heatmap_figure_sha256=_sha256_text(heatmap_csv),
        configurations=configurations,
        semantic_paired_comparisons=comparisons,
        security_utility_points=points,
        threat_heatmap=heatmap,
        limitations=[
            "A completed development analysis cannot be reported as held-out final performance.",
            "Human semantic outcomes remain subject to the registered rubric and reviewer judgment.",
            "McNemar tests treat cases as paired observations; clustered bootstrap intervals are primary because template-family paraphrases are related.",
            "Exact leakage covers only registered synthetic secrets and fictional PII.",
            "Figure-ready CSV files contain measured values only and are not rendered evidence by themselves.",
        ],
    )
    return CompletedIntegratedAnalysis(
        report=report,
        integrated_trials_jsonl=trials_jsonl,
        configuration_summary_csv=configuration_csv,
        paired_comparisons_csv=comparisons_csv,
        security_utility_csv=security_utility_csv,
        threat_heatmap_csv=heatmap_csv,
    )


def write_integrated_analysis(run_directory: Path, completed: CompletedIntegratedAnalysis) -> Path:
    """Write integrated analysis once, with its completion marker last."""

    output = run_directory.resolve() / "integrated-analysis"
    if output.exists():
        raise IntegratedAnalysisError(f"refusing to replace existing integrated analysis: {output}")
    try:
        output.mkdir(parents=False, exist_ok=False)
        artifacts = {
            "integrated-trials.jsonl": completed.integrated_trials_jsonl,
            "configuration-summary.csv": completed.configuration_summary_csv,
            "semantic-paired-comparisons.csv": completed.paired_comparisons_csv,
            "figure-security-utility.csv": completed.security_utility_csv,
            "figure-threat-heatmap.csv": completed.threat_heatmap_csv,
        }
        for name, content in artifacts.items():
            _write_new_text(output / name, content)
        report_json = (
            json.dumps(completed.report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        _write_new_text(output / "integrated-report.json", report_json)
        _write_new_text(output / "COMPLETE", _sha256_text(report_json) + "\n")
    except OSError as exc:
        raise IntegratedAnalysisError(f"unable to write integrated analysis: {output}") from exc
    return output


def _load_score_analysis(directory: Path, report_path: Path) -> ExperimentScoreReport:
    report = _load_completed_report(
        directory, report_path, ExperimentScoreReport, "automated score"
    )
    expected = {
        report.trial_scores_file: report.trial_scores_sha256,
        report.configuration_summary_file: report.configuration_summary_sha256,
        report.attack_category_summary_file: report.attack_category_summary_sha256,
        report.paired_comparison_file: report.paired_comparison_sha256,
    }
    _verify_artifacts(directory, expected, "automated analysis")
    return report


def _load_semantic_analysis(directory: Path, report_path: Path) -> SemanticAnalysisReport:
    report = _load_completed_report(directory, report_path, SemanticAnalysisReport, "semantic")
    expected = {
        report.unblinded_labels_file: report.unblinded_labels_sha256,
        report.configuration_summary_file: report.configuration_summary_sha256,
        report.attack_category_summary_file: report.attack_category_summary_sha256,
    }
    _verify_artifacts(directory, expected, "semantic analysis")
    return report


def _load_completed_report(
    directory: Path,
    report_path: Path,
    model: type[ReportModel],
    label: str,
) -> ReportModel:
    try:
        report_text = report_path.read_text(encoding="utf-8")
        completion = (directory / "COMPLETE").read_text(encoding="utf-8").strip()
        report = model.model_validate_json(report_text)
    except OSError as exc:
        raise IntegratedAnalysisError(
            f"unable to read completed {label} analysis: {directory}"
        ) from exc
    except ValidationError as exc:
        raise IntegratedAnalysisError(f"{label} report validation failed:\n{exc}") from exc
    if completion != _sha256_text(report_text):
        raise IntegratedAnalysisError(f"{label} completion marker does not match its report")
    return report


def _verify_artifacts(directory: Path, expected: dict[str, str], label: str) -> None:
    for name, digest in expected.items():
        if _sha256_file(directory / name) != digest:
            raise IntegratedAnalysisError(f"{label} checksum mismatch: {name}")


def _load_jsonl(path: Path, model: type[BaseModel], label: str) -> list[BaseModel]:
    records: list[BaseModel] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise IntegratedAnalysisError(f"{label} contain a blank line at {line_number}")
            records.append(model.model_validate_json(line))
    except OSError as exc:
        raise IntegratedAnalysisError(f"unable to read {label}: {path}") from exc
    except ValidationError as exc:
        raise IntegratedAnalysisError(f"{label} validation failed:\n{exc}") from exc
    if not records:
        raise IntegratedAnalysisError(f"{label} must not be empty")
    return records


def _join_trials(
    automated_records: list[BaseModel], human_records: list[BaseModel]
) -> list[IntegratedTrialResult]:
    automated = [TrialScore.model_validate(item.model_dump()) for item in automated_records]
    human = [UnblindedAdjudicationItem.model_validate(item.model_dump()) for item in human_records]
    automated_by_id = {item.trial_id: item for item in automated}
    human_by_id = {item.trial_id: item for item in human}
    if len(automated_by_id) != len(automated) or len(human_by_id) != len(human):
        raise IntegratedAnalysisError("analysis inputs contain duplicate trial IDs")
    if set(automated_by_id) != set(human_by_id):
        raise IntegratedAnalysisError(
            "automated and semantic evidence do not contain identical trial IDs"
        )
    combined: list[IntegratedTrialResult] = []
    for trial_id in sorted(automated_by_id):
        score = automated_by_id[trial_id]
        label = human_by_id[trial_id]
        identity = (
            score.case_id,
            score.template_family,
            score.configuration,
            score.kind,
            score.attack_category,
        )
        semantic_identity = (
            label.case_id,
            label.template_family,
            label.configuration,
            label.kind,
            label.attack_category,
        )
        if identity != semantic_identity:
            raise IntegratedAnalysisError(
                f"automated and semantic trial identity differs: {trial_id}"
            )
        combined.append(
            IntegratedTrialResult(
                trial_id=trial_id,
                case_id=score.case_id,
                template_family=score.template_family,
                configuration=score.configuration,
                kind=score.kind,
                attack_category=score.attack_category,
                coarse_action_alignment=score.coarse_action_alignment,
                exact_forbidden_leakage=score.exact_forbidden_leakage,
                retrieval_recall_at_5=score.retrieval_recall_at_5,
                retrieval_reciprocal_rank_at_5=score.retrieval_reciprocal_rank_at_5,
                has_valid_citation=score.has_valid_citation,
                guardrail_blocked=score.guardrail_blocked,
                duration_ms=score.duration_ms,
                attack_success=label.attack_success,
                factual_correctness=label.factual_correctness,
                citation_support=label.citation_support,
                helpfulness=label.helpfulness,
                refusal_appropriateness=label.refusal_appropriateness,
            )
        )
    expected = len({item.case_id for item in combined}) * len(ExperimentConfiguration)
    if len(combined) != expected:
        raise IntegratedAnalysisError(
            "integrated evidence does not contain one result per case and configuration"
        )
    configurations_by_case: dict[str, set[ExperimentConfiguration]] = defaultdict(set)
    for item in combined:
        configurations_by_case[item.case_id].add(item.configuration)
    required_configurations = set(ExperimentConfiguration)
    if any(
        configurations != required_configurations
        for configurations in configurations_by_case.values()
    ):
        raise IntegratedAnalysisError(
            "integrated evidence does not contain exactly C0, C1, and C2 for every case"
        )
    return combined


def _configuration_summary(
    items: list[IntegratedTrialResult],
    score: ExperimentScoreReport,
    configuration: ExperimentConfiguration,
) -> IntegratedConfigurationSummary:
    selected = [item for item in items if item.configuration is configuration]
    automated = next(
        item for item in score.configuration_summaries if item.configuration is configuration
    )
    if len(selected) != automated.case_count:
        raise IntegratedAnalysisError(
            f"integrated trial count disagrees with automated report for {configuration.value}"
        )
    return IntegratedConfigurationSummary(
        configuration=configuration,
        case_count=len(selected),
        coarse_action_alignment=automated.coarse_action_alignment,
        exact_sensitive_data_leakage=automated.exact_sensitive_data_leakage_rate,
        retrieval_recall_at_5=automated.retrieval_recall_at_5,
        attack_success=_semantic_estimate(selected, "attack_success"),
        factual_correctness=_semantic_estimate(selected, "factual_correctness"),
        citation_support=_semantic_estimate(selected, "citation_support"),
        helpfulness=_semantic_estimate(selected, "helpfulness"),
        refusal_appropriateness=_semantic_estimate(selected, "refusal_appropriateness"),
    )


def _semantic_estimate(
    items: list[IntegratedTrialResult], dimension: DimensionName
) -> MetricEstimate:
    observations = [
        (item.template_family, float(value is ReviewDecision.YES))
        for item in items
        if (value := getattr(item, dimension)) is not ReviewDecision.NOT_APPLICABLE
    ]
    if not observations:
        raise IntegratedAnalysisError(
            f"registered semantic dimension has no observations: {dimension}"
        )
    point = sum(value for _, value in observations) / len(observations)
    lower, upper = _cluster_bootstrap_interval(observations, _stable_seed(dimension, observations))
    return MetricEstimate(
        value=point,
        confidence_lower=min(lower, point),
        confidence_upper=max(upper, point),
        sample_count=len(observations),
        cluster_count=len({family for family, _ in observations}),
    )


def _semantic_paired_comparisons(
    items: list[IntegratedTrialResult],
) -> list[SemanticPairedComparison]:
    comparisons = [
        _paired_semantic(items, dimension, reference, candidate)
        for reference, candidate in REGISTERED_PAIRS
        for dimension in DIMENSIONS
    ]
    ordered = sorted(enumerate(comparisons), key=lambda item: item[1].mcnemar_exact_p_value)
    adjusted = [1.0] * len(comparisons)
    running_max = 0.0
    total = len(comparisons)
    for rank, (index, comparison) in enumerate(ordered):
        corrected = min(1.0, (total - rank) * comparison.mcnemar_exact_p_value)
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return [
        comparison.model_copy(update={"holm_adjusted_p_value": adjusted[index]})
        for index, comparison in enumerate(comparisons)
    ]


def _paired_semantic(
    items: list[IntegratedTrialResult],
    dimension: DimensionName,
    reference: ExperimentConfiguration,
    candidate: ExperimentConfiguration,
) -> SemanticPairedComparison:
    by_key = {(item.case_id, item.configuration): item for item in items}
    paired: list[tuple[str, bool, bool]] = []
    for item in items:
        if item.configuration is not reference:
            continue
        reference_value = getattr(item, dimension)
        candidate_value = getattr(by_key[(item.case_id, candidate)], dimension)
        if reference_value is ReviewDecision.NOT_APPLICABLE:
            if candidate_value is not ReviewDecision.NOT_APPLICABLE:
                raise IntegratedAnalysisError(
                    f"semantic applicability differs across configurations: {item.case_id}"
                )
            continue
        if candidate_value is ReviewDecision.NOT_APPLICABLE:
            raise IntegratedAnalysisError(
                f"semantic applicability differs across configurations: {item.case_id}"
            )
        paired.append(
            (
                item.template_family,
                reference_value is ReviewDecision.YES,
                candidate_value is ReviewDecision.YES,
            )
        )
    if not paired:
        raise IntegratedAnalysisError(
            f"registered paired semantic metric has no observations: {dimension}"
        )
    reference_rate = sum(reference_value for _, reference_value, _ in paired) / len(paired)
    candidate_rate = sum(candidate_value for _, _, candidate_value in paired) / len(paired)
    differences = [
        (family, float(candidate_value) - float(reference_value))
        for family, reference_value, candidate_value in paired
    ]
    difference = candidate_rate - reference_rate
    lower, upper = _cluster_bootstrap_interval(
        differences,
        _stable_seed(dimension, differences, reference.value, candidate.value),
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    reference_only = sum(
        reference_value and not candidate_value for _, reference_value, candidate_value in paired
    )
    candidate_only = sum(
        candidate_value and not reference_value for _, reference_value, candidate_value in paired
    )
    p_value = _mcnemar_exact(reference_only, candidate_only)
    return SemanticPairedComparison(
        dimension=dimension,
        direction=("lower-is-better" if dimension == "attack_success" else "higher-is-better"),
        reference_configuration=reference,
        candidate_configuration=candidate,
        reference_rate=reference_rate,
        candidate_rate=candidate_rate,
        difference=difference,
        confidence_lower=min(lower, difference),
        confidence_upper=max(upper, difference),
        sample_count=len(paired),
        cluster_count=len({family for family, _, _ in paired}),
        reference_only_yes=reference_only,
        candidate_only_yes=candidate_only,
        mcnemar_exact_p_value=p_value,
        holm_adjusted_p_value=p_value,
    )


def _security_utility_points(
    summaries: list[IntegratedConfigurationSummary],
) -> list[SecurityUtilityPoint]:
    return [
        SecurityUtilityPoint(
            configuration=item.configuration,
            attack_success_rate=item.attack_success.value,
            factual_correctness_rate=item.factual_correctness.value,
            helpfulness_rate=item.helpfulness.value,
            exact_sensitive_data_leakage_rate=item.exact_sensitive_data_leakage.value,
        )
        for item in summaries
    ]


def _cluster_bootstrap_interval(
    observations: list[tuple[str, float]],
    seed: int,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for family, value in observations:
        clusters[family].append(value)
    keys = sorted(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        selected = [rng.choice(keys) for _ in keys]
        values = [value for key in selected for value in clusters[key]]
        samples.append(sum(values) / len(values))
    samples.sort()
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return (
        max(lower_bound, _quantile(samples, alpha)),
        min(upper_bound, _quantile(samples, 1 - alpha)),
    )


def _mcnemar_exact(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = float(
        sum(
            math.comb(discordant, value) for value in range(min(reference_only, candidate_only) + 1)
        )
    ) / float(2**discordant)
    return min(1.0, 2 * tail)


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _render_configuration_summary(
    summaries: list[IntegratedConfigurationSummary],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "configuration",
            "case_count",
            "coarse_action_alignment",
            "exact_sensitive_data_leakage",
            "retrieval_recall_at_5",
            "attack_success",
            "factual_correctness",
            "citation_support",
            "helpfulness",
            "refusal_appropriateness",
        ]
    )
    for item in summaries:
        writer.writerow(
            [
                item.configuration.value,
                item.case_count,
                _format_metric(item.coarse_action_alignment),
                _format_metric(item.exact_sensitive_data_leakage),
                _format_metric(item.retrieval_recall_at_5),
                _format_metric(item.attack_success),
                _format_metric(item.factual_correctness),
                _format_metric(item.citation_support),
                _format_metric(item.helpfulness),
                _format_metric(item.refusal_appropriateness),
            ]
        )
    return output.getvalue()


def _render_paired_comparisons(
    comparisons: list[SemanticPairedComparison],
) -> str:
    output = io.StringIO(newline="")
    fields = list(SemanticPairedComparison.model_fields)
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in comparisons:
        writer.writerow(item.model_dump(mode="json"))
    return output.getvalue()


def _render_security_utility(points: list[SecurityUtilityPoint]) -> str:
    output = io.StringIO(newline="")
    fields = list(SecurityUtilityPoint.model_fields)
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in points:
        writer.writerow(item.model_dump(mode="json"))
    return output.getvalue()


def _render_threat_heatmap(cells: list[ThreatHeatmapCell]) -> str:
    output = io.StringIO(newline="")
    fields = list(ThreatHeatmapCell.model_fields)
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in cells:
        writer.writerow(item.model_dump(mode="json"))
    return output.getvalue()


def _format_metric(metric: MetricEstimate) -> str:
    return f"{metric.value:.12f} [{metric.confidence_lower:.12f}, {metric.confidence_upper:.12f}]"


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise IntegratedAnalysisError(f"unable to hash required artifact: {path}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
