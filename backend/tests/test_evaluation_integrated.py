import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.adjudication import ReviewDecision
from deepshield_api.evaluation.agreement import DimensionName
from deepshield_api.evaluation.integrated import (
    IntegratedAnalysisError,
    integrate_completed_analysis,
    render_integrated_report_schema,
    write_integrated_analysis,
)
from deepshield_api.evaluation.models import (
    AttackCategory,
    BenchmarkKind,
    BenchmarkSplit,
    ExpectedAction,
)
from deepshield_api.evaluation.runs import (
    ExperimentConfiguration,
    ExperimentPurpose,
    ObservedOutcome,
)
from deepshield_api.evaluation.scoring import (
    AttackCategoryScoreSummary,
    ConfigurationScoreSummary,
    DistributionSummary,
    ExperimentScoreReport,
    MetricEstimate,
    PairedBinaryComparison,
    ResourceSummary,
    TrialScore,
)
from deepshield_api.evaluation.semantic import (
    SemanticAnalysisReport,
    SemanticAttackCategorySummary,
    SemanticConfigurationSummary,
    SemanticDimensionSummary,
    UnblindedAdjudicationItem,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "data/evaluation/integrated-analysis.schema.json"


def test_integrated_analysis_is_deterministic_and_figure_ready(tmp_path: Path) -> None:
    run = _write_completed_inputs(tmp_path)

    first = integrate_completed_analysis(run)
    second = integrate_completed_analysis(run)

    assert first == second
    assert first.report.item_count == 6
    assert first.report.case_count == 2
    assert len(first.report.semantic_paired_comparisons) == 15
    assert first.report.configurations[0].attack_success.value == 1
    assert first.report.configurations[1].attack_success.value == 0
    assert first.report.configurations[2].factual_correctness.value == 1
    assert "attack_success_rate" in first.security_utility_csv
    assert "attack_category" in first.threat_heatmap_csv
    assert "holm_adjusted_p_value" in first.paired_comparisons_csv

    output = write_integrated_analysis(run, first)
    assert (output / "COMPLETE").is_file()
    assert (output / "figure-security-utility.csv").is_file()
    with pytest.raises(IntegratedAnalysisError, match="refusing to replace"):
        write_integrated_analysis(run, first)


def test_integrated_analysis_rejects_semantic_checksum_drift(tmp_path: Path) -> None:
    run = _write_completed_inputs(tmp_path)
    labels = run / "adjudication/semantic-analysis/unblinded-labels.jsonl"
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(IntegratedAnalysisError, match="checksum mismatch"):
        integrate_completed_analysis(run)


def test_integrated_analysis_rejects_identity_mismatch(tmp_path: Path) -> None:
    run = _write_completed_inputs(tmp_path)
    directory = run / "adjudication/semantic-analysis"
    labels_path = directory / "unblinded-labels.jsonl"
    rows = labels_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["case_id"] = "different-case"
    rows[0] = json.dumps(first, separators=(",", ":"))
    labels_text = "\n".join(rows) + "\n"
    labels_path.write_text(labels_text, encoding="utf-8", newline="\n")
    report_path = directory / "semantic-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["unblinded_labels_sha256"] = _sha256(labels_text)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    (directory / "COMPLETE").write_text(_sha256(report_text) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(IntegratedAnalysisError, match="trial identity differs"):
        integrate_completed_analysis(run)


def test_integrated_cli_and_schema_are_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _write_completed_inputs(tmp_path)

    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_integrated_report_schema()
    assert evaluation_cli.main(["integrated-schema", "--check", "--output", str(SCHEMA_PATH)]) == 0
    assert evaluation_cli.main(["integrate", str(run)]) == 0
    assert "Integrated 6 trials with 15 paired semantic comparisons" in capsys.readouterr().out
    assert evaluation_cli.main(["integrate", str(run)]) == 1


def _write_completed_inputs(root: Path) -> Path:
    run = root / "run"
    automated_directory = run / "analysis"
    semantic_directory = run / "adjudication/semantic-analysis"
    automated_directory.mkdir(parents=True)
    semantic_directory.mkdir(parents=True)
    trials = _trial_scores()
    labels = _semantic_labels()
    trial_text = "".join(item.model_dump_json() + "\n" for item in trials)
    labels_text = "".join(item.model_dump_json() + "\n" for item in labels)
    automated_files = {
        "trial-scores.jsonl": trial_text,
        "configuration-summary.csv": "configuration,fixture\nC0,1\n",
        "attack-category-summary.csv": "configuration,attack_category\nC0,fixture\n",
        "paired-comparisons.csv": "metric,difference\nfixture,0\n",
    }
    semantic_files = {
        "unblinded-labels.jsonl": labels_text,
        "configuration-summary.csv": "configuration,dimension\nC0,fixture\n",
        "attack-category-summary.csv": "configuration,attack_category\nC0,fixture\n",
    }
    for name, content in automated_files.items():
        (automated_directory / name).write_text(content, encoding="utf-8", newline="\n")
    for name, content in semantic_files.items():
        (semantic_directory / name).write_text(content, encoding="utf-8", newline="\n")
    score_report = _score_report(automated_files)
    semantic_report = _semantic_report(semantic_files)
    _write_report(automated_directory, "score-report.json", score_report)
    _write_report(semantic_directory, "semantic-report.json", semantic_report)
    return run


def _trial_scores() -> list[TrialScore]:
    records: list[TrialScore] = []
    for case_id, kind in (
        ("benign-case", BenchmarkKind.BENIGN),
        ("attack-case", BenchmarkKind.ADVERSARIAL),
    ):
        for configuration in ExperimentConfiguration:
            baseline = configuration is ExperimentConfiguration.BASELINE
            benign = kind is BenchmarkKind.BENIGN
            records.append(
                TrialScore(
                    trial_id=f"trial-{case_id}-{configuration.value.lower()}",
                    case_id=case_id,
                    template_family=f"{case_id}-family",
                    configuration=configuration,
                    kind=kind,
                    attack_category=(None if benign else AttackCategory.DIRECT_PROMPT_INJECTION),
                    expected_action=ExpectedAction.ANSWER if benign else ExpectedAction.REFUSE,
                    observed_outcome=(
                        ObservedOutcome.ANSWERED if benign or baseline else ObservedOutcome.REFUSED
                    ),
                    coarse_action_alignment=not baseline,
                    exact_forbidden_leakage=None if benign else baseline,
                    leaked_forbidden_value_count=int(not benign and baseline),
                    retrieval_recall_at_5=1.0 if benign else None,
                    retrieval_reciprocal_rank_at_5=1.0 if benign else None,
                    valid_citation_fraction=1.0 if benign else None,
                    has_valid_citation=benign,
                    guardrail_blocked=not benign and not baseline,
                    duration_ms=10,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    estimated_cost_usd=0.001,
                )
            )
    return records


def _semantic_labels() -> list[UnblindedAdjudicationItem]:
    records: list[UnblindedAdjudicationItem] = []
    for case_id, kind in (
        ("benign-case", BenchmarkKind.BENIGN),
        ("attack-case", BenchmarkKind.ADVERSARIAL),
    ):
        for configuration in ExperimentConfiguration:
            baseline = configuration is ExperimentConfiguration.BASELINE
            benign = kind is BenchmarkKind.BENIGN
            records.append(
                UnblindedAdjudicationItem(
                    blinded_item_id=f"review-{len(records) + 1:024x}",
                    trial_id=f"trial-{case_id}-{configuration.value.lower()}",
                    case_id=case_id,
                    template_family=f"{case_id}-family",
                    configuration=configuration,
                    kind=kind,
                    attack_category=(None if benign else AttackCategory.DIRECT_PROMPT_INJECTION),
                    attack_success=(
                        ReviewDecision.NOT_APPLICABLE
                        if benign
                        else ReviewDecision.YES
                        if baseline
                        else ReviewDecision.NO
                    ),
                    factual_correctness=(
                        ReviewDecision.NO
                        if benign and baseline
                        else ReviewDecision.YES
                        if benign
                        else ReviewDecision.NOT_APPLICABLE
                    ),
                    citation_support=(
                        ReviewDecision.YES if benign else ReviewDecision.NOT_APPLICABLE
                    ),
                    helpfulness=(ReviewDecision.YES if benign else ReviewDecision.NOT_APPLICABLE),
                    refusal_appropriateness=(
                        ReviewDecision.NOT_APPLICABLE
                        if benign
                        else ReviewDecision.NO
                        if baseline
                        else ReviewDecision.YES
                    ),
                    consensus_dimensions=[],
                )
            )
    return records


def _score_report(files: dict[str, str]) -> ExperimentScoreReport:
    summaries = [_score_configuration(item) for item in ExperimentConfiguration]
    categories = [
        AttackCategoryScoreSummary(
            configuration=configuration,
            attack_category=category,
            case_count=1,
            coarse_action_alignment=_metric(1.0),
            refusal_rate=_metric(1.0),
            guardrail_block_rate=_metric(1.0),
            exact_sensitive_data_leakage_rate=(
                _metric(0.0)
                if category
                in {
                    AttackCategory.SYNTHETIC_SECRET_EXFILTRATION,
                    AttackCategory.FICTIONAL_PII_EXFILTRATION,
                }
                else None
            ),
        )
        for configuration in ExperimentConfiguration
        for category in AttackCategory
    ]
    comparisons = [
        PairedBinaryComparison(
            metric=metric,
            direction=(
                "higher-is-better" if metric == "coarse-action-alignment" else "lower-is-better"
            ),
            reference_configuration=reference,
            candidate_configuration=candidate,
            reference_rate=0,
            candidate_rate=0,
            difference=0,
            confidence_lower=0,
            confidence_upper=0,
            sample_count=1,
            cluster_count=1,
            reference_only_positive=0,
            candidate_only_positive=0,
            mcnemar_exact_p_value=1,
            holm_adjusted_p_value=1,
        )
        for reference, candidate in (
            (ExperimentConfiguration.BASELINE, ExperimentConfiguration.HARDENED),
            (ExperimentConfiguration.HARDENED, ExperimentConfiguration.DEEPSHIELD),
            (ExperimentConfiguration.BASELINE, ExperimentConfiguration.DEEPSHIELD),
        )
        for metric in (
            "coarse-action-alignment",
            "benign-refusal",
            "exact-sensitive-data-leakage",
        )
    ]
    return ExperimentScoreReport(
        run_id="integrated-test-run",
        purpose=ExperimentPurpose.DEVELOPMENT,
        benchmark_split=BenchmarkSplit.DEVELOPMENT,
        run_manifest_sha256="1" * 64,
        results_sha256="2" * 64,
        benchmark_sha256="3" * 64,
        generation_configuration_sha256="4" * 64,
        scored_at=datetime(2026, 1, 1, tzinfo=UTC),
        configuration_summaries=summaries,
        attack_category_summaries=categories,
        paired_comparisons=comparisons,
        trial_scores_sha256=_sha256(files["trial-scores.jsonl"]),
        configuration_summary_sha256=_sha256(files["configuration-summary.csv"]),
        attack_category_summary_sha256=_sha256(files["attack-category-summary.csv"]),
        paired_comparison_sha256=_sha256(files["paired-comparisons.csv"]),
        limitations=["fixture limitation"],
    )


def _score_configuration(
    configuration: ExperimentConfiguration,
) -> ConfigurationScoreSummary:
    baseline = configuration is ExperimentConfiguration.BASELINE
    return ConfigurationScoreSummary(
        configuration=configuration,
        case_count=2,
        benign_case_count=1,
        adversarial_case_count=1,
        coarse_action_alignment=_metric(0.0 if baseline else 1.0),
        benign_answer_rate=_metric(1.0),
        benign_refusal_rate=_metric(0.0),
        exact_sensitive_data_leakage_rate=_metric(1.0 if baseline else 0.0),
        retrieval_recall_at_5=_metric(1.0),
        retrieval_mrr_at_5=_metric(1.0),
        valid_citation_presence_rate=_metric(1.0),
        valid_citation_fraction=_metric(1.0),
        guardrail_block_rate=_metric(0.0 if baseline else 0.5),
        latency_ms=DistributionSummary(
            sample_count=2,
            mean=10,
            median=10,
            p95=10,
            minimum=10,
            maximum=10,
        ),
        resources=ResourceSummary(
            complete_usage_trial_count=2,
            missing_usage_trial_count=0,
            prompt_tokens=20,
            completion_tokens=4,
            total_tokens=24,
            known_estimated_cost_usd=0.002,
            complete_cost=True,
        ),
    )


def _semantic_report(files: dict[str, str]) -> SemanticAnalysisReport:
    summaries = [
        SemanticConfigurationSummary(
            configuration=configuration,
            item_count=2,
            dimensions=[
                _dimension(
                    "attack_success", 1, int(configuration is ExperimentConfiguration.BASELINE)
                ),
                _dimension(
                    "factual_correctness",
                    1,
                    int(configuration is not ExperimentConfiguration.BASELINE),
                ),
                _dimension("citation_support", 1, 1),
                _dimension("helpfulness", 1, 1),
                _dimension(
                    "refusal_appropriateness",
                    1,
                    int(configuration is not ExperimentConfiguration.BASELINE),
                ),
            ],
        )
        for configuration in ExperimentConfiguration
    ]
    categories = [
        SemanticAttackCategorySummary(
            configuration=configuration,
            attack_category=AttackCategory.DIRECT_PROMPT_INJECTION,
            item_count=1,
            attack_success_count=int(configuration is ExperimentConfiguration.BASELINE),
            attack_success_rate=float(configuration is ExperimentConfiguration.BASELINE),
        )
        for configuration in ExperimentConfiguration
    ]
    return SemanticAnalysisReport(
        run_id="integrated-test-run",
        purpose=ExperimentPurpose.DEVELOPMENT,
        benchmark_split=BenchmarkSplit.DEVELOPMENT,
        adjudication_manifest_sha256="5" * 64,
        blinding_key_sha256="6" * 64,
        consensus_report_sha256="7" * 64,
        resolved_labels_sha256="8" * 64,
        item_count=6,
        case_count=2,
        unblinded_labels_sha256=_sha256(files["unblinded-labels.jsonl"]),
        configuration_summary_sha256=_sha256(files["configuration-summary.csv"]),
        attack_category_summary_sha256=_sha256(files["attack-category-summary.csv"]),
        configurations=summaries,
        attack_categories=categories,
        limitations=["one", "two", "three"],
    )


def _metric(value: float) -> MetricEstimate:
    return MetricEstimate(
        value=value,
        confidence_lower=value,
        confidence_upper=value,
        sample_count=1,
        cluster_count=1,
    )


def _dimension(
    name: DimensionName, applicable_count: int, yes_count: int
) -> SemanticDimensionSummary:
    return SemanticDimensionSummary.model_validate(
        {
            "dimension": name,
            "applicable_item_count": applicable_count,
            "yes_count": yes_count,
            "no_count": applicable_count - yes_count,
            "yes_rate": yes_count / applicable_count,
        }
    )


def _write_report(directory: Path, name: str, report: BaseModel) -> None:
    report_text = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (directory / name).write_text(report_text, encoding="utf-8", newline="\n")
    (directory / "COMPLETE").write_text(_sha256(report_text) + "\n", encoding="utf-8", newline="\n")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
