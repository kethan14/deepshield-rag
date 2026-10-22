from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.generation import build_draft_benchmark
from deepshield_api.evaluation.models import (
    BenchmarkCase,
    BenchmarkManifest,
    BenchmarkStatus,
    ExpectedAction,
    ReviewStatus,
)
from deepshield_api.evaluation.runs import (
    ExperimentConfiguration,
    ExperimentPurpose,
    ObservedOutcome,
    RetrievalEvidence,
    TrialObservation,
    execute_frozen_experiment,
    write_completed_run,
)
from deepshield_api.evaluation.scoring import (
    ExperimentScoringError,
    render_score_report_schema,
    score_completed_experiment,
    write_completed_score,
)
from deepshield_api.evaluation.synthetic import build_synthetic_environment
from deepshield_api.rag.models import (
    GenerationUsage,
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvent,
    GuardrailStage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GENERATION_PATH = Path("config/experiments/generation-model.v1.json")
SCORED_AT = datetime(2026, 10, 21, 18, 0, tzinfo=UTC)
SCORE_SCHEMA_PATH = REPOSITORY_ROOT / "data/evaluation/score-report.schema.json"


@dataclass
class ScoringBackend:
    configuration: ExperimentConfiguration
    retrieval_top_k: Literal[5] = 5

    def run(self, case: BenchmarkCase) -> TrialObservation:
        should_refuse = (
            self.configuration is not ExperimentConfiguration.BASELINE
            and case.expected_action is ExpectedAction.REFUSE
        )
        answer = "I cannot provide that protected information."
        if not should_refuse:
            answer = "Controlled policy answer"
            if case.forbidden_content and self.configuration is ExperimentConfiguration.BASELINE:
                protected = case.forbidden_content[0]
                if protected.startswith("+1-"):
                    protected = " ".join(
                        character for character in protected if character.isdigit()
                    )
                answer = f"Unsafe synthetic disclosure: {protected}"

        retrieved: list[RetrievalEvidence] = []
        citations: list[str] = []
        if case.reference_chunk_ids:
            retrieved.append(
                RetrievalEvidence(
                    reference_id="S1",
                    rank=1,
                    score=0.9,
                    chunk_id=case.reference_chunk_ids[0],
                    source_id=case.reference_source_ids[0],
                )
            )
            citations = ["S1"]
        events: list[GuardrailEvent] = []
        if should_refuse:
            events.append(
                GuardrailEvent(
                    stage=GuardrailStage.INPUT,
                    action=GuardrailAction.BLOCK,
                    category=GuardrailCategory.SENSITIVE_DATA_REQUEST,
                    code="synthetic-test-block",
                    detail="Synthetic test refusal.",
                )
            )
        usage = GenerationUsage()
        finish_reason = None
        if not should_refuse:
            usage = GenerationUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
            finish_reason = "stop"
        return TrialObservation(
            outcome=ObservedOutcome.REFUSED if should_refuse else ObservedOutcome.ANSWERED,
            answer=answer,
            cited_reference_ids=citations,
            retrieved=retrieved,
            guardrail_events=events,
            generator_model="gpt-5.4-mini-2026-03-17",
            retrieval_model="synthetic-retriever",
            finish_reason=finish_reason,
            usage=usage,
        )


def _frozen_benchmark(tmp_path: Path) -> Path:
    draft = build_draft_benchmark(REPOSITORY_ROOT, build_synthetic_environment())
    approved = [
        case.model_copy(
            update={
                "review_status": ReviewStatus.APPROVED,
                "reviewer_id": "test-reviewer",
                "reviewed_at": SCORED_AT,
            }
        )
        for case in draft.cases
    ]
    frozen = BenchmarkManifest.model_validate(
        {
            **draft.model_dump(mode="python"),
            "status": BenchmarkStatus.FROZEN,
            "cases": approved,
        }
    )
    path = tmp_path / "frozen-benchmark.json"
    path.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
    return path


def _completed_run(tmp_path: Path) -> tuple[Path, Path]:
    benchmark = _frozen_benchmark(tmp_path)
    completed = execute_frozen_experiment(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=benchmark,
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.DEVELOPMENT,
        run_id="scoring-test-run",
        backends=[ScoringBackend(item) for item in ExperimentConfiguration],
    )
    run_directory = write_completed_run(tmp_path / "runs", completed)
    return benchmark, run_directory


def test_scoring_produces_deterministic_security_utility_and_resource_metrics(
    tmp_path: Path,
) -> None:
    benchmark, run_directory = _completed_run(tmp_path)

    first = score_completed_experiment(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark,
        generation_configuration_path=GENERATION_PATH,
        scored_at=SCORED_AT,
    )
    second = score_completed_experiment(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark,
        generation_configuration_path=GENERATION_PATH,
        scored_at=SCORED_AT,
    )

    assert first == second
    assert len(first.trial_scores) == 600
    c0, c1, c2 = first.report.configuration_summaries
    assert c0.configuration is ExperimentConfiguration.BASELINE
    assert c0.exact_sensitive_data_leakage_rate.value == 1
    assert c1.exact_sensitive_data_leakage_rate.value == 0
    assert c2.exact_sensitive_data_leakage_rate.value == 0
    assert c0.coarse_action_alignment.value < c1.coarse_action_alignment.value
    assert c1.coarse_action_alignment.value == 1
    assert all(item.retrieval_recall_at_5.value == 1 for item in (c0, c1, c2))
    assert all(item.retrieval_mrr_at_5.value == 1 for item in (c0, c1, c2))
    assert all(item.resources.complete_cost for item in (c0, c1, c2))
    assert c0.resources.known_estimated_cost_usd > c1.resources.known_estimated_cost_usd
    assert len(first.report.attack_category_summaries) == 18
    assert len(first.report.paired_comparisons) == 9
    leakage_c1_vs_c0 = next(
        item
        for item in first.report.paired_comparisons
        if item.metric == "exact-sensitive-data-leakage"
        and item.reference_configuration is ExperimentConfiguration.BASELINE
        and item.candidate_configuration is ExperimentConfiguration.HARDENED
    )
    assert leakage_c1_vs_c0.difference == -1
    assert leakage_c1_vs_c0.mcnemar_exact_p_value < 0.05
    assert leakage_c1_vs_c0.holm_adjusted_p_value >= leakage_c1_vs_c0.mcnemar_exact_p_value
    assert "configuration,case_count" in first.configuration_summary_csv
    assert "configuration,attack_category" in first.attack_category_summary_csv
    assert "holm_adjusted_p_value" in first.paired_comparisons_csv
    assert not any(
        case.forbidden_content[0] in first.trial_scores_jsonl
        for case in build_draft_benchmark(REPOSITORY_ROOT, build_synthetic_environment()).cases
        if case.forbidden_content
    )


def test_committed_score_schema_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    assert SCORE_SCHEMA_PATH.read_text(encoding="utf-8") == render_score_report_schema()
    assert evaluation_cli.main(["score-schema", "--check", "--output", str(SCORE_SCHEMA_PATH)]) == 0
    assert "Score schema is current" in capsys.readouterr().out


def test_score_writer_is_no_overwrite_and_keeps_sanitized_artifacts(tmp_path: Path) -> None:
    benchmark, run_directory = _completed_run(tmp_path)
    completed = score_completed_experiment(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark,
        generation_configuration_path=GENERATION_PATH,
        scored_at=SCORED_AT,
    )

    output = write_completed_score(run_directory, completed)

    assert (output / "score-report.json").is_file()
    assert (output / "trial-scores.jsonl").is_file()
    assert (output / "configuration-summary.csv").is_file()
    assert (output / "attack-category-summary.csv").is_file()
    assert (output / "paired-comparisons.csv").is_file()
    with pytest.raises(ExperimentScoringError, match="refusing to replace"):
        write_completed_score(run_directory, completed)


def test_scoring_rejects_generation_configuration_checksum_drift(tmp_path: Path) -> None:
    benchmark, run_directory = _completed_run(tmp_path)
    changed = tmp_path / "generation.json"
    changed.write_text(
        (REPOSITORY_ROOT / GENERATION_PATH).read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentScoringError, match="configuration checksum"):
        score_completed_experiment(
            repository_root=REPOSITORY_ROOT,
            run_directory=run_directory,
            benchmark_path=benchmark,
            generation_configuration_path=changed,
            scored_at=SCORED_AT,
        )


def test_score_cli_writes_analysis_without_model_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark, run_directory = _completed_run(tmp_path)
    arguments = [
        "score",
        str(run_directory),
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--benchmark",
        str(benchmark),
        "--generation-configuration",
        str(REPOSITORY_ROOT / GENERATION_PATH),
    ]

    assert evaluation_cli.main(arguments) == 0
    assert "Scored 600 trials with 9 paired comparisons" in capsys.readouterr().out
    assert evaluation_cli.main(arguments) == 1
    assert "refusing to replace" in capsys.readouterr().err
