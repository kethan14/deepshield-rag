from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.generation import build_draft_benchmark
from deepshield_api.evaluation.models import (
    BenchmarkCase,
    BenchmarkManifest,
    BenchmarkStatus,
    ReviewStatus,
)
from deepshield_api.evaluation.runs import (
    ExperimentConfiguration,
    ExperimentPurpose,
    ExperimentRunError,
    ObservedOutcome,
    RetrievalEvidence,
    TrialObservation,
    execute_frozen_experiment,
    load_completed_run,
    prepare_experiment_inputs,
    write_completed_run,
)
from deepshield_api.evaluation.synthetic import build_synthetic_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = Path("data/evaluation/f1-security-benchmark.v1.json")
GENERATION_PATH = Path("config/experiments/generation-model.v1.json")
REVIEWED_AT = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@dataclass
class FakeBackend:
    configuration: ExperimentConfiguration
    retrieval_top_k: Literal[5] = 5
    calls: list[str] = field(default_factory=list)

    def run(self, case: BenchmarkCase) -> TrialObservation:
        self.calls.append(case.case_id)
        return TrialObservation(
            outcome=ObservedOutcome.ANSWERED,
            answer=f"Synthetic {self.configuration.value} answer for {case.case_id} [S1]",
            cited_reference_ids=["S1"],
            retrieved=[
                RetrievalEvidence(
                    reference_id="S1",
                    rank=1,
                    score=0.75,
                    chunk_id="uscis-pm-f-chapter-2-v1-c00000",
                    source_id="uscis-pm-f-chapter-2",
                )
            ],
            generator_model="gpt-5.4-mini-2026-03-17",
            retrieval_model="test-retriever",
        )


class StepTimer:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.001
        return current


class StepClock:
    def __init__(self) -> None:
        self.value = REVIEWED_AT

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(minutes=1)
        return current


def _backends() -> list[FakeBackend]:
    return [FakeBackend(configuration) for configuration in ExperimentConfiguration]


def _frozen_benchmark(tmp_path: Path) -> Path:
    draft = build_draft_benchmark(REPOSITORY_ROOT, build_synthetic_environment())
    approved = [
        case.model_copy(
            update={
                "review_status": ReviewStatus.APPROVED,
                "reviewer_id": "test-reviewer",
                "reviewed_at": REVIEWED_AT,
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


def test_runner_refuses_draft_before_invoking_any_backend() -> None:
    backends = _backends()

    with pytest.raises(ExperimentRunError, match="benchmark must be frozen"):
        execute_frozen_experiment(
            repository_root=REPOSITORY_ROOT,
            benchmark_path=BENCHMARK_PATH,
            generation_configuration_path=GENERATION_PATH,
            purpose=ExperimentPurpose.DEVELOPMENT,
            run_id="draft-must-not-run",
            backends=backends,
        )

    assert all(not backend.calls for backend in backends)


def test_development_run_is_complete_ordered_and_checksum_verifiable(tmp_path: Path) -> None:
    benchmark_path = _frozen_benchmark(tmp_path)
    backends = _backends()
    completed = execute_frozen_experiment(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=benchmark_path,
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.DEVELOPMENT,
        run_id="development-test-run",
        backends=backends,
        clock=StepClock(),
        timer=StepTimer(),
    )

    assert completed.manifest.case_count == 200
    assert completed.manifest.trial_count == 600
    assert completed.manifest.benchmark_split.value == "development"
    assert [trial.configuration for trial in completed.trials[:3]] == list(ExperimentConfiguration)
    assert [trial.sequence for trial in completed.trials] == list(range(1, 601))
    assert all(len(backend.calls) == 200 for backend in backends)
    assert backends[0].calls == backends[1].calls == backends[2].calls
    assert {trial.duration_ms for trial in completed.trials} == {1}

    run_directory = write_completed_run(tmp_path / "runs", completed)
    loaded = load_completed_run(run_directory)

    assert loaded.manifest == completed.manifest
    assert loaded.trials == completed.trials
    assert loaded.results_jsonl == completed.results_jsonl
    with pytest.raises(ExperimentRunError, match="refusing to replace"):
        write_completed_run(tmp_path / "runs", completed)


def test_runner_rejects_missing_or_reordered_configuration(tmp_path: Path) -> None:
    benchmark_path = _frozen_benchmark(tmp_path)
    with pytest.raises(ExperimentRunError, match="C0, C1, and C2"):
        execute_frozen_experiment(
            repository_root=REPOSITORY_ROOT,
            benchmark_path=benchmark_path,
            generation_configuration_path=GENERATION_PATH,
            purpose=ExperimentPurpose.FINAL,
            run_id="invalid-backends",
            backends=list(reversed(_backends())),
        )


def test_final_run_uses_only_the_held_out_test_split(tmp_path: Path) -> None:
    completed = execute_frozen_experiment(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=_frozen_benchmark(tmp_path),
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.FINAL,
        run_id="final-test-run",
        backends=_backends(),
        clock=StepClock(),
        timer=StepTimer(),
    )

    assert completed.manifest.case_count == 400
    assert completed.manifest.trial_count == 1_200
    assert completed.manifest.benchmark_split.value == "test"
    assert {trial.split.value for trial in completed.trials} == {"test"}


def test_loader_detects_result_tampering(tmp_path: Path) -> None:
    benchmark_path = _frozen_benchmark(tmp_path)
    completed = execute_frozen_experiment(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=benchmark_path,
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.DEVELOPMENT,
        run_id="tamper-test-run",
        backends=_backends(),
        clock=StepClock(),
        timer=StepTimer(),
    )
    run_directory = write_completed_run(tmp_path / "runs", completed)
    results = run_directory / "trials.jsonl"
    results.write_text(results.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ExperimentRunError, match="unable to load|checksum"):
        load_completed_run(run_directory)


def test_preflight_exposes_validated_split_without_initializing_backends(tmp_path: Path) -> None:
    prepared = prepare_experiment_inputs(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=_frozen_benchmark(tmp_path),
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.DEVELOPMENT,
    )

    assert prepared.repository_root == REPOSITORY_ROOT
    assert prepared.benchmark.status is BenchmarkStatus.FROZEN
    assert prepared.generation.model == "gpt-5.4-mini-2026-03-17"
    assert prepared.selected_case_count == 200


def test_run_cli_rejects_draft_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_runtime(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime must not be constructed for a draft benchmark")

    monkeypatch.setattr(
        evaluation_cli,
        "build_experiment_comparison_backends",
        unexpected_runtime,
    )

    result = evaluation_cli.main(
        [
            "run",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--purpose",
            "development",
            "--run-id",
            "draft-cli-run",
            "--confirm-paid-run",
        ]
    )

    assert result == 1
    assert "benchmark must be frozen" in capsys.readouterr().err


def test_run_cli_requires_explicit_confirmations_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _frozen_benchmark(tmp_path)

    def unexpected_runtime(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime must not be constructed before confirmation")

    monkeypatch.setattr(
        evaluation_cli,
        "build_experiment_comparison_backends",
        unexpected_runtime,
    )
    common = [
        "run",
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--benchmark",
        str(benchmark),
        "--run-id",
        "confirmation-test-run",
    ]

    assert evaluation_cli.main([*common, "--purpose", "development"]) == 1
    assert "--confirm-paid-run" in capsys.readouterr().err
    assert evaluation_cli.main([*common, "--purpose", "final", "--confirm-paid-run"]) == 1
    assert "--confirm-held-out-test" in capsys.readouterr().err


def test_run_cli_writes_complete_local_evidence_with_fake_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _frozen_benchmark(tmp_path)
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("DEEPSHIELD_LLM_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        evaluation_cli,
        "build_experiment_comparison_backends",
        lambda _settings: tuple(_backends()),
    )

    result = evaluation_cli.main(
        [
            "run",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--benchmark",
            str(benchmark),
            "--runs-root",
            str(runs_root),
            "--purpose",
            "development",
            "--run-id",
            "successful-cli-run",
            "--confirm-paid-run",
        ]
    )

    assert result == 0
    completed = load_completed_run(runs_root / "successful-cli-run")
    assert completed.manifest.case_count == 200
    assert completed.manifest.trial_count == 600
    assert "Completed development experiment" in capsys.readouterr().out
