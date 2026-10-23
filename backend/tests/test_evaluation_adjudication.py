import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.adjudication import (
    PRIVATE_FIXTURE_PLACEHOLDER,
    AdjudicationError,
    AdjudicationPacketItem,
    BlindingKey,
    ReviewDecision,
    prepare_adjudication_package,
    render_adjudication_manifest_schema,
    validate_completed_labels,
    write_adjudication_package,
)
from deepshield_api.evaluation.manifest import load_benchmark_manifest
from deepshield_api.evaluation.models import (
    AttackCategory,
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
from deepshield_api.rag.experiment import load_fixed_generation_configuration
from deepshield_api.rag.models import GenerationUsage

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = Path("data/evaluation/f1-security-benchmark.v1.json")
GENERATION_PATH = Path("config/experiments/generation-model.v1.json")
SCHEMA_PATH = REPOSITORY_ROOT / "data/evaluation/adjudication-manifest.schema.json"


@dataclass
class AdjudicationBackend:
    configuration: ExperimentConfiguration
    generator_model: str
    retrieval_top_k: Literal[5] = 5

    def run(self, case: BenchmarkCase) -> TrialObservation:
        retrieved: list[RetrievalEvidence] = []
        if case.attack_category is AttackCategory.INDIRECT_PROMPT_INJECTION:
            synthetic = case.synthetic_retrieval_chunks[0]
            retrieved.append(
                RetrievalEvidence(
                    reference_id="S1",
                    rank=1,
                    score=1.0,
                    chunk_id=synthetic.synthetic_chunk_id,
                    source_id="synthetic-poisoned-retrieval",
                )
            )
        if case.attack_category in {
            AttackCategory.SYNTHETIC_SECRET_EXFILTRATION,
            AttackCategory.FICTIONAL_PII_EXFILTRATION,
        }:
            retrieved.append(
                RetrievalEvidence(
                    reference_id=f"S{len(retrieved) + 1}",
                    rank=len(retrieved) + 1,
                    score=1.0,
                    chunk_id=f"private-{case.safety_fixture_ids[0]}",
                    source_id="synthetic-private-records",
                )
            )
        for index, chunk_id in enumerate(case.reference_chunk_ids, len(retrieved) + 1):
            if index > 5:
                break
            retrieved.append(
                RetrievalEvidence(
                    reference_id=f"S{index}",
                    rank=index,
                    score=0.9,
                    chunk_id=chunk_id,
                    source_id=case.reference_source_ids[0],
                )
            )
        should_refuse = case.expected_action is ExpectedAction.REFUSE
        cited = [
            item.reference_id
            for item in retrieved
            if item.source_id not in {"synthetic-private-records", "synthetic-poisoned-retrieval"}
        ]
        return TrialObservation(
            outcome=ObservedOutcome.REFUSED if should_refuse else ObservedOutcome.ANSWERED,
            answer=(
                "I cannot comply with that unsafe request."
                if should_refuse
                else "The controlled policy answer follows the required facts and cited evidence."
            ),
            cited_reference_ids=cited,
            retrieved=retrieved,
            generator_model=self.generator_model,
            retrieval_model="adjudication-test-retriever",
            finish_reason=None if should_refuse else "stop",
            usage=(
                GenerationUsage()
                if should_refuse
                else GenerationUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
        )


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("adjudication-run")
    draft = load_benchmark_manifest(REPOSITORY_ROOT / BENCHMARK_PATH)
    reviewed_at = datetime(2026, 10, 22, 0, 0, tzinfo=UTC)
    frozen = BenchmarkManifest.model_validate(
        {
            **draft.model_dump(mode="python"),
            "status": BenchmarkStatus.FROZEN,
            "cases": [
                case.model_copy(
                    update={
                        "review_status": ReviewStatus.APPROVED,
                        "reviewer_id": "test-reviewer",
                        "reviewed_at": reviewed_at,
                    }
                )
                for case in draft.cases
            ],
        }
    )
    frozen_path = root / "frozen-benchmark.json"
    frozen_path.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
    generation = load_fixed_generation_configuration(REPOSITORY_ROOT / GENERATION_PATH)
    completed = execute_frozen_experiment(
        repository_root=REPOSITORY_ROOT,
        benchmark_path=frozen_path,
        generation_configuration_path=GENERATION_PATH,
        purpose=ExperimentPurpose.DEVELOPMENT,
        run_id="adjudication-test-run",
        backends=[
            AdjudicationBackend(configuration, generation.model)
            for configuration in ExperimentConfiguration
        ],
    )
    return write_completed_run(root / "runs", completed), frozen_path


def test_package_is_deterministic_blinded_and_stratified(
    completed_run: tuple[Path, Path],
) -> None:
    run_directory, benchmark_path = completed_run
    first = prepare_adjudication_package(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark_path,
    )
    second = prepare_adjudication_package(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark_path,
    )

    assert first == second
    assert first.manifest.case_count == 200
    assert first.manifest.item_count == 600
    assert first.manifest.secondary_case_count == 40
    assert first.manifest.secondary_item_count == 120
    assert '"configuration"' not in first.primary_packet_jsonl
    assert "generator_model" not in first.primary_packet_jsonl
    assert "guardrail_events" not in first.primary_packet_jsonl
    assert PRIVATE_FIXTURE_PLACEHOLDER in first.primary_packet_jsonl

    key = BlindingKey.model_validate_json(first.blinding_key_json)
    assert {item.configuration for item in key.entries} == set(ExperimentConfiguration)
    assert sum(item.secondary_position is not None for item in key.entries) == 120
    secondary_cases = {item.case_id for item in key.entries if item.secondary_position is not None}
    assert len(secondary_cases) == 40
    for case_id in secondary_cases:
        assert {
            item.configuration
            for item in key.entries
            if item.case_id == case_id and item.secondary_position is not None
        } == set(ExperimentConfiguration)


def test_writer_is_no_overwrite_and_schema_is_current(
    completed_run: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_directory, benchmark_path = completed_run
    package = prepare_adjudication_package(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark_path,
    )
    copied_run = tmp_path / "run"
    copied_run.mkdir()
    output = write_adjudication_package(copied_run, package)

    assert (output / "COMPLETE").is_file()
    assert (output / "blinding-key.json").is_file()
    assert (output / "primary-packet.jsonl").is_file()
    with pytest.raises(AdjudicationError, match="refusing to replace"):
        write_adjudication_package(copied_run, package)
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_adjudication_manifest_schema()
    assert (
        evaluation_cli.main(["adjudication-schema", "--check", "--output", str(SCHEMA_PATH)]) == 0
    )
    assert "Adjudication schema is current" in capsys.readouterr().out


def test_complete_labels_validate_with_packet_applicability(
    completed_run: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    run_directory, benchmark_path = completed_run
    package = prepare_adjudication_package(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark_path,
    )
    packet_path = tmp_path / "secondary-packet.jsonl"
    packet_path.write_text(package.secondary_packet_jsonl, encoding="utf-8")
    items = [
        AdjudicationPacketItem.model_validate_json(line)
        for line in package.secondary_packet_jsonl.splitlines()
    ]
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(_completed_labels(items), encoding="utf-8")

    validated = validate_completed_labels(packet_path=packet_path, labels_path=labels_path)

    assert validated.reviewer_id == "independent-reviewer"
    assert len(validated.labels) == 120
    assert (
        evaluation_cli.main(
            ["adjudication-validate", str(packet_path), str(labels_path)]
        )
        == 0
    )


def test_labels_reject_inapplicable_decision(
    completed_run: tuple[Path, Path], tmp_path: Path
) -> None:
    run_directory, benchmark_path = completed_run
    package = prepare_adjudication_package(
        repository_root=REPOSITORY_ROOT,
        run_directory=run_directory,
        benchmark_path=benchmark_path,
    )
    packet_path = tmp_path / "primary-packet.jsonl"
    packet_path.write_text(package.primary_packet_jsonl, encoding="utf-8")
    items = [
        AdjudicationPacketItem.model_validate_json(line)
        for line in package.primary_packet_jsonl.splitlines()
    ]
    rows = list(csv.DictReader(io.StringIO(_completed_labels(items))))
    target = next(
        row for row, item in zip(rows, items, strict=True) if not item.attack_success_applicable
    )
    target["attack_success"] = ReviewDecision.NO.value
    invalid_path = tmp_path / "invalid.csv"
    invalid_path.write_text(_render_rows(rows), encoding="utf-8")

    with pytest.raises(AdjudicationError, match="attack_success applicability"):
        validate_completed_labels(packet_path=packet_path, labels_path=invalid_path)


def test_prepare_cli_writes_once(completed_run: tuple[Path, Path], tmp_path: Path) -> None:
    run_directory, benchmark_path = completed_run
    copied_run = tmp_path / "run"
    copied_run.mkdir()
    for name in ("run-manifest.json", "trials.jsonl"):
        (copied_run / name).write_bytes((run_directory / name).read_bytes())
    arguments = [
        "adjudication-prepare",
        str(copied_run),
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--benchmark",
        str(benchmark_path),
    ]

    assert evaluation_cli.main(arguments) == 0
    assert evaluation_cli.main(arguments) == 1


def _completed_labels(items: list[AdjudicationPacketItem]) -> str:
    rows = []
    for item in items:
        rows.append(
            {
                "blinded_item_id": item.blinded_item_id,
                "reviewer_id": "independent-reviewer",
                "attack_success": _decision(item.attack_success_applicable, positive=False),
                "factual_correctness": _decision(
                    item.factual_correctness_applicable, positive=True
                ),
                "citation_support": _decision(item.citation_support_applicable, positive=True),
                "helpfulness": _decision(item.helpfulness_applicable, positive=True),
                "refusal_appropriateness": _decision(
                    item.refusal_appropriateness_applicable, positive=True
                ),
                "confidence": "high",
                "notes": "",
            }
        )
    return _render_rows(rows)


def _decision(applicable: bool, *, positive: bool) -> str:
    if not applicable:
        return ReviewDecision.NOT_APPLICABLE.value
    return ReviewDecision.YES.value if positive else ReviewDecision.NO.value


def _render_rows(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
