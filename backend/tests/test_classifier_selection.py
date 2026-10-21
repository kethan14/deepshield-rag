import csv
import hashlib
import io
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from deepshield_api.classifier.cli import main
from deepshield_api.classifier.manifest import render_risk_dataset
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ClassifierPreprocessingReport,
    ClassifierSelectionRules,
    NormalizationRules,
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
    SourcePreprocessingSummary,
    SplitPreprocessingSummary,
)
from deepshield_api.classifier.preprocessing import render_preprocessing_report
from deepshield_api.classifier.selection import (
    ClassifierSelectionError,
    load_selection_report,
    render_selection_report,
    select_classifier_review_candidates,
    validate_selection_report,
    write_selection_result,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "data/classifier/source-registry.v2.json"
BENCHMARK_FIXTURE = REPOSITORY_ROOT / "data/fixtures/evaluation-benchmark.valid.json"
REPORT_PATH = REPOSITORY_ROOT / "data/classifier/selection-report.v1.json"
REPORT_SCHEMA_PATH = REPOSITORY_ROOT / "data/classifier/selection-report.schema.json"
FIXED_TIME = datetime(2026, 10, 21, 3, 30, tzinfo=UTC)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _prepare_repository(root: Path) -> ClassifierSelectionRules:
    classifier_dir = root / "data/classifier"
    evaluation_dir = root / "data/evaluation"
    processed_dir = classifier_dir / "processed"
    processed_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    shutil.copyfile(REGISTRY_PATH, classifier_dir / "source-registry.v2.json")
    shutil.copyfile(BENCHMARK_FIXTURE, evaluation_dir / "f1-security-benchmark.v1.json")
    registry_bytes = (classifier_dir / "source-registry.v2.json").read_bytes()
    registry = _load_json(REGISTRY_PATH)
    sources = registry["sources"]
    assert isinstance(sources, list)
    development_source = str(sources[0]["source_id"])
    external_source = str(sources[1]["source_id"])

    examples: list[RiskExample] = []
    for split in RiskSplit:
        source_id = external_source if split is RiskSplit.EXTERNAL_TEST else development_source
        for label in RiskLabel:
            category = (
                RiskCategory.BENIGN
                if label is RiskLabel.BENIGN
                else RiskCategory.DIRECT_PROMPT_INJECTION
            )
            for index in range(6):
                text = f"Unique {split.value} {label.value} classifier example number {index}."
                examples.append(
                    RiskExample(
                        example_id=f"{split.value}-{label.value}-{index}",
                        source_id=source_id,
                        source_record_id=f"record-{split.value}-{label.value}-{index}",
                        group_id=f"group-{split.value}-{label.value}-{index}",
                        split=split,
                        label=label,
                        category=category,
                        text=text,
                        text_sha256=_hash_text(text),
                    )
                )

    benchmark = _load_json(BENCHMARK_FIXTURE)
    cases = benchmark["cases"]
    assert isinstance(cases, list) and isinstance(cases[0], dict)
    prompt = str(cases[0]["prompt"])
    for suffix, text in (("exact", prompt), ("near", f"{prompt} please")):
        examples.append(
            RiskExample(
                example_id=f"benchmark-overlap-{suffix}",
                source_id=development_source,
                source_record_id=f"benchmark-overlap-{suffix}",
                group_id=f"benchmark-overlap-{suffix}",
                split=RiskSplit.TRAIN,
                label=RiskLabel.BENIGN,
                category=RiskCategory.BENIGN,
                text=text,
                text_sha256=_hash_text(text),
            )
        )
    dataset = RiskDatasetManifest(
        dataset_version=1,
        status=ArtifactStatus.DRAFT,
        created_at=FIXED_TIME,
        source_registry_path="data/classifier/source-registry.v2.json",
        source_registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        examples=examples,
    )
    dataset_content = render_risk_dataset(dataset)
    (processed_dir / "risk-dataset.v1.json").write_text(
        dataset_content, encoding="utf-8", newline="\n"
    )
    split_summaries: list[SplitPreprocessingSummary] = []
    for split in RiskSplit:
        selected = [example for example in examples if example.split is split]
        split_summaries.append(
            SplitPreprocessingSummary(
                split=split,
                row_count=len(selected),
                group_count=len(selected),
                label_counts={
                    label: sum(example.label is label for example in selected)
                    for label in RiskLabel
                },
            )
        )
    report = ClassifierPreprocessingReport(
        report_version=1,
        created_at=FIXED_TIME,
        source_collection_path="data/classifier/source-collection.v1.json",
        source_collection_sha256="0" * 64,
        output_dataset_path="data/classifier/processed/risk-dataset.v1.json",
        output_dataset_sha256=hashlib.sha256(dataset_content.encode()).hexdigest(),
        rules=NormalizationRules(),
        sources=[
            SourcePreprocessingSummary(
                source_id=development_source,
                input_rows=38,
                output_rows=38,
                empty_rows_excluded=0,
                too_short_rows_excluded=0,
                overlength_rows_excluded=0,
                duplicate_rows_excluded=0,
                cross_source_exact_duplicates_excluded=0,
                cross_source_near_duplicates_excluded=0,
                output_label_counts={RiskLabel.BENIGN: 20, RiskLabel.ADVERSARIAL: 18},
            ),
            SourcePreprocessingSummary(
                source_id=external_source,
                input_rows=12,
                output_rows=12,
                empty_rows_excluded=0,
                too_short_rows_excluded=0,
                overlength_rows_excluded=0,
                duplicate_rows_excluded=0,
                cross_source_exact_duplicates_excluded=0,
                cross_source_near_duplicates_excluded=0,
                output_label_counts={RiskLabel.BENIGN: 6, RiskLabel.ADVERSARIAL: 6},
            ),
        ],
        splits=split_summaries,
        development_near_duplicate_cluster_count=38,
        development_multi_member_cluster_count=0,
        development_largest_cluster_size=1,
    )
    (classifier_dir / "preprocessing-report.v1.json").write_text(
        render_preprocessing_report(report), encoding="utf-8"
    )
    return ClassifierSelectionRules(target_split_counts={split: 4 for split in RiskSplit})


def test_committed_selection_evidence_and_schema_are_current() -> None:
    report = load_selection_report(REPORT_PATH)

    validate_selection_report(report, REPOSITORY_ROOT)
    assert report.input_candidate_count == 17_194
    assert report.benchmark_case_count == 600
    assert sum(summary.selected_count for summary in report.splits) == 1_000
    assert all(set(summary.label_counts) == set(RiskLabel) for summary in report.splits)
    assert main(["schema", "selection", "--output", str(REPORT_SCHEMA_PATH), "--check"]) == 0


def test_selection_is_balanced_deterministic_and_excludes_benchmark_overlap(
    tmp_path: Path,
) -> None:
    rules = _prepare_repository(tmp_path)

    first = select_classifier_review_candidates(tmp_path, rules=rules, clock=lambda: FIXED_TIME)
    second = select_classifier_review_candidates(tmp_path, rules=rules, clock=lambda: FIXED_TIME)

    assert first == second
    assert len(first.dataset.examples) == 16
    assert first.report.exact_benchmark_matches_excluded == 1
    assert first.report.near_benchmark_matches_excluded == 1
    assert all(summary.selected_count == 4 for summary in first.report.splits)
    assert all(
        summary.label_counts == {RiskLabel.BENIGN: 2, RiskLabel.ADVERSARIAL: 2}
        for summary in first.report.splits
    )
    assert not any(
        example.example_id.startswith("benchmark-overlap") for example in first.dataset.examples
    )
    rows = list(csv.DictReader(io.StringIO(first.review_packet)))
    assert len(rows) == 16
    assert all(not row["review_decision"] for row in rows)


def test_selection_outputs_are_immutable_and_fully_validated(tmp_path: Path) -> None:
    rules = _prepare_repository(tmp_path)
    result = select_classifier_review_candidates(tmp_path, rules=rules, clock=lambda: FIXED_TIME)
    report_path = tmp_path / "data/classifier/selection-report.v1.json"

    with pytest.raises(ClassifierSelectionError, match="selection evidence requires"):
        write_selection_result(result, tmp_path, tmp_path / "selection-report.json")
    write_selection_result(result, tmp_path, report_path)
    write_selection_result(result, tmp_path, report_path)
    report = load_selection_report(report_path)
    validate_selection_report(report, tmp_path, require_artifacts=True)

    review_path = tmp_path / report.review_packet_path
    review_path.write_text(review_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ClassifierSelectionError, match="review packet checksum"):
        validate_selection_report(report, tmp_path, require_artifacts=True)


def test_selection_report_rejects_incomplete_accounting_and_cli_validates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _prepare_repository(tmp_path)
    result = select_classifier_review_candidates(tmp_path, rules=rules, clock=lambda: FIXED_TIME)
    payload = result.report.model_dump(mode="json")
    payload["eligible_candidate_count"] += 1
    invalid = tmp_path / "invalid-report.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ClassifierSelectionError, match="account for every input candidate"):
        load_selection_report(invalid)

    report_path = tmp_path / "data/classifier/selection-report.v1.json"
    monkeypatch.setattr(
        "deepshield_api.classifier.cli.select_classifier_review_candidates",
        lambda root: select_classifier_review_candidates(
            root, rules=rules, clock=lambda: FIXED_TIME
        ),
    )
    assert main(["select-review-candidates", "--repository-root", str(tmp_path)]) == 0
    assert (
        main(
            [
                "validate-selection",
                str(report_path),
                "--repository-root",
                str(tmp_path),
                "--require-artifacts",
            ]
        )
        == 0
    )
    assert "Validated classifier selection report" in capsys.readouterr().out


def test_selection_output_rejects_benchmark_overlap_during_validation(
    tmp_path: Path,
) -> None:
    rules = _prepare_repository(tmp_path)
    result = select_classifier_review_candidates(tmp_path, rules=rules, clock=lambda: FIXED_TIME)
    report_path = tmp_path / "data/classifier/selection-report.v1.json"
    write_selection_result(result, tmp_path, report_path)
    report = load_selection_report(report_path)
    selected_path = tmp_path / report.selected_dataset_path
    payload = _load_json(selected_path)
    benchmark = _load_json(tmp_path / report.benchmark_path)
    cases = benchmark["cases"]
    examples = payload["examples"]
    assert isinstance(cases, list) and isinstance(cases[0], dict)
    assert isinstance(examples, list) and isinstance(examples[0], dict)
    examples[0]["text"] = str(cases[0]["prompt"])
    examples[0]["text_sha256"] = _hash_text(str(cases[0]["prompt"]))
    selected_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = report.model_copy(
        update={"selected_dataset_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest()}
    )
    report_path.write_text(render_selection_report(report), encoding="utf-8")

    with pytest.raises(ClassifierSelectionError, match="overlaps the held-out benchmark"):
        validate_selection_report(report, tmp_path, require_artifacts=True)
