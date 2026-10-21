import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path

import pytest

from deepshield_api.classifier.cli import main
from deepshield_api.classifier.freezing import (
    freeze_classifier_review,
    load_and_validate_review_report,
    write_review_freeze_result,
)
from deepshield_api.classifier.manifest import render_review_report, render_risk_dataset
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ReviewStatus,
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
)
from deepshield_api.classifier.review import (
    REVIEW_FIELDS,
    ClassifierReviewError,
    render_classifier_review_worksheet,
)
from deepshield_api.classifier.selection import load_selection_report

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SELECTION_REPORT = REPOSITORY_ROOT / "data/classifier/selection-report.v1.json"


def _selected_dataset() -> RiskDatasetManifest:
    split_counts = {
        RiskSplit.TRAIN: 500,
        RiskSplit.VALIDATION: 100,
        RiskSplit.TEST: 200,
        RiskSplit.EXTERNAL_TEST: 200,
    }
    examples: list[RiskExample] = []
    index = 0
    for split, count in split_counts.items():
        source_id = "spml-prompt-injection" if split is RiskSplit.EXTERNAL_TEST else "development"
        for split_index in range(count):
            label = RiskLabel.BENIGN if split_index % 2 == 0 else RiskLabel.ADVERSARIAL
            category = (
                RiskCategory.BENIGN
                if label is RiskLabel.BENIGN
                else RiskCategory.DIRECT_PROMPT_INJECTION
            )
            text = f"Unique classifier review example {index} for {split.value}."
            examples.append(
                RiskExample(
                    example_id=f"review-example-{index:04d}",
                    source_id=source_id,
                    source_record_id=str(index),
                    group_id=f"review-group-{index:04d}",
                    split=split,
                    label=label,
                    category=category,
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
            index += 1
    return RiskDatasetManifest(
        dataset_version=2,
        status=ArtifactStatus.DRAFT,
        created_at=datetime.fromisoformat("2026-10-20T12:00:00-04:00"),
        source_registry_path="data/classifier/source-registry.v2.json",
        source_registry_sha256="a" * 64,
        benchmark_case_ids_sha256="b" * 64,
        examples=examples,
    )


def _completed_worksheet(dataset: RiskDatasetManifest, *, exclude_first: bool = False) -> str:
    rows = list(csv.DictReader(io.StringIO(render_classifier_review_worksheet(dataset))))
    for index, row in enumerate(rows):
        row["review_decision"] = "approve"
        row["reviewer_id"] = "reviewer-one" if index % 2 == 0 else "reviewer-two"
        row["reviewed_at"] = "2026-10-20T13:00:00-04:00"
    rows[0]["review_decision"] = "exclude" if exclude_first else "correct"
    rows[0]["reviewer_notes"] = "Excluded as ambiguous." if exclude_first else "Corrected label."
    if not exclude_first:
        rows[0]["final_label"] = "adversarial"
        rows[0]["final_category"] = "direct-prompt-injection"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exclude_first: bool = False,
) -> Path:
    dataset = _selected_dataset()
    selected_path = tmp_path / "data/classifier/processed/risk-dataset-selected.v1.json"
    selected_path.parent.mkdir(parents=True)
    selected_content = render_risk_dataset(dataset)
    selected_path.write_text(selected_content, encoding="utf-8", newline="\n")
    worksheet = selected_path.parent / "risk-dataset-human-review.v1.csv"
    worksheet.write_text(
        _completed_worksheet(dataset, exclude_first=exclude_first),
        encoding="utf-8",
        newline="\n",
    )
    selection_path = tmp_path / "data/classifier/selection-report.v1.json"
    selection_path.write_text("synthetic selection evidence\n", encoding="utf-8", newline="\n")
    selection = load_selection_report(SELECTION_REPORT).model_copy(
        update={
            "selected_dataset_sha256": hashlib.sha256(selected_content.encode()).hexdigest(),
        }
    )
    monkeypatch.setattr(
        "deepshield_api.classifier.freezing.load_selection_report",
        lambda _path: selection,
    )
    monkeypatch.setattr(
        "deepshield_api.classifier.freezing.validate_selection_report",
        lambda _report, _root, require_artifacts=False: None,
    )
    monkeypatch.setattr(
        "deepshield_api.classifier.freezing.validate_dataset_references",
        lambda _dataset, _root: None,
    )
    return tmp_path


def test_completed_review_freezes_data_and_validates_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, monkeypatch)

    result = freeze_classifier_review(root)

    assert result.dataset.status is ArtifactStatus.FROZEN
    assert result.dataset.dataset_version == 3
    assert len(result.dataset.examples) == 1000
    assert all(
        example.review_status is ReviewStatus.APPROVED for example in result.dataset.examples
    )
    assert result.dataset.examples[0].label is RiskLabel.ADVERSARIAL
    assert result.report.approved_count == 999
    assert result.report.corrected_count == 1
    assert result.report.excluded_count == 0
    assert result.report.reviewer_counts == {"reviewer-one": 500, "reviewer-two": 500}

    report_path = root / "data/classifier/review-report.v1.json"
    write_review_freeze_result(result, root, report_path)
    validated = load_and_validate_review_report(report_path, root, require_artifacts=True)
    assert validated == result.report
    assert (
        main(
            [
                "freeze-review",
                "--repository-root",
                str(root),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate-review",
                str(report_path),
                "--repository-root",
                str(root),
                "--require-artifacts",
            ]
        )
        == 0
    )

    with pytest.raises(ClassifierReviewError, match="review evidence requires"):
        write_review_freeze_result(result, root, root / "outside.json")

    with pytest.raises(ClassifierReviewError, match="already exists with different content"):
        output_path = root / result.report.output_dataset_path
        output_path.write_text("changed\n", encoding="utf-8", newline="\n")
        write_review_freeze_result(result, root, report_path)


def test_exclusion_requires_replacement_before_freezing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, monkeypatch, exclude_first=True)

    with pytest.raises(ClassifierReviewError, match="replacement candidates"):
        freeze_classifier_review(root)


def test_validation_reconstructs_frozen_examples_from_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, monkeypatch)
    result = freeze_classifier_review(root)
    report_path = root / "data/classifier/review-report.v1.json"
    write_review_freeze_result(result, root, report_path)

    first = result.dataset.examples[0].model_copy(update={"reviewer_id": "reviewer-three"})
    tampered = RiskDatasetManifest.model_validate(
        result.dataset.model_dump() | {"examples": [first, *result.dataset.examples[1:]]}
    )
    tampered_content = render_risk_dataset(tampered)
    output_path = root / result.report.output_dataset_path
    output_path.write_text(tampered_content, encoding="utf-8", newline="\n")
    tampered_report = result.report.model_copy(
        update={
            "output_dataset_sha256": hashlib.sha256(tampered_content.encode()).hexdigest(),
        }
    )
    report_path.write_text(
        render_review_report(tampered_report),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ClassifierReviewError, match="does not match completed review"):
        load_and_validate_review_report(report_path, root, require_artifacts=True)
