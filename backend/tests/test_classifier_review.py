import csv
import io
from collections.abc import Callable
from pathlib import Path

import pytest

from deepshield_api.classifier.cli import main
from deepshield_api.classifier.manifest import load_risk_dataset
from deepshield_api.classifier.review import (
    REVIEW_FIELDS,
    ClassifierReviewError,
    audit_classifier_review_worksheet,
    render_classifier_review_worksheet,
    write_classifier_review_worksheet,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_FIXTURE = REPOSITORY_ROOT / "data/fixtures/risk-dataset.valid.json"


def _rows(content: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content)))


def _render_rows(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _approve(row: dict[str, str], reviewer: str = "reviewer-one") -> None:
    row["review_decision"] = "approve"
    row["reviewer_id"] = reviewer
    row["reviewed_at"] = "2026-10-20T12:00:00-04:00"


def test_review_worksheet_starts_pending_and_records_valid_progress(tmp_path: Path) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    content = render_classifier_review_worksheet(dataset)
    path = tmp_path / "review.csv"
    path.write_text(content, encoding="utf-8-sig", newline="\n")

    progress = audit_classifier_review_worksheet(path, dataset)
    assert progress.total == 4
    assert progress.pending == 4
    assert not progress.complete

    rows = _rows(content)
    _approve(rows[0])
    original_label = rows[1]["proposed_label"]
    rows[1]["review_decision"] = "correct"
    rows[1]["reviewer_id"] = "reviewer-two"
    rows[1]["reviewed_at"] = "2026-10-20T12:05:00-04:00"
    rows[1]["reviewer_notes"] = "Corrected after checking the source label."
    if original_label == "benign":
        rows[1]["final_label"] = "adversarial"
        rows[1]["final_category"] = "direct-prompt-injection"
    else:
        rows[1]["final_label"] = "benign"
        rows[1]["final_category"] = "benign"
    rows[2]["review_decision"] = "exclude"
    rows[2]["reviewer_id"] = "reviewer-one"
    rows[2]["reviewed_at"] = "2026-10-20T12:10:00-04:00"
    rows[2]["reviewer_notes"] = "Ambiguous and unsuitable for training."
    _approve(rows[3], reviewer="reviewer-two")
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")

    progress = audit_classifier_review_worksheet(path, dataset, require_complete=True)
    assert progress.complete
    assert (progress.approved, progress.corrected, progress.excluded) == (2, 1, 1)
    assert progress.reviewer_counts == {"reviewer-one": 2, "reviewer-two": 2}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(text="changed"), "immutable selection evidence"),
        (
            lambda row: row.update(reviewer_id="reviewer-one"),
            "pending rows cannot contain reviewer evidence",
        ),
        (
            lambda row: row.update(
                review_decision="exclude",
                reviewer_id="reviewer-one",
                reviewed_at="2026-10-20T12:00:00-04:00",
            ),
            "exclusions require reviewer_notes",
        ),
        (
            lambda row: row.update(
                review_decision="correct",
                reviewer_id="reviewer-one",
                reviewed_at="2026-10-20T12:00:00-04:00",
                reviewer_notes="Changed.",
            ),
            "correct requires a changed label or category",
        ),
    ],
)
def test_review_audit_rejects_invalid_rows(
    tmp_path: Path,
    mutator: Callable[[dict[str, str]], None],
    message: str,
) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    rows = _rows(render_classifier_review_worksheet(dataset))
    mutator(rows[0])
    path = tmp_path / "review.csv"
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")

    with pytest.raises(ClassifierReviewError, match=message):
        audit_classifier_review_worksheet(path, dataset)


def test_review_audit_requires_completion_and_consistent_final_labels(tmp_path: Path) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    rows = _rows(render_classifier_review_worksheet(dataset))
    path = tmp_path / "review.csv"
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")
    with pytest.raises(ClassifierReviewError, match="pending decision"):
        audit_classifier_review_worksheet(path, dataset, require_complete=True)

    _approve(rows[0])
    rows[0]["review_decision"] = "correct"
    rows[0]["reviewer_notes"] = "Changed category only."
    rows[0]["final_category"] = "direct-prompt-injection"
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")
    with pytest.raises(ClassifierReviewError, match="benign label requires benign category"):
        audit_classifier_review_worksheet(path, dataset)


def test_review_audit_requires_timezone_and_rejects_pending_corrections(tmp_path: Path) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    rows = _rows(render_classifier_review_worksheet(dataset))
    rows[0]["final_label"] = "adversarial"
    rows[0]["final_category"] = "direct-prompt-injection"
    path = tmp_path / "review.csv"
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")
    with pytest.raises(ClassifierReviewError, match="pending rows cannot contain label"):
        audit_classifier_review_worksheet(path, dataset)

    rows = _rows(render_classifier_review_worksheet(dataset))
    _approve(rows[0])
    rows[0]["reviewed_at"] = "2026-10-20T12:00:00"
    path.write_text(_render_rows(rows), encoding="utf-8", newline="\n")
    with pytest.raises(ClassifierReviewError, match="must include a timezone"):
        audit_classifier_review_worksheet(path, dataset)


def test_review_writer_refuses_overwrite_and_cli_reports_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    content = render_classifier_review_worksheet(dataset)
    output = tmp_path / "data/classifier/processed/risk-dataset-human-review.v1.csv"
    write_classifier_review_worksheet(output, content)
    with pytest.raises(ClassifierReviewError, match="refusing to replace human work"):
        write_classifier_review_worksheet(output, content)

    monkeypatch.setattr(
        "deepshield_api.classifier.cli.load_selected_dataset_for_review",
        lambda _root: dataset,
    )
    assert (
        main(
            [
                "audit-review",
                str(output),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "4 total, 4 pending" in capsys.readouterr().out


def test_prepare_review_cli_uses_only_the_fixed_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = render_classifier_review_worksheet(load_risk_dataset(DATASET_FIXTURE))
    monkeypatch.setattr(
        "deepshield_api.classifier.cli.prepare_classifier_review_worksheet",
        lambda _root: content,
    )
    assert main(["prepare-review", "--repository-root", str(tmp_path)]) == 0
    expected = tmp_path / "data/classifier/processed/risk-dataset-human-review.v1.csv"
    assert expected.read_text(encoding="utf-8") == content
    assert (
        main(
            [
                "prepare-review",
                "--repository-root",
                str(tmp_path),
                "--output",
                "outside.csv",
            ]
        )
        == 2
    )


def test_review_worksheet_neutralizes_spreadsheet_formulas() -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    unsafe_example = dataset.examples[0].model_copy(
        update={"text": '=HYPERLINK("https://example.test")'}
    )
    safe_dataset = dataset.model_copy(update={"examples": [unsafe_example, *dataset.examples[1:]]})

    rows = _rows(render_classifier_review_worksheet(safe_dataset))

    assert rows[0]["text"].startswith("'=")
