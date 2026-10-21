"""Human-review worksheet preparation and validation for classifier data."""

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from deepshield_api.classifier.manifest import load_risk_dataset
from deepshield_api.classifier.models import (
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
)
from deepshield_api.classifier.selection import (
    ClassifierSelectionError,
    load_selection_report,
    validate_selection_report,
)

DEFAULT_SELECTION_REPORT = Path("data/classifier/selection-report.v1.json")
DEFAULT_REVIEW_WORKSHEET = Path("data/classifier/processed/risk-dataset-human-review.v1.csv")
REVIEW_FIELDS = (
    "example_id",
    "split",
    "source_id",
    "source_record_id",
    "group_id",
    "proposed_label",
    "proposed_category",
    "text",
    "review_decision",
    "final_label",
    "final_category",
    "reviewer_id",
    "reviewed_at",
    "reviewer_notes",
)


class ReviewDecision(StrEnum):
    """Permitted human decisions for one proposed classifier example."""

    APPROVE = "approve"
    CORRECT = "correct"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class ReviewProgress:
    """Validated progress counts for one review worksheet."""

    total: int
    pending: int
    approved: int
    corrected: int
    excluded: int
    reviewer_counts: dict[str, int]

    @property
    def complete(self) -> bool:
        """Return whether every row has a valid human decision."""

        return self.pending == 0


class ClassifierReviewError(RuntimeError):
    """Raised when a review worksheet is incomplete or inconsistent."""


def prepare_classifier_review_worksheet(repository_root: Path) -> str:
    """Validate selection evidence and render a separate editable worksheet."""

    root = repository_root.resolve()
    report = load_selection_report(root / DEFAULT_SELECTION_REPORT)
    try:
        validate_selection_report(report, root, require_artifacts=True)
    except ClassifierSelectionError as exc:
        raise ClassifierReviewError(str(exc)) from exc
    dataset = load_risk_dataset(root / report.selected_dataset_path)
    return render_classifier_review_worksheet(dataset)


def render_classifier_review_worksheet(dataset: RiskDatasetManifest) -> str:
    """Render review rows with immutable evidence and explicit final-label columns."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for example in dataset.examples:
        writer.writerow(
            {
                "example_id": example.example_id,
                "split": example.split.value,
                "source_id": example.source_id,
                "source_record_id": example.source_record_id,
                "group_id": example.group_id,
                "proposed_label": example.label.value,
                "proposed_category": example.category.value,
                "text": _spreadsheet_safe(example.text),
                "review_decision": "",
                "final_label": example.label.value,
                "final_category": example.category.value,
                "reviewer_id": "",
                "reviewed_at": "",
                "reviewer_notes": "",
            }
        )
    return buffer.getvalue()


def write_classifier_review_worksheet(path: Path, content: str) -> None:
    """Create the editable worksheet without replacing existing human work."""

    if path.exists():
        raise ClassifierReviewError(
            f"review worksheet already exists; refusing to replace human work: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def audit_classifier_review_worksheet(
    path: Path,
    selected_dataset: RiskDatasetManifest,
    *,
    require_complete: bool = False,
) -> ReviewProgress:
    """Validate worksheet identity, decisions, corrections, and reviewer evidence."""

    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ClassifierReviewError(f"unable to read review worksheet: {path}") from exc
    reader = csv.DictReader(io.StringIO(content))
    if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
        raise ClassifierReviewError("review worksheet columns do not match the contract")
    rows: list[dict[str, str]] = []
    for row_number, raw_row in enumerate(reader, start=2):
        if any(key is None or not isinstance(value, str) for key, value in raw_row.items()):
            raise ClassifierReviewError(f"row {row_number}: malformed CSV row")
        rows.append({field: raw_row[field] for field in REVIEW_FIELDS})
    if len(rows) != len(selected_dataset.examples):
        raise ClassifierReviewError("review worksheet row count does not match selected dataset")

    decision_counts: Counter[str] = Counter()
    reviewer_counts: Counter[str] = Counter()
    for row_number, (row, example) in enumerate(
        zip(rows, selected_dataset.examples, strict=True), start=2
    ):
        _validate_immutable_columns(row, example, row_number)
        decision = row["review_decision"].strip().casefold()
        if not decision:
            if (
                row["reviewer_id"].strip()
                or row["reviewed_at"].strip()
                or row["reviewer_notes"].strip()
            ):
                raise ClassifierReviewError(
                    f"row {row_number}: pending rows cannot contain reviewer evidence"
                )
            if (
                row["final_label"] != row["proposed_label"]
                or row["final_category"] != row["proposed_category"]
            ):
                raise ClassifierReviewError(
                    f"row {row_number}: pending rows cannot contain label corrections"
                )
            decision_counts["pending"] += 1
            continue
        try:
            parsed_decision = ReviewDecision(decision)
            final_label = RiskLabel(row["final_label"].strip())
            final_category = RiskCategory(row["final_category"].strip())
        except ValueError as exc:
            raise ClassifierReviewError(
                f"row {row_number}: invalid decision, final label, or final category"
            ) from exc
        reviewer_id = row["reviewer_id"].strip()
        reviewed_at = row["reviewed_at"].strip()
        notes = row["reviewer_notes"].strip()
        _validate_reviewer_evidence(reviewer_id, reviewed_at, row_number)
        _require_consistent_label_category(final_label, final_category, row_number)
        proposed_label = RiskLabel(row["proposed_label"])
        proposed_category = RiskCategory(row["proposed_category"])
        changed = final_label is not proposed_label or final_category is not proposed_category
        if parsed_decision is ReviewDecision.APPROVE and changed:
            raise ClassifierReviewError(
                f"row {row_number}: changed labels require the correct decision"
            )
        if parsed_decision is ReviewDecision.CORRECT:
            if not changed:
                raise ClassifierReviewError(
                    f"row {row_number}: correct requires a changed label or category"
                )
            if not notes:
                raise ClassifierReviewError(f"row {row_number}: corrections require reviewer_notes")
        if parsed_decision is ReviewDecision.EXCLUDE and not notes:
            raise ClassifierReviewError(f"row {row_number}: exclusions require reviewer_notes")
        decision_counts[parsed_decision.value] += 1
        reviewer_counts[reviewer_id] += 1

    progress = ReviewProgress(
        total=len(rows),
        pending=decision_counts["pending"],
        approved=decision_counts[ReviewDecision.APPROVE.value],
        corrected=decision_counts[ReviewDecision.CORRECT.value],
        excluded=decision_counts[ReviewDecision.EXCLUDE.value],
        reviewer_counts=dict(sorted(reviewer_counts.items())),
    )
    if require_complete and not progress.complete:
        raise ClassifierReviewError(f"review worksheet has {progress.pending} pending decision(s)")
    return progress


def load_selected_dataset_for_review(repository_root: Path) -> RiskDatasetManifest:
    """Load the selected dataset only after validating its evidence chain."""

    root = repository_root.resolve()
    report = load_selection_report(root / DEFAULT_SELECTION_REPORT)
    try:
        validate_selection_report(report, root, require_artifacts=True)
    except ClassifierSelectionError as exc:
        raise ClassifierReviewError(str(exc)) from exc
    return load_risk_dataset(root / report.selected_dataset_path)


def _validate_immutable_columns(row: dict[str, str], example: RiskExample, row_number: int) -> None:
    expected = {
        "example_id": example.example_id,
        "split": example.split.value,
        "source_id": example.source_id,
        "source_record_id": example.source_record_id,
        "group_id": example.group_id,
        "proposed_label": example.label.value,
        "proposed_category": example.category.value,
        "text": _spreadsheet_safe(example.text),
    }
    if any(row[key] != value for key, value in expected.items()):
        raise ClassifierReviewError(
            f"row {row_number}: immutable selection evidence was modified or reordered"
        )


def _require_consistent_label_category(
    label: RiskLabel, category: RiskCategory, row_number: int
) -> None:
    if label is RiskLabel.BENIGN and category is not RiskCategory.BENIGN:
        raise ClassifierReviewError(f"row {row_number}: benign label requires benign category")
    if label is RiskLabel.ADVERSARIAL and category is RiskCategory.BENIGN:
        raise ClassifierReviewError(
            f"row {row_number}: adversarial label requires an adversarial category"
        )


def _validate_reviewer_evidence(reviewer_id: str, reviewed_at: str, row_number: int) -> None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", reviewer_id) or len(reviewer_id) > 80:
        raise ClassifierReviewError(
            f"row {row_number}: completed review requires a valid reviewer_id slug"
        )
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise ClassifierReviewError(
            f"row {row_number}: completed review requires an ISO 8601 reviewed_at timestamp"
        ) from exc
    if timestamp.utcoffset() is None:
        raise ClassifierReviewError(
            f"row {row_number}: reviewed_at timestamp must include a timezone"
        )


def _spreadsheet_safe(value: str) -> str:
    """Prevent source text from being interpreted as a spreadsheet formula."""

    if value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value
