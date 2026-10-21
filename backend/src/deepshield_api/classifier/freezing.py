"""Apply completed human decisions and freeze reproducible classifier data."""

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from deepshield_api.classifier.manifest import (
    load_review_report,
    load_risk_dataset,
    render_review_report,
    render_risk_dataset,
    validate_dataset_references,
)
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ClassifierReviewReport,
    ReviewSplitSummary,
    ReviewStatus,
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
)
from deepshield_api.classifier.review import (
    DEFAULT_REVIEW_WORKSHEET,
    REVIEW_FIELDS,
    ClassifierReviewError,
    ReviewDecision,
    audit_classifier_review_worksheet,
)
from deepshield_api.classifier.selection import (
    ClassifierSelectionError,
    load_selection_report,
    validate_selection_report,
)

DEFAULT_SELECTION_REPORT = Path("data/classifier/selection-report.v1.json")
DEFAULT_REVIEW_REPORT = Path("data/classifier/review-report.v1.json")
DEFAULT_FROZEN_DATASET = Path("data/classifier/processed/risk-dataset-reviewed.v1.json")


@dataclass(frozen=True)
class ReviewFreezeResult:
    """Frozen local data and committed aggregate review evidence."""

    dataset: RiskDatasetManifest
    report: ClassifierReviewReport


@dataclass(frozen=True)
class _CompletedDecision:
    split: RiskSplit
    decision: ReviewDecision
    final_label: RiskLabel
    final_category: RiskCategory
    reviewer_id: str
    reviewed_at: datetime


def freeze_classifier_review(repository_root: Path) -> ReviewFreezeResult:
    """Create frozen data only from a complete, unchanged, valid review worksheet."""

    root = repository_root.resolve()
    selection_path = root / DEFAULT_SELECTION_REPORT
    worksheet_path = root / DEFAULT_REVIEW_WORKSHEET
    try:
        selection_bytes = selection_path.read_bytes()
        worksheet_bytes = worksheet_path.read_bytes()
    except OSError as exc:
        raise ClassifierReviewError(f"unable to read review freeze input: {exc.filename}") from exc

    selection = load_selection_report(selection_path)
    try:
        validate_selection_report(selection, root, require_artifacts=True)
    except ClassifierSelectionError as exc:
        raise ClassifierReviewError(str(exc)) from exc
    selected_path = root / selection.selected_dataset_path
    selected_bytes = selected_path.read_bytes()
    if _sha256(selected_bytes) != selection.selected_dataset_sha256:
        raise ClassifierReviewError("selected dataset changed during freeze validation")
    selected = load_risk_dataset(selected_path)
    progress = audit_classifier_review_worksheet(
        worksheet_path,
        selected,
        require_complete=True,
    )
    if worksheet_path.read_bytes() != worksheet_bytes:
        raise ClassifierReviewError("review worksheet changed during freeze validation")
    if selection_path.read_bytes() != selection_bytes:
        raise ClassifierReviewError("selection report changed during freeze validation")
    if selected_path.read_bytes() != selected_bytes:
        raise ClassifierReviewError("selected dataset changed during freeze validation")

    rows = _review_rows(worksheet_bytes)
    decisions = [_completed_decision(row) for row in rows]
    frozen = _apply_completed_decisions(selected, decisions)
    created_at = frozen.created_at

    frozen_content = render_risk_dataset(frozen)
    report = ClassifierReviewReport(
        report_version=1,
        created_at=created_at,
        selection_report_path="data/classifier/selection-report.v1.json",
        selection_report_sha256=_sha256(selection_bytes),
        input_dataset_path=selection.selected_dataset_path,
        input_dataset_sha256=_sha256(selected_bytes),
        review_worksheet_path=("data/classifier/processed/risk-dataset-human-review.v1.csv"),
        review_worksheet_sha256=_sha256(worksheet_bytes),
        output_dataset_path=("data/classifier/processed/risk-dataset-reviewed.v1.json"),
        output_dataset_sha256=_sha256(frozen_content.encode()),
        input_count=progress.total,
        approved_count=progress.approved,
        corrected_count=progress.corrected,
        excluded_count=progress.excluded,
        output_count=len(frozen.examples),
        reviewer_counts=progress.reviewer_counts,
        splits=_split_summaries(selected, frozen.examples, decisions),
    )
    return ReviewFreezeResult(dataset=frozen, report=report)


def write_review_freeze_result(
    result: ReviewFreezeResult,
    repository_root: Path,
    report_path: Path,
) -> None:
    """Write the ignored frozen dataset and committed report without replacement."""

    root = repository_root.resolve()
    expected_report = (root / DEFAULT_REVIEW_REPORT).resolve()
    if report_path.resolve() != expected_report:
        raise ClassifierReviewError(f"review evidence requires {DEFAULT_REVIEW_REPORT.as_posix()}")
    output_path = (root / result.report.output_dataset_path).resolve()
    processed_root = (root / "data" / "classifier" / "processed").resolve()
    if not output_path.is_relative_to(processed_root):
        raise ClassifierReviewError("frozen classifier dataset resolves outside processed data")
    output_content = render_risk_dataset(result.dataset)
    report_content = render_review_report(result.report)
    _require_compatible_target(output_path, output_content)
    _require_compatible_target(expected_report, report_content)
    _write_new(output_path, output_content)
    _write_new(expected_report, report_content)


def validate_review_report(
    report: ClassifierReviewReport,
    repository_root: Path,
    *,
    require_artifacts: bool = False,
) -> None:
    """Verify the selection, worksheet, frozen dataset, checksums, and aggregate counts."""

    root = repository_root.resolve()
    selection_path = root / report.selection_report_path
    try:
        selection_bytes = selection_path.read_bytes()
    except OSError as exc:
        raise ClassifierReviewError(f"unable to read selection report: {selection_path}") from exc
    if _sha256(selection_bytes) != report.selection_report_sha256:
        raise ClassifierReviewError("selection report checksum does not match review report")
    selection = load_selection_report(selection_path)
    try:
        validate_selection_report(selection, root, require_artifacts=require_artifacts)
    except ClassifierSelectionError as exc:
        raise ClassifierReviewError(str(exc)) from exc
    if (
        report.input_dataset_path != selection.selected_dataset_path
        or report.input_dataset_sha256 != selection.selected_dataset_sha256
    ):
        raise ClassifierReviewError("review input does not match selection evidence")

    worksheet_path = root / report.review_worksheet_path
    output_path = root / report.output_dataset_path
    worksheet_exists = worksheet_path.exists()
    output_exists = output_path.exists()
    if worksheet_exists != output_exists:
        raise ClassifierReviewError("local review artifacts are incomplete")
    if not worksheet_exists:
        if require_artifacts:
            raise ClassifierReviewError("required local review artifacts are missing")
        return
    if _sha256(worksheet_path.read_bytes()) != report.review_worksheet_sha256:
        raise ClassifierReviewError("review worksheet checksum does not match")
    if _sha256(output_path.read_bytes()) != report.output_dataset_sha256:
        raise ClassifierReviewError("frozen dataset checksum does not match")

    selected = load_risk_dataset(root / report.input_dataset_path)
    progress = audit_classifier_review_worksheet(
        worksheet_path,
        selected,
        require_complete=True,
    )
    frozen = load_risk_dataset(output_path)
    validate_dataset_references(frozen, root)
    if frozen.status is not ArtifactStatus.FROZEN:
        raise ClassifierReviewError("review output dataset must be frozen")
    decisions = [_completed_decision(row) for row in _review_rows(worksheet_path.read_bytes())]
    expected_frozen = _apply_completed_decisions(selected, decisions)
    if frozen != expected_frozen:
        raise ClassifierReviewError("frozen dataset does not match completed review decisions")
    expected_summaries = _split_summaries(selected, expected_frozen.examples, decisions)
    if (
        progress.total != report.input_count
        or progress.approved != report.approved_count
        or progress.corrected != report.corrected_count
        or progress.excluded != report.excluded_count
        or progress.reviewer_counts != report.reviewer_counts
        or len(frozen.examples) != report.output_count
        or expected_summaries != report.splits
    ):
        raise ClassifierReviewError("review aggregates do not match review report")


def load_and_validate_review_report(
    path: Path,
    repository_root: Path,
    *,
    require_artifacts: bool = False,
) -> ClassifierReviewReport:
    """Load one report and validate its complete provenance chain."""

    report = load_review_report(path)
    validate_review_report(report, repository_root, require_artifacts=require_artifacts)
    return report


def _review_rows(content: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
        raise ClassifierReviewError("review worksheet columns do not match the contract")
    return [{field: cast(str, row[field]) for field in REVIEW_FIELDS} for row in reader]


def _completed_decision(row: dict[str, str]) -> _CompletedDecision:
    return _CompletedDecision(
        split=RiskSplit(row["split"]),
        decision=ReviewDecision(row["review_decision"].strip().casefold()),
        final_label=RiskLabel(row["final_label"].strip()),
        final_category=RiskCategory(row["final_category"].strip()),
        reviewer_id=row["reviewer_id"].strip(),
        reviewed_at=datetime.fromisoformat(row["reviewed_at"].strip()),
    )


def _apply_completed_decisions(
    selected: RiskDatasetManifest,
    decisions: list[_CompletedDecision],
) -> RiskDatasetManifest:
    retained: list[RiskExample] = []
    for example, decision in zip(selected.examples, decisions, strict=True):
        if decision.decision is ReviewDecision.EXCLUDE:
            continue
        retained.append(
            RiskExample.model_validate(
                example.model_dump()
                | {
                    "label": decision.final_label,
                    "category": decision.final_category,
                    "review_status": ReviewStatus.APPROVED,
                    "reviewer_id": decision.reviewer_id,
                    "reviewed_at": decision.reviewed_at,
                }
            )
        )
    try:
        return RiskDatasetManifest.model_validate(
            selected.model_dump()
            | {
                "dataset_version": selected.dataset_version + 1,
                "status": ArtifactStatus.FROZEN,
                "created_at": max(decision.reviewed_at for decision in decisions),
                "examples": retained,
            }
        )
    except ValidationError as exc:
        raise ClassifierReviewError(
            "completed review cannot produce a valid frozen dataset; exclusions may require "
            "replacement candidates before training"
        ) from exc


def _split_summaries(
    selected: RiskDatasetManifest,
    retained: list[RiskExample],
    decisions: list[_CompletedDecision],
) -> list[ReviewSplitSummary]:
    summaries: list[ReviewSplitSummary] = []
    for split in RiskSplit:
        split_decisions = [decision for decision in decisions if decision.split is split]
        split_output = [example for example in retained if example.split is split]
        decision_counts = Counter(decision.decision for decision in split_decisions)
        label_counts = Counter(example.label for example in split_output)
        summaries.append(
            ReviewSplitSummary(
                split=split,
                input_count=sum(example.split is split for example in selected.examples),
                approved_count=decision_counts[ReviewDecision.APPROVE],
                corrected_count=decision_counts[ReviewDecision.CORRECT],
                excluded_count=decision_counts[ReviewDecision.EXCLUDE],
                output_count=len(split_output),
                output_label_counts={label: label_counts[label] for label in RiskLabel},
            )
        )
    return summaries


def _require_compatible_target(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file():
            raise ClassifierReviewError("review output is not a regular file")
        if path.read_text(encoding="utf-8") != content:
            raise ClassifierReviewError(
                f"review output already exists with different content: {path}"
            )


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except FileExistsError:
        _require_compatible_target(path, content)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
