"""Benchmark-safe, group-preserving selection for C2 classifier review."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from deepshield_api.classifier.manifest import (
    load_preprocessing_report,
    load_risk_dataset,
    render_risk_dataset,
    validate_dataset_references,
)
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ClassifierSelectionReport,
    ClassifierSelectionRules,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
    SelectionSplitSummary,
)
from deepshield_api.classifier.preprocessing import normalize_classifier_text
from deepshield_api.evaluation.manifest import load_benchmark_manifest
from deepshield_api.evaluation.models import BenchmarkManifest

INPUT_DATASET_PATH = Path("data/classifier/processed/risk-dataset.v1.json")
BENCHMARK_PATH = Path("data/evaluation/f1-security-benchmark.v1.json")
SELECTED_DATASET_PATH = Path("data/classifier/processed/risk-dataset-selected.v1.json")
REVIEW_PACKET_PATH = Path("data/classifier/processed/risk-dataset-selection-review.v1.csv")
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
    "reviewer_id",
    "reviewer_notes",
)


class ClassifierSelectionError(RuntimeError):
    """Raised when review-set selection or its evidence is invalid."""


@dataclass(frozen=True)
class SelectionResult:
    """In-memory outputs from one deterministic selection run."""

    dataset: RiskDatasetManifest
    report: ClassifierSelectionReport
    review_packet: str


def select_classifier_review_candidates(
    repository_root: Path,
    *,
    rules: ClassifierSelectionRules | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SelectionResult:
    """Exclude benchmark overlap and select a balanced, group-safe review set."""

    root = repository_root.resolve()
    input_path = root / INPUT_DATASET_PATH
    benchmark_path = root / BENCHMARK_PATH
    selection_rules = rules or ClassifierSelectionRules()

    try:
        input_bytes = input_path.read_bytes()
        benchmark_bytes = benchmark_path.read_bytes()
    except OSError as exc:
        raise ClassifierSelectionError(f"unable to read selection input: {exc.filename}") from exc

    candidates = load_risk_dataset(input_path)
    validate_dataset_references(candidates, root)
    if candidates.status is not ArtifactStatus.DRAFT:
        raise ClassifierSelectionError("selection input must be the draft candidate dataset")
    benchmark = load_benchmark_manifest(benchmark_path)
    now = clock() if clock is not None else max(candidates.created_at, benchmark.created_at)
    if now.utcoffset() is None:
        raise ClassifierSelectionError("selection clock must return a timezone-aware timestamp")
    benchmark_inputs = _benchmark_inputs(benchmark)
    benchmark_case_ids_sha256 = _hash_values(case.case_id for case in benchmark.cases)
    benchmark_input_texts_sha256 = _hash_values(benchmark_inputs)

    exact_inputs = {text.casefold() for text in benchmark_inputs}
    benchmark_shingles = [
        _token_shingles(text, selection_rules.benchmark_near_duplicate_token_ngram_size)
        for text in benchmark_inputs
    ]
    shingle_index: dict[str, set[int]] = defaultdict(set)
    for index, shingles in enumerate(benchmark_shingles):
        for shingle in shingles:
            shingle_index[shingle].add(index)

    exact_excluded = 0
    near_excluded = 0
    eligible: list[RiskExample] = []
    for example in candidates.examples:
        normalized = normalize_classifier_text(example.text).casefold()
        if normalized in exact_inputs:
            exact_excluded += 1
            continue
        if _has_near_benchmark_match(
            normalized,
            benchmark_shingles,
            shingle_index,
            selection_rules,
        ):
            near_excluded += 1
            continue
        eligible.append(example)

    selected: list[RiskExample] = []
    split_summaries: list[SelectionSplitSummary] = []
    for split in RiskSplit:
        chosen = _select_split(eligible, split, selection_rules)
        selected.extend(chosen)
        labels = Counter(example.label for example in chosen)
        split_summaries.append(
            SelectionSplitSummary(
                split=split,
                target_count=selection_rules.target_split_counts[split],
                selected_count=len(chosen),
                group_count=len({example.group_id for example in chosen}),
                label_counts={label: labels[label] for label in RiskLabel},
            )
        )

    split_order = {split: index for index, split in enumerate(RiskSplit)}
    selected.sort(key=lambda item: (split_order[item.split], item.example_id))
    dataset = candidates.model_copy(
        update={
            "dataset_version": candidates.dataset_version + 1,
            "status": ArtifactStatus.DRAFT,
            "created_at": now,
            "benchmark_case_ids_sha256": benchmark_case_ids_sha256,
            "examples": selected,
        }
    )
    dataset = RiskDatasetManifest.model_validate(dataset.model_dump())
    dataset_content = render_risk_dataset(dataset)
    review_packet = render_selection_review_packet(dataset)
    report = ClassifierSelectionReport(
        report_version=1,
        created_at=now,
        input_dataset_path="data/classifier/processed/risk-dataset.v1.json",
        input_dataset_sha256=_sha256(input_bytes),
        benchmark_path="data/evaluation/f1-security-benchmark.v1.json",
        benchmark_sha256=_sha256(benchmark_bytes),
        benchmark_case_ids_sha256=benchmark_case_ids_sha256,
        benchmark_input_texts_sha256=benchmark_input_texts_sha256,
        benchmark_case_count=len(benchmark.cases),
        benchmark_input_text_count=len(benchmark_inputs),
        rules=selection_rules,
        input_candidate_count=len(candidates.examples),
        exact_benchmark_matches_excluded=exact_excluded,
        near_benchmark_matches_excluded=near_excluded,
        eligible_candidate_count=len(eligible),
        selected_dataset_path="data/classifier/processed/risk-dataset-selected.v1.json",
        selected_dataset_sha256=_sha256(dataset_content.encode()),
        review_packet_path=("data/classifier/processed/risk-dataset-selection-review.v1.csv"),
        review_packet_sha256=_sha256(review_packet.encode()),
        splits=split_summaries,
    )
    return SelectionResult(dataset=dataset, report=report, review_packet=review_packet)


def render_selection_report(report: ClassifierSelectionReport) -> str:
    """Return a deterministic selection report."""

    return report.model_dump_json(indent=2) + "\n"


def render_selection_report_schema() -> str:
    """Return the canonical selection-report JSON Schema."""

    return (
        json.dumps(ClassifierSelectionReport.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


def render_selection_review_packet(dataset: RiskDatasetManifest) -> str:
    """Create a local worksheet without inventing human-review decisions."""

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
                "text": example.text,
                "review_decision": "",
                "reviewer_id": "",
                "reviewer_notes": "",
            }
        )
    return buffer.getvalue()


def write_selection_result(
    result: SelectionResult,
    repository_root: Path,
    report_path: Path,
) -> None:
    """Write local review artifacts and committed aggregate evidence immutably."""

    root = repository_root.resolve()
    expected_report_path = (root / "data/classifier/selection-report.v1.json").resolve()
    if report_path.resolve() != expected_report_path:
        raise ClassifierSelectionError(
            "selection evidence requires data/classifier/selection-report.v1.json"
        )
    _write_immutable(
        root / result.report.selected_dataset_path,
        render_risk_dataset(result.dataset),
    )
    _write_immutable(root / result.report.review_packet_path, result.review_packet)
    _write_immutable(report_path.resolve(), render_selection_report(result.report))


def load_selection_report(path: Path) -> ClassifierSelectionReport:
    """Load and validate a UTF-8 selection report with sanitized errors."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClassifierSelectionError(f"unable to read selection report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ClassifierSelectionError(
            f"selection report is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    try:
        return ClassifierSelectionReport.model_validate(payload)
    except ValidationError as exc:
        raise ClassifierSelectionError(f"selection report validation failed:\n{exc}") from exc


def validate_selection_report(
    report: ClassifierSelectionReport,
    repository_root: Path,
    *,
    require_artifacts: bool = False,
) -> None:
    """Verify benchmark evidence and, when present, every local selected artifact."""

    root = repository_root.resolve()
    preprocessing = load_preprocessing_report(root / "data/classifier/preprocessing-report.v1.json")
    if (
        preprocessing.output_dataset_path != report.input_dataset_path
        or preprocessing.output_dataset_sha256 != report.input_dataset_sha256
    ):
        raise ClassifierSelectionError(
            "selection input does not match the committed preprocessing evidence"
        )
    if sum(source.output_rows for source in preprocessing.sources) != report.input_candidate_count:
        raise ClassifierSelectionError(
            "selection input count does not match the committed preprocessing evidence"
        )
    benchmark_path = _safe_path(root, report.benchmark_path, root / "data" / "evaluation")
    benchmark_bytes = _read_required(benchmark_path, "benchmark")
    if _sha256(benchmark_bytes) != report.benchmark_sha256:
        raise ClassifierSelectionError("benchmark checksum does not match selection report")
    benchmark = load_benchmark_manifest(benchmark_path)
    inputs = _benchmark_inputs(benchmark)
    if len(benchmark.cases) != report.benchmark_case_count:
        raise ClassifierSelectionError("benchmark case count does not match selection report")
    if len(inputs) != report.benchmark_input_text_count:
        raise ClassifierSelectionError("benchmark input count does not match selection report")
    if _hash_values(case.case_id for case in benchmark.cases) != report.benchmark_case_ids_sha256:
        raise ClassifierSelectionError("benchmark case-ID checksum does not match")
    if _hash_values(inputs) != report.benchmark_input_texts_sha256:
        raise ClassifierSelectionError("benchmark input-text checksum does not match")

    input_path = _safe_path(root, report.input_dataset_path, root / "data" / "classifier")
    if input_path.exists():
        if _sha256(input_path.read_bytes()) != report.input_dataset_sha256:
            raise ClassifierSelectionError("input candidate checksum does not match")
    elif require_artifacts:
        raise ClassifierSelectionError("required input candidate dataset is missing")

    dataset_path = _safe_path(root, report.selected_dataset_path, root / "data" / "classifier")
    review_path = _safe_path(root, report.review_packet_path, root / "data" / "classifier")
    dataset_exists = dataset_path.exists()
    review_exists = review_path.exists()
    if dataset_exists != review_exists:
        raise ClassifierSelectionError("local selection artifacts are incomplete")
    if not dataset_exists:
        if require_artifacts:
            raise ClassifierSelectionError("required local selection artifacts are missing")
        return
    dataset_bytes = dataset_path.read_bytes()
    review_bytes = review_path.read_bytes()
    if _sha256(dataset_bytes) != report.selected_dataset_sha256:
        raise ClassifierSelectionError("selected dataset checksum does not match")
    if _sha256(review_bytes) != report.review_packet_sha256:
        raise ClassifierSelectionError("review packet checksum does not match")
    dataset = load_risk_dataset(dataset_path)
    validate_dataset_references(dataset, root)
    if dataset.status is not ArtifactStatus.DRAFT:
        raise ClassifierSelectionError("selected dataset must remain draft before human review")
    if dataset.benchmark_case_ids_sha256 != report.benchmark_case_ids_sha256:
        raise ClassifierSelectionError("selected dataset benchmark checksum does not match")
    _validate_no_benchmark_overlap(dataset.examples, inputs, report.rules)
    _validate_split_summaries(dataset.examples, report.splits)
    _validate_review_packet(review_bytes.decode("utf-8"), dataset.examples)


def _benchmark_inputs(benchmark: BenchmarkManifest) -> list[str]:
    values: set[str] = set()
    for case in benchmark.cases:
        values.add(normalize_classifier_text(case.prompt))
        values.update(
            normalize_classifier_text(chunk.text) for chunk in case.synthetic_retrieval_chunks
        )
    return sorted(values, key=str.casefold)


def _select_split(
    eligible: list[RiskExample],
    split: RiskSplit,
    rules: ClassifierSelectionRules,
) -> list[RiskExample]:
    groups: dict[str, list[RiskExample]] = defaultdict(list)
    for example in eligible:
        if example.split is split:
            groups[example.group_id].append(example)
    by_label: dict[RiskLabel, list[tuple[str, list[RiskExample]]]] = {
        label: [] for label in RiskLabel
    }
    for group_id, examples in groups.items():
        labels = {example.label for example in examples}
        if len(labels) != 1:
            raise ClassifierSelectionError(f"classifier group {group_id} contains mixed labels")
        label = next(iter(labels))
        by_label[label].append((group_id, examples))

    target_per_label = rules.target_split_counts[split] // 2
    selected: list[RiskExample] = []
    for label in RiskLabel:
        ranked = sorted(
            by_label[label],
            key=lambda item: _selection_rank(rules.selection_seed, split, label, item[0]),
        )
        label_rows = 0
        for _, examples in ranked:
            selected.extend(examples)
            label_rows += len(examples)
            if label_rows >= target_per_label:
                break
        if label_rows < target_per_label:
            raise ClassifierSelectionError(
                f"split {split.value} lacks {target_per_label} eligible {label.value} examples"
            )
    return selected


def _selection_rank(seed: str, split: RiskSplit, label: RiskLabel, group_id: str) -> str:
    value = f"{seed}\0{split.value}\0{label.value}\0{group_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _has_near_benchmark_match(
    normalized: str,
    benchmark_shingles: list[frozenset[str]],
    shingle_index: dict[str, set[int]],
    rules: ClassifierSelectionRules,
) -> bool:
    shingles = _token_shingles(normalized, rules.benchmark_near_duplicate_token_ngram_size)
    candidate_indexes: set[int] = set()
    for shingle in shingles:
        candidate_indexes.update(shingle_index.get(shingle, ()))
    for index in candidate_indexes:
        other = benchmark_shingles[index]
        union = shingles | other
        similarity = len(shingles & other) / len(union) if union else 1.0
        if similarity >= rules.benchmark_near_duplicate_jaccard_threshold:
            return True
    return False


def _token_shingles(text: str, size: int) -> frozenset[str]:
    tokens = text.casefold().split()
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(
        " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def _validate_no_benchmark_overlap(
    examples: list[RiskExample],
    benchmark_inputs: list[str],
    rules: ClassifierSelectionRules,
) -> None:
    exact_inputs = {text.casefold() for text in benchmark_inputs}
    benchmark_shingles = [
        _token_shingles(text, rules.benchmark_near_duplicate_token_ngram_size)
        for text in benchmark_inputs
    ]
    shingle_index: dict[str, set[int]] = defaultdict(set)
    for index, shingles in enumerate(benchmark_shingles):
        for shingle in shingles:
            shingle_index[shingle].add(index)
    for example in examples:
        normalized = normalize_classifier_text(example.text).casefold()
        if normalized in exact_inputs or _has_near_benchmark_match(
            normalized, benchmark_shingles, shingle_index, rules
        ):
            raise ClassifierSelectionError(
                f"selected example overlaps the held-out benchmark: {example.example_id}"
            )


def _validate_split_summaries(
    examples: list[RiskExample], summaries: list[SelectionSplitSummary]
) -> None:
    for summary in summaries:
        selected = [example for example in examples if example.split is summary.split]
        labels = Counter(example.label for example in selected)
        if (
            len(selected) != summary.selected_count
            or len({example.group_id for example in selected}) != summary.group_count
            or any(labels[label] != summary.label_counts[label] for label in RiskLabel)
        ):
            raise ClassifierSelectionError(
                f"selected {summary.split.value} counts do not match the report"
            )


def _validate_review_packet(content: str, examples: list[RiskExample]) -> None:
    reader = csv.DictReader(io.StringIO(content))
    if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
        raise ClassifierSelectionError("review packet columns do not match")
    rows = list(reader)
    if [row["example_id"] for row in rows] != [example.example_id for example in examples]:
        raise ClassifierSelectionError("review packet example IDs do not match selected dataset")
    if any(row["review_decision"] or row["reviewer_id"] or row["reviewer_notes"] for row in rows):
        raise ClassifierSelectionError("generated review packet must not contain review decisions")


def _safe_path(root: Path, value: str, expected_parent: Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(expected_parent.resolve()):
        raise ClassifierSelectionError("selection artifact resolves outside its allowed directory")
    return path


def _read_required(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ClassifierSelectionError(f"unable to read {label}: {path}") from exc


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file():
            raise ClassifierSelectionError("selection output is not a regular file")
        if path.read_text(encoding="utf-8") != content:
            raise ClassifierSelectionError(
                f"selection output already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _hash_values(values: Iterable[str]) -> str:
    canonical = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return _sha256(canonical.encode())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
