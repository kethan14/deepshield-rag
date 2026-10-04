"""Deterministic, leakage-resistant preprocessing for classifier source data."""

import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from deepshield_api.classifier.collection import validate_collection_receipt
from deepshield_api.classifier.manifest import (
    load_collection_receipt,
    load_risk_dataset,
    load_source_registry,
    render_risk_dataset,
    validate_dataset_references,
)
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ClassifierPreprocessingReport,
    ClassifierSource,
    NormalizationRules,
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
    SourcePreprocessingSummary,
    SourcePurpose,
    SplitPreprocessingSummary,
)

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
DEFAULT_OUTPUT_DATASET_PATH = PurePosixPath("data/classifier/processed/risk-dataset.v1.json")


class ClassifierPreprocessingError(RuntimeError):
    """Raised when classifier candidates cannot be derived safely."""


@dataclass(frozen=True)
class _Candidate:
    source: ClassifierSource
    source_record_id: str
    text: str
    text_sha256: str
    label: RiskLabel


@dataclass(frozen=True)
class PreprocessingResult:
    """Validated in-memory artifacts produced by one preprocessing run."""

    dataset: RiskDatasetManifest
    report: ClassifierPreprocessingReport


def normalize_classifier_text(value: str) -> str:
    """Normalize compatibility characters and collapse all Unicode whitespace."""

    return " ".join(unicodedata.normalize("NFKC", value).split())


def preprocess_classifier_sources(
    collection_path: Path,
    repository_root: Path,
    *,
    output_dataset_path: PurePosixPath = DEFAULT_OUTPUT_DATASET_PATH,
    rules: NormalizationRules | None = None,
) -> PreprocessingResult:
    """Build a draft candidate dataset and a complete row-accounting report."""

    root = repository_root.resolve()
    resolved_collection = collection_path.resolve()
    expected_collection = (root / "data" / "classifier" / "source-collection.v1.json").resolve()
    if resolved_collection != expected_collection:
        raise ClassifierPreprocessingError("preprocessing requires source-collection.v1.json")
    _require_safe_output_path(output_dataset_path)
    active_rules = rules or NormalizationRules()
    receipt = load_collection_receipt(resolved_collection)
    validate_collection_receipt(receipt, root, require_files=True)
    registry_path = (root / receipt.source_registry_path).resolve()
    registry = load_source_registry(registry_path)
    sources = {source.source_id: source for source in registry.sources}

    retained: list[_Candidate] = []
    seen_text: dict[str, _Candidate] = {}
    source_counts: dict[str, Counter[str]] = {
        source.source_id: Counter() for source in registry.sources
    }
    ordered_files = sorted(
        enumerate(receipt.files),
        key=lambda item: (
            sources[item[1].source_id].purpose is SourcePurpose.EXTERNAL_TEST,
            item[0],
        ),
    )
    for _, collected_file in ordered_files:
        source = sources[collected_file.source_id]
        raw_path = (root / collected_file.raw_path).resolve()
        for candidate in _read_candidates(source, collected_file.source_file, raw_path):
            counts = source_counts[source.source_id]
            counts["input"] += 1
            if not candidate.text:
                counts["empty"] += 1
                continue
            if len(candidate.text) < 3:
                counts["too_short"] += 1
                continue
            if len(candidate.text) > active_rules.maximum_text_characters:
                counts["overlength"] += 1
                continue
            dedup_key = candidate.text.casefold()
            prior = seen_text.get(dedup_key)
            if prior is not None:
                if (
                    prior.source.source_id == candidate.source.source_id
                    and prior.label is not candidate.label
                ):
                    raise ClassifierPreprocessingError(
                        "normalized duplicate has conflicting labels: "
                        f"{prior.source_record_id} and {candidate.source_record_id}"
                    )
                key = (
                    "duplicate"
                    if prior.source.source_id == candidate.source.source_id
                    else "cross_source_exact"
                )
                counts[key] += 1
                continue
            seen_text[dedup_key] = candidate
            retained.append(candidate)
            counts["output"] += 1
            counts[f"label:{candidate.label.value}"] += 1

    development = [
        candidate for candidate in retained if candidate.source.purpose is SourcePurpose.DEVELOPMENT
    ]
    external = [
        candidate
        for candidate in retained
        if candidate.source.purpose is SourcePurpose.EXTERNAL_TEST
    ]
    if not development or not external:
        raise ClassifierPreprocessingError(
            "preprocessing requires retained development and external-test examples"
        )

    external, cross_source_near_duplicates = _exclude_external_near_duplicates(
        development, external, active_rules
    )
    for candidate in cross_source_near_duplicates:
        counts = source_counts[candidate.source.source_id]
        counts["output"] -= 1
        counts[f"label:{candidate.label.value}"] -= 1
        counts["cross_source_near"] += 1
    if not external:
        raise ClassifierPreprocessingError(
            "near-duplicate exclusion removed every external-test example"
        )

    cluster_ids = _cluster_development_examples(development, active_rules)
    cluster_sizes = Counter(cluster_ids)
    examples = [
        _to_example(candidate, cluster_ids[index], active_rules)
        for index, candidate in enumerate(development)
    ]
    examples.extend(
        _to_example(candidate, f"external-{candidate.text_sha256[:20]}", active_rules)
        for candidate in external
    )
    examples.sort(key=lambda example: example.example_id)

    dataset = RiskDatasetManifest(
        dataset_version=1,
        status=ArtifactStatus.DRAFT,
        created_at=receipt.collected_at,
        source_registry_path=receipt.source_registry_path,
        source_registry_sha256=receipt.source_registry_sha256,
        examples=examples,
    )
    validate_dataset_references(dataset, root)
    rendered_dataset = render_risk_dataset(dataset)
    dataset_sha256 = _sha256(rendered_dataset.encode("utf-8"))

    summaries = [
        SourcePreprocessingSummary(
            source_id=source.source_id,
            input_rows=source_counts[source.source_id]["input"],
            output_rows=source_counts[source.source_id]["output"],
            empty_rows_excluded=source_counts[source.source_id]["empty"],
            too_short_rows_excluded=source_counts[source.source_id]["too_short"],
            overlength_rows_excluded=source_counts[source.source_id]["overlength"],
            duplicate_rows_excluded=source_counts[source.source_id]["duplicate"],
            cross_source_exact_duplicates_excluded=source_counts[source.source_id][
                "cross_source_exact"
            ],
            cross_source_near_duplicates_excluded=source_counts[source.source_id][
                "cross_source_near"
            ],
            output_label_counts={
                label: source_counts[source.source_id][f"label:{label.value}"]
                for label in RiskLabel
            },
        )
        for source in registry.sources
    ]
    split_summaries = [_summarize_split(examples, split) for split in RiskSplit]
    report = ClassifierPreprocessingReport(
        report_version=1,
        created_at=receipt.collected_at,
        source_collection_path="data/classifier/source-collection.v1.json",
        source_collection_sha256=_sha256(resolved_collection.read_bytes()),
        output_dataset_path=output_dataset_path.as_posix(),
        output_dataset_sha256=dataset_sha256,
        rules=active_rules,
        sources=summaries,
        splits=split_summaries,
        development_near_duplicate_cluster_count=len(cluster_sizes),
        development_multi_member_cluster_count=sum(size > 1 for size in cluster_sizes.values()),
        development_largest_cluster_size=max(cluster_sizes.values()),
    )
    return PreprocessingResult(dataset=dataset, report=report)


def render_preprocessing_report(report: ClassifierPreprocessingReport) -> str:
    """Return a deterministic preprocessing report document."""

    return report.model_dump_json(indent=2) + "\n"


def render_preprocessing_report_schema() -> str:
    """Return the canonical preprocessing-report JSON Schema."""

    return (
        json.dumps(ClassifierPreprocessingReport.model_json_schema(), indent=2, sort_keys=True)
        + "\n"
    )


def write_preprocessing_result(
    result: PreprocessingResult,
    repository_root: Path,
    report_path: Path,
) -> None:
    """Write immutable derived artifacts, accepting only identical reruns."""

    root = repository_root.resolve()
    dataset_path = (root / result.report.output_dataset_path).resolve()
    processed_root = (root / "data" / "classifier" / "processed").resolve()
    if not dataset_path.is_relative_to(processed_root):
        raise ClassifierPreprocessingError("derived dataset path escapes the processed directory")
    expected_report = (root / "data" / "classifier" / "preprocessing-report.v1.json").resolve()
    if report_path.resolve() != expected_report:
        raise ClassifierPreprocessingError("preprocessing report must use the versioned v1 path")
    _write_immutable(dataset_path, render_risk_dataset(result.dataset))
    _write_immutable(expected_report, render_preprocessing_report(result.report))


def validate_preprocessing_report(
    report: ClassifierPreprocessingReport,
    repository_root: Path,
    *,
    require_dataset: bool,
) -> None:
    """Verify the report against its source receipt and optional local dataset."""

    root = repository_root.resolve()
    collection_path = (root / report.source_collection_path).resolve()
    classifier_root = (root / "data" / "classifier").resolve()
    if not collection_path.is_relative_to(classifier_root):
        raise ClassifierPreprocessingError("source collection resolves outside data/classifier")
    try:
        collection_bytes = collection_path.read_bytes()
    except OSError as exc:
        raise ClassifierPreprocessingError(
            f"unable to read source collection: {collection_path}"
        ) from exc
    if _sha256(collection_bytes) != report.source_collection_sha256:
        raise ClassifierPreprocessingError(
            "preprocessing source collection checksum does not match"
        )
    receipt = load_collection_receipt(collection_path)
    validate_collection_receipt(receipt, root, require_files=False)
    expected_input_counts: Counter[str] = Counter()
    for item in receipt.files:
        expected_input_counts[item.source_id] += item.row_count
    reported_sources = {summary.source_id: summary for summary in report.sources}
    if set(reported_sources) != set(expected_input_counts):
        raise ClassifierPreprocessingError(
            "preprocessing report sources do not match the source collection"
        )
    for source_id, input_rows in expected_input_counts.items():
        if reported_sources[source_id].input_rows != input_rows:
            raise ClassifierPreprocessingError(
                f"preprocessing input row count does not match collection for {source_id}"
            )
    if require_dataset:
        dataset_path = (root / report.output_dataset_path).resolve()
        processed_root = (root / "data" / "classifier" / "processed").resolve()
        if not dataset_path.is_relative_to(processed_root) or not dataset_path.is_file():
            raise ClassifierPreprocessingError("preprocessed dataset is unavailable")
        if _sha256(dataset_path.read_bytes()) != report.output_dataset_sha256:
            raise ClassifierPreprocessingError("preprocessed dataset checksum does not match")
        dataset = load_risk_dataset(dataset_path)
        validate_dataset_references(dataset, root)
        _validate_reported_dataset_counts(report, dataset)


def _validate_reported_dataset_counts(
    report: ClassifierPreprocessingReport,
    dataset: RiskDatasetManifest,
) -> None:
    source_examples: dict[str, list[RiskExample]] = {
        summary.source_id: [] for summary in report.sources
    }
    split_examples: dict[RiskSplit, list[RiskExample]] = {split: [] for split in RiskSplit}
    for example in dataset.examples:
        if example.source_id not in source_examples:
            raise ClassifierPreprocessingError(
                f"preprocessed dataset contains an unreported source: {example.source_id}"
            )
        source_examples[example.source_id].append(example)
        split_examples[example.split].append(example)
    for source_summary in report.sources:
        examples = source_examples[source_summary.source_id]
        labels = Counter(example.label for example in examples)
        if len(examples) != source_summary.output_rows or any(
            labels[label] != source_summary.output_label_counts[label] for label in RiskLabel
        ):
            raise ClassifierPreprocessingError(
                f"preprocessing source summary does not match dataset: {source_summary.source_id}"
            )
    for split_summary in report.splits:
        examples = split_examples[split_summary.split]
        labels = Counter(example.label for example in examples)
        if (
            len(examples) != split_summary.row_count
            or len({example.group_id for example in examples}) != split_summary.group_count
            or any(labels[label] != split_summary.label_counts[label] for label in RiskLabel)
        ):
            raise ClassifierPreprocessingError(
                f"preprocessing split summary does not match dataset: {split_summary.split.value}"
            )


def _read_candidates(
    source: ClassifierSource,
    source_file: str,
    path: Path,
) -> list[_Candidate]:
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ClassifierPreprocessingError(
            f"unable to read collected source file: {source.source_id}/{source_file}"
        ) from exc
    reader = csv.DictReader(StringIO(text, newline=""))
    candidates: list[_Candidate] = []
    try:
        for row_number, row in enumerate(reader, start=1):
            raw_text = row.get(source.columns.text)
            raw_label = row.get(source.columns.label)
            if raw_text is None or raw_label is None:
                raise ClassifierPreprocessingError(
                    f"collected row is missing registered columns: {source.source_id}/{source_file}"
                )
            label = source.label_mapping.get(raw_label)
            if label is None:
                raise ClassifierPreprocessingError(
                    f"collected row has an unmapped label: {source.source_id}/{source_file}"
                )
            normalized = normalize_classifier_text(raw_text)
            candidates.append(
                _Candidate(
                    source=source,
                    source_record_id=f"{source_file}#row-{row_number:06d}",
                    text=normalized,
                    text_sha256=_sha256(normalized.encode("utf-8")),
                    label=label,
                )
            )
    except csv.Error as exc:
        raise ClassifierPreprocessingError(
            f"collected source is not valid CSV: {source.source_id}/{source_file}"
        ) from exc
    return candidates


def _cluster_development_examples(
    candidates: list[_Candidate], rules: NormalizationRules
) -> list[str]:
    parent = list(range(len(candidates)))
    shingles = [
        _token_shingles(candidate.text, rules.near_duplicate_token_ngram_size)
        for candidate in candidates
    ]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    threshold = rules.near_duplicate_jaccard_threshold
    for left in range(len(candidates)):
        left_shingles = shingles[left]
        for right in range(left + 1, len(candidates)):
            right_shingles = shingles[right]
            size_ratio = min(len(left_shingles), len(right_shingles)) / max(
                len(left_shingles), len(right_shingles)
            )
            if size_ratio < threshold:
                continue
            similarity = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
            if similarity >= threshold:
                union(left, right)

    component_hashes: dict[int, list[str]] = {}
    for index, candidate in enumerate(candidates):
        component_hashes.setdefault(find(index), []).append(candidate.text_sha256)
    group_ids = {root: f"near-{min(hashes)[:20]}" for root, hashes in component_hashes.items()}
    return [group_ids[find(index)] for index in range(len(candidates))]


def _exclude_external_near_duplicates(
    development: list[_Candidate],
    external: list[_Candidate],
    rules: NormalizationRules,
) -> tuple[list[_Candidate], list[_Candidate]]:
    """Remove external examples too similar to any development example."""

    size = rules.near_duplicate_token_ngram_size
    threshold = rules.near_duplicate_jaccard_threshold
    development_shingles = [_token_shingles(candidate.text, size) for candidate in development]
    inverted: dict[str, list[int]] = {}
    for index, shingles in enumerate(development_shingles):
        for shingle in shingles:
            inverted.setdefault(shingle, []).append(index)

    kept: list[_Candidate] = []
    excluded: list[_Candidate] = []
    for candidate in external:
        candidate_shingles = _token_shingles(candidate.text, size)
        shared_counts: Counter[int] = Counter()
        for shingle in candidate_shingles:
            shared_counts.update(inverted.get(shingle, ()))
        is_near_duplicate = False
        for development_index, shared_count in shared_counts.items():
            reference = development_shingles[development_index]
            size_ratio = min(len(candidate_shingles), len(reference)) / max(
                len(candidate_shingles), len(reference)
            )
            if size_ratio < threshold:
                continue
            similarity = shared_count / (len(candidate_shingles) + len(reference) - shared_count)
            if similarity >= threshold:
                is_near_duplicate = True
                break
        (excluded if is_near_duplicate else kept).append(candidate)
    return kept, excluded


def _token_shingles(text: str, size: int) -> frozenset[str]:
    tokens = _TOKEN_PATTERN.findall(text.casefold())
    if len(tokens) < size:
        return frozenset(tokens or [text.casefold()])
    return frozenset(
        "\x1f".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def _to_example(
    candidate: _Candidate,
    group_id: str,
    rules: NormalizationRules,
) -> RiskExample:
    split = (
        RiskSplit.EXTERNAL_TEST
        if candidate.source.purpose is SourcePurpose.EXTERNAL_TEST
        else _development_split(group_id, rules)
    )
    category = (
        RiskCategory.BENIGN
        if candidate.label is RiskLabel.BENIGN
        else RiskCategory.OTHER_ADVERSARIAL
    )
    return RiskExample(
        example_id=f"{candidate.source.source_id}-{candidate.text_sha256[:20]}",
        source_id=candidate.source.source_id,
        source_record_id=candidate.source_record_id,
        group_id=group_id,
        split=split,
        label=candidate.label,
        category=category,
        text=candidate.text,
        text_sha256=candidate.text_sha256,
    )


def _development_split(group_id: str, rules: NormalizationRules) -> RiskSplit:
    value = (
        int(
            hashlib.sha256(f"{rules.development_split_seed}:{group_id}".encode()).hexdigest(),
            16,
        )
        % 10_000
    )
    train_limit = rules.development_split_percentages[RiskSplit.TRAIN] * 100
    validation_limit = train_limit + rules.development_split_percentages[RiskSplit.VALIDATION] * 100
    if value < train_limit:
        return RiskSplit.TRAIN
    if value < validation_limit:
        return RiskSplit.VALIDATION
    return RiskSplit.TEST


def _summarize_split(examples: list[RiskExample], split: RiskSplit) -> SplitPreprocessingSummary:
    selected = [example for example in examples if example.split is split]
    if not selected:
        raise ClassifierPreprocessingError(f"preprocessing produced an empty {split.value} split")
    labels = Counter(example.label for example in selected)
    return SplitPreprocessingSummary(
        split=split,
        row_count=len(selected),
        group_count=len({example.group_id for example in selected}),
        label_counts={label: labels[label] for label in RiskLabel},
    )


def _require_safe_output_path(path: PurePosixPath) -> None:
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:3] != ("data", "classifier", "processed")
        or path.suffix.casefold() != ".json"
    ):
        raise ClassifierPreprocessingError(
            "output dataset must be a JSON path under data/classifier/processed"
        )


def _write_immutable(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ClassifierPreprocessingError("preprocessing output is not a regular file")
        if path.read_bytes() != encoded:
            raise ClassifierPreprocessingError(
                f"preprocessing output already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ClassifierPreprocessingError("preprocessing output directory is a symbolic link")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
