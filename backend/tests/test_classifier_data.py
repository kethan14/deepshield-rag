import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepshield_api.classifier.cli import main
from deepshield_api.classifier.manifest import (
    ClassifierManifestError,
    load_risk_dataset,
    load_source_registry,
    render_risk_dataset,
    render_risk_dataset_schema,
    render_source_registry,
    render_source_registry_schema,
    validate_dataset_references,
)
from deepshield_api.classifier.models import (
    ArtifactStatus,
    ClassifierSource,
    ClassifierSourceRegistry,
    ReviewStatus,
    RiskCategory,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "data" / "classifier" / "source-registry.v1.json"
DATASET_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "risk-dataset.valid.json"


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("test fixture must be a JSON object")
    return payload


def example_payload(
    *,
    example_id: str,
    split: str,
    label: str,
    category: str,
    text: str,
    group_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "example_id": example_id,
        "source_id": "jackhhao-jailbreak-classification",
        "source_record_id": example_id,
        "group_id": group_id or f"group-{example_id}",
        "split": split,
        "label": label,
        "category": category,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "synthetic": True,
        "review_status": "draft",
        "reviewer_id": None,
        "reviewed_at": None,
    }


def frozen_dataset_payload() -> dict[str, object]:
    """Build the smallest fully reviewed artifact permitted by the frozen contract."""

    payload = load_json(DATASET_FIXTURE)
    payload["status"] = "frozen"
    payload["benchmark_case_ids_sha256"] = "1" * 64
    examples: list[dict[str, object]] = []
    split_counts = {
        "train": 500,
        "validation": 100,
        "test": 200,
        "external-test": 200,
    }
    for split, count in split_counts.items():
        for index in range(count):
            label = "benign" if index % 2 == 0 else "adversarial"
            category = "benign" if label == "benign" else "direct-prompt-injection"
            text = f"Synthetic classifier example for {split} number {index}."
            item = example_payload(
                example_id=f"{split}-example-{index:04d}",
                split=split,
                label=label,
                category=category,
                text=text,
            )
            if split == "external-test":
                item["source_id"] = "spml-contextual-prompt-injection"
            item.update(
                review_status="approved",
                reviewer_id="reviewer-one",
                reviewed_at="2026-10-04T02:00:00Z",
            )
            examples.append(item)
    payload["examples"] = examples
    return payload


def test_frozen_source_registry_records_independent_licensed_uses() -> None:
    registry = load_source_registry(REGISTRY_PATH)

    assert registry.status is ArtifactStatus.FROZEN
    assert [source.purpose.value for source in registry.sources] == [
        "development",
        "external-test",
    ]
    assert {source.license_spdx for source in registry.sources} == {"Apache-2.0", "MIT"}
    assert all(len(source.revision) == 40 for source in registry.sources)


def test_committed_dataset_fixture_has_valid_provenance() -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)

    validate_dataset_references(dataset, REPOSITORY_ROOT)

    assert dataset.status is ArtifactStatus.DRAFT
    assert {example.split for example in dataset.examples} == set(RiskSplit)
    assert dataset.benchmark_case_ids_sha256 is None


def test_renderers_are_deterministic_and_schemas_have_expected_titles() -> None:
    registry = load_source_registry(REGISTRY_PATH)
    dataset = load_risk_dataset(DATASET_FIXTURE)

    assert render_source_registry(registry) == render_source_registry(registry)
    assert render_risk_dataset(dataset) == render_risk_dataset(dataset)
    assert '"title": "ClassifierSourceRegistry"' in render_source_registry_schema()
    assert '"title": "RiskDatasetManifest"' in render_risk_dataset_schema()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "main", "revision"),
        ("files", ["../unsafe.csv"], "safe CSV"),
        ("label_mapping", {"benign": "benign"}, "benign and adversarial"),
        ("dataset_url", "https://example.com/dataset", "must match dataset_id"),
    ],
)
def test_source_rejects_unpinned_or_inconsistent_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = load_json(REGISTRY_PATH)["sources"]
    assert isinstance(payload, list)
    source = copy.deepcopy(payload[0])
    assert isinstance(source, dict)
    source[field] = value

    with pytest.raises(ValidationError, match=message):
        ClassifierSource.model_validate(source)


def test_development_source_requires_leakage_resistant_grouping() -> None:
    payload = load_json(REGISTRY_PATH)["sources"]
    assert isinstance(payload, list)
    source = copy.deepcopy(payload[0])
    assert isinstance(source, dict)
    source["grouping_strategy"] = "external-source-split"

    with pytest.raises(ValidationError, match="development sources"):
        ClassifierSource.model_validate(source)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"files": ["same.csv", "same.csv"]}, "unique"),
        ({"lfs_sha256": {"undeclared.csv": "0" * 64}}, "declared source files"),
        (
            {"grouping_strategy": "upstream-column"},
            "requires a group column",
        ),
        (
            {
                "grouping_strategy": "derived-near-duplicate-cluster",
                "columns": {
                    "text": "prompt",
                    "label": "type",
                    "group": "family",
                    "context": None,
                    "severity": None,
                },
            },
            "group column requires",
        ),
    ],
)
def test_source_rejects_unsafe_grouping_or_checksum_metadata(
    updates: dict[str, object],
    message: str,
) -> None:
    sources = load_json(REGISTRY_PATH)["sources"]
    assert isinstance(sources, list)
    source = copy.deepcopy(sources[0])
    assert isinstance(source, dict)
    source.update(updates)

    with pytest.raises(ValidationError, match=message):
        ClassifierSource.model_validate(source)


def test_registry_rejects_duplicate_sources_and_missing_purpose() -> None:
    payload = load_json(REGISTRY_PATH)
    sources = payload["sources"]
    assert isinstance(sources, list)
    payload["sources"] = [sources[0], sources[0]]

    with pytest.raises(ValidationError, match="source IDs must be unique"):
        ClassifierSourceRegistry.model_validate(payload)


def test_registry_rejects_duplicate_dataset_missing_purpose_and_naive_timestamp() -> None:
    payload = load_json(REGISTRY_PATH)
    sources = payload["sources"]
    assert isinstance(sources, list)
    second = copy.deepcopy(sources[1])
    assert isinstance(second, dict)
    first = sources[0]
    assert isinstance(first, dict)
    second["dataset_id"] = first["dataset_id"]
    second["dataset_url"] = (
        f"https://huggingface.co/datasets/{first['dataset_id']}/tree/{second['revision']}"
    )
    payload["sources"] = [first, second]
    with pytest.raises(ValidationError, match="dataset IDs must be unique"):
        ClassifierSourceRegistry.model_validate(payload)

    payload = load_json(REGISTRY_PATH)
    sources = payload["sources"]
    assert isinstance(sources, list)
    first = copy.deepcopy(sources[0])
    second = copy.deepcopy(sources[1])
    assert isinstance(first, dict) and isinstance(second, dict)
    second["purpose"] = "development"
    second["grouping_strategy"] = "derived-near-duplicate-cluster"
    payload["sources"] = [first, second]
    with pytest.raises(ValidationError, match="requires development and external-test"):
        ClassifierSourceRegistry.model_validate(payload)

    payload = load_json(REGISTRY_PATH)
    payload["created_at"] = "2026-10-04T02:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        ClassifierSourceRegistry.model_validate(payload)


@pytest.mark.parametrize(
    ("label", "category"),
    [("benign", "role-manipulation"), ("adversarial", "benign")],
)
def test_example_requires_consistent_binary_and_category_labels(
    label: str,
    category: str,
) -> None:
    payload = example_payload(
        example_id="invalid-label",
        split="train",
        label=label,
        category=category,
        text="A sufficiently long classifier example.",
    )

    with pytest.raises(ValidationError, match="category"):
        RiskExample.model_validate(payload)


def test_example_rejects_wrong_text_hash_and_incomplete_review() -> None:
    payload = example_payload(
        example_id="wrong-hash",
        split="train",
        label="benign",
        category="benign",
        text="A benign policy question for validation.",
    )
    payload["text_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="text_sha256"):
        RiskExample.model_validate(payload)

    payload["text_sha256"] = hashlib.sha256(str(payload["text"]).encode()).hexdigest()
    payload["review_status"] = "approved"
    with pytest.raises(ValidationError, match="approved examples"):
        RiskExample.model_validate(payload)


def test_approved_example_requires_timezone_aware_review() -> None:
    payload = example_payload(
        example_id="approved-example",
        split="train",
        label="benign",
        category="benign",
        text="A reviewed benign policy question.",
    )
    payload.update(
        review_status=ReviewStatus.APPROVED,
        reviewer_id="reviewer-one",
        reviewed_at=datetime(2026, 10, 4),
    )

    with pytest.raises(ValidationError, match="timezone"):
        RiskExample.model_validate(payload)

    payload["reviewed_at"] = datetime(2026, 10, 4, tzinfo=UTC)
    assert RiskExample.model_validate(payload).review_status is ReviewStatus.APPROVED


def test_draft_example_rejects_review_metadata() -> None:
    payload = example_payload(
        example_id="draft-example",
        split="train",
        label="benign",
        category="benign",
        text="A draft benign policy question.",
    )
    payload["reviewer_id"] = "reviewer-one"

    with pytest.raises(ValidationError, match="draft examples"):
        RiskExample.model_validate(payload)


def test_dataset_rejects_duplicate_text_and_cross_split_groups() -> None:
    payload = load_json(DATASET_FIXTURE)
    examples = payload["examples"]
    assert isinstance(examples, list)
    duplicate = copy.deepcopy(examples[0])
    assert isinstance(duplicate, dict)
    duplicate["example_id"] = "duplicate-text"
    duplicate["group_id"] = "different-group"
    examples.append(duplicate)

    with pytest.raises(ValidationError, match="duplicate normalized texts"):
        RiskDatasetManifest.model_validate(payload)

    payload = load_json(DATASET_FIXTURE)
    examples = payload["examples"]
    assert isinstance(examples, list)
    assert isinstance(examples[1], dict)
    examples[1]["group_id"] = examples[0]["group_id"]
    with pytest.raises(ValidationError, match="crosses data splits"):
        RiskDatasetManifest.model_validate(payload)


def test_dataset_rejects_duplicate_ids_and_naive_timestamp() -> None:
    payload = load_json(DATASET_FIXTURE)
    examples = payload["examples"]
    assert isinstance(examples, list)
    assert isinstance(examples[0], dict) and isinstance(examples[1], dict)
    examples[1]["example_id"] = examples[0]["example_id"]
    with pytest.raises(ValidationError, match="duplicate example IDs"):
        RiskDatasetManifest.model_validate(payload)

    payload = load_json(DATASET_FIXTURE)
    payload["created_at"] = "2026-10-04T02:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        RiskDatasetManifest.model_validate(payload)


def test_frozen_dataset_requires_exclusion_checksum_reviews_splits_and_labels() -> None:
    payload = load_json(DATASET_FIXTURE)
    payload["status"] = "frozen"

    with pytest.raises(ValidationError, match="benchmark exclusion checksum"):
        RiskDatasetManifest.model_validate(payload)


def test_frozen_dataset_rejects_toy_sample_counts() -> None:
    payload = load_json(DATASET_FIXTURE)
    payload["status"] = "frozen"
    payload["benchmark_case_ids_sha256"] = "1" * 64
    examples = payload["examples"]
    assert isinstance(examples, list)
    for item in examples:
        assert isinstance(item, dict)
        item["review_status"] = "approved"
        item["reviewer_id"] = "reviewer-one"
        item["reviewed_at"] = "2026-10-04T02:00:00Z"

    with pytest.raises(ValidationError, match="minimum example count"):
        RiskDatasetManifest.model_validate(payload)


def test_minimum_frozen_dataset_contract_can_be_satisfied() -> None:
    dataset = RiskDatasetManifest.model_validate(frozen_dataset_payload())

    assert dataset.status is ArtifactStatus.FROZEN
    assert len(dataset.examples) == 1000
    assert {example.split for example in dataset.examples} == set(RiskSplit)


def test_frozen_dataset_enforces_per_split_counts_and_labels() -> None:
    payload = frozen_dataset_payload()
    examples = payload["examples"]
    assert isinstance(examples, list)
    removed = next(item for item in examples if item["split"] == "train")
    examples.remove(removed)
    replacement = example_payload(
        example_id="extra-external-example",
        split="external-test",
        label="benign",
        category="benign",
        text="An extra independent external example for count validation.",
    )
    replacement.update(
        source_id="spml-contextual-prompt-injection",
        review_status="approved",
        reviewer_id="reviewer-one",
        reviewed_at="2026-10-04T02:00:00Z",
    )
    examples.append(replacement)
    with pytest.raises(ValidationError, match="train does not meet its minimum count"):
        RiskDatasetManifest.model_validate(payload)

    payload = frozen_dataset_payload()
    examples = payload["examples"]
    assert isinstance(examples, list)
    for item in examples:
        if item["split"] == "validation":
            item["label"] = "adversarial"
            item["category"] = "direct-prompt-injection"
    with pytest.raises(ValidationError, match="validation requires both labels"):
        RiskDatasetManifest.model_validate(payload)


def test_reference_validation_rejects_checksum_unknown_source_and_purpose_mixing() -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    wrong_checksum = dataset.model_copy(update={"source_registry_sha256": "0" * 64})
    with pytest.raises(ClassifierManifestError, match="checksum"):
        validate_dataset_references(wrong_checksum, REPOSITORY_ROOT)

    unknown = dataset.model_copy(deep=True)
    unknown.examples[0].source_id = "unknown-source"
    with pytest.raises(ClassifierManifestError, match="unknown source ID"):
        validate_dataset_references(unknown, REPOSITORY_ROOT)

    mixed = dataset.model_copy(deep=True)
    mixed.examples[-1].split = RiskSplit.TRAIN
    with pytest.raises(ClassifierManifestError, match="cannot enter development"):
        validate_dataset_references(mixed, REPOSITORY_ROOT)

    development_in_external = dataset.model_copy(deep=True)
    development_in_external.examples[0].split = RiskSplit.EXTERNAL_TEST
    with pytest.raises(ClassifierManifestError, match="cannot enter the external-test"):
        validate_dataset_references(development_in_external, REPOSITORY_ROOT)


def test_reference_validation_rejects_escaped_or_missing_registry(tmp_path: Path) -> None:
    dataset = load_risk_dataset(DATASET_FIXTURE)
    escaped = dataset.model_copy(update={"source_registry_path": "../registry.json"})
    with pytest.raises(ClassifierManifestError, match="outside data/classifier"):
        validate_dataset_references(escaped, tmp_path)

    with pytest.raises(ClassifierManifestError, match="unable to read source registry"):
        validate_dataset_references(dataset, tmp_path)


def test_loaders_sanitize_missing_and_invalid_json_errors(tmp_path: Path) -> None:
    with pytest.raises(ClassifierManifestError, match="unable to read"):
        load_source_registry(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ClassifierManifestError, match="not valid JSON"):
        load_risk_dataset(invalid)

    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ClassifierManifestError, match="validation failed"):
        load_risk_dataset(invalid)


def test_classifier_cli_checks_schemas_and_validates_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_schema = tmp_path / "source-schema.json"
    dataset_schema = tmp_path / "dataset-schema.json"
    assert main(["schema", "sources", "--output", str(source_schema)]) == 0
    assert main(["schema", "sources", "--output", str(source_schema), "--check"]) == 0
    assert main(["schema", "dataset", "--output", str(dataset_schema)]) == 0
    assert main(["schema", "dataset", "--output", str(dataset_schema), "--check"]) == 0
    assert main(["validate-sources", str(REGISTRY_PATH)]) == 0
    assert (
        main(
            [
                "validate-dataset",
                str(DATASET_FIXTURE),
                "--repository-root",
                str(REPOSITORY_ROOT),
            ]
        )
        == 0
    )
    assert "Validated draft classifier dataset" in capsys.readouterr().out


def test_classifier_cli_requires_frozen_training_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "validate-dataset",
            str(DATASET_FIXTURE),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--require-frozen",
        ]
    )

    assert result == 2
    assert "must be frozen before training" in capsys.readouterr().err


def test_classifier_cli_reports_schema_drift_and_missing_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    drifted = tmp_path / "drifted.json"
    drifted.write_text("{}\n", encoding="utf-8")
    assert main(["schema", "dataset", "--output", str(drifted), "--check"]) == 2
    assert "schema drift detected" in capsys.readouterr().err

    missing = tmp_path / "missing.json"
    assert main(["schema", "sources", "--output", str(missing), "--check"]) == 2
    assert "unable to read schema" in capsys.readouterr().err


def test_enums_preserve_expected_research_labels() -> None:
    assert set(RiskLabel) == {RiskLabel.BENIGN, RiskLabel.ADVERSARIAL}
    assert RiskCategory.SENSITIVE_DATA_EXFILTRATION.value == "sensitive-data-exfiltration"
