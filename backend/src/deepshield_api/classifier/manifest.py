"""Deterministic I/O and schema export for classifier-data artifacts."""

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from deepshield_api.classifier.models import (
    ClassifierCollectionReceipt,
    ClassifierPreprocessingReport,
    ClassifierSourceRegistry,
    RiskDatasetManifest,
    RiskSplit,
    SourcePurpose,
)

ManifestModel = TypeVar("ManifestModel", bound=BaseModel)


class ClassifierManifestError(ValueError):
    """Raised when a classifier registry or dataset artifact is invalid."""


def load_source_registry(path: Path) -> ClassifierSourceRegistry:
    """Load a UTF-8 classifier source registry with sanitized errors."""

    return _load_model(path, ClassifierSourceRegistry, "source registry")


def load_risk_dataset(path: Path) -> RiskDatasetManifest:
    """Load a UTF-8 classifier dataset manifest with sanitized errors."""

    return _load_model(path, RiskDatasetManifest, "risk dataset")


def load_collection_receipt(path: Path) -> ClassifierCollectionReceipt:
    """Load a UTF-8 classifier source-collection receipt."""

    return _load_model(path, ClassifierCollectionReceipt, "collection receipt")


def load_preprocessing_report(path: Path) -> ClassifierPreprocessingReport:
    """Load a UTF-8 classifier preprocessing report."""

    return _load_model(path, ClassifierPreprocessingReport, "preprocessing report")


def render_source_registry_schema() -> str:
    """Return the canonical source-registry JSON Schema."""

    return json.dumps(ClassifierSourceRegistry.model_json_schema(), indent=2, sort_keys=True) + "\n"


def render_risk_dataset_schema() -> str:
    """Return the canonical classifier-dataset JSON Schema."""

    return json.dumps(RiskDatasetManifest.model_json_schema(), indent=2, sort_keys=True) + "\n"


def render_collection_receipt_schema() -> str:
    """Return the canonical classifier collection-receipt JSON Schema."""

    return (
        json.dumps(ClassifierCollectionReceipt.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


def render_source_registry(registry: ClassifierSourceRegistry) -> str:
    """Return a deterministic source-registry document."""

    return registry.model_dump_json(indent=2) + "\n"


def render_risk_dataset(dataset: RiskDatasetManifest) -> str:
    """Return a deterministic classifier-dataset document."""

    return dataset.model_dump_json(indent=2) + "\n"


def render_collection_receipt(receipt: ClassifierCollectionReceipt) -> str:
    """Return a deterministic classifier collection receipt."""

    return receipt.model_dump_json(indent=2) + "\n"


def validate_dataset_references(dataset: RiskDatasetManifest, repository_root: Path) -> None:
    """Verify the pinned registry checksum and every example's allowed source purpose."""

    root = repository_root.resolve()
    registry_path = (root / dataset.source_registry_path).resolve()
    classifier_root = (root / "data" / "classifier").resolve()
    if not registry_path.is_relative_to(classifier_root):
        raise ClassifierManifestError("source registry resolves outside data/classifier")
    try:
        registry_bytes = registry_path.read_bytes()
    except OSError as exc:
        raise ClassifierManifestError(f"unable to read source registry: {registry_path}") from exc
    actual_checksum = hashlib.sha256(registry_bytes).hexdigest()
    if actual_checksum != dataset.source_registry_sha256:
        raise ClassifierManifestError("source registry checksum does not match")
    registry = load_source_registry(registry_path)
    sources = {source.source_id: source for source in registry.sources}
    for example in dataset.examples:
        source = sources.get(example.source_id)
        if source is None:
            raise ClassifierManifestError(
                f"classifier example references unknown source ID: {example.source_id}"
            )
        if source.purpose is SourcePurpose.EXTERNAL_TEST:
            if example.split is not RiskSplit.EXTERNAL_TEST:
                raise ClassifierManifestError(
                    f"external-test source {source.source_id} cannot enter development splits"
                )
        elif example.split is RiskSplit.EXTERNAL_TEST:
            raise ClassifierManifestError(
                f"development source {source.source_id} cannot enter the external-test split"
            )


def _load_model(path: Path, model_type: type[ManifestModel], label: str) -> ManifestModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClassifierManifestError(f"unable to read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ClassifierManifestError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ClassifierManifestError(f"{label} validation failed:\n{exc}") from exc
