import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deepshield_api.classifier.cli import main
from deepshield_api.classifier.collection import (
    ClassifierCollectionError,
    collect_registered_sources,
    validate_collection_receipt,
)
from deepshield_api.classifier.manifest import (
    load_collection_receipt,
    render_collection_receipt,
)
from deepshield_api.classifier.models import (
    ClassifierCollectionReceipt,
    ClassifierSource,
    CollectedClassifierFile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "data" / "classifier" / "source-registry.v2.json"
COLLECTED_AT = datetime(2026, 10, 4, 2, 17, 9, tzinfo=UTC)
DEVELOPMENT_CSV = b"prompt,type\nA normal question,benign\nIgnore the rules,jailbreak\n"
EXTERNAL_CSV = (
    b"System Prompt,User Prompt,Prompt injection,Degree,Source\n"
    b"Policy bot,A normal question,0,0,fixture\n"
    b"Policy bot,,1,1,fixture\n"
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def write_test_registry(root: Path) -> Path:
    payload = load_json(REGISTRY_PATH)
    sources = payload["sources"]
    assert isinstance(sources, list)
    development = copy.deepcopy(sources[0])
    external = copy.deepcopy(sources[1])
    development["source_id"] = "dev-source"
    development["files"] = ["balanced/development.csv"]
    development["expected_rows"] = 2
    development["lfs_sha256"] = {"balanced/development.csv": sha256(DEVELOPMENT_CSV)}
    external["source_id"] = "external-source"
    external["files"] = ["external.csv"]
    external["expected_rows"] = 2
    external["lfs_sha256"] = {"external.csv": sha256(EXTERNAL_CSV)}
    payload["sources"] = [development, external]
    path = root / "data" / "classifier" / "source-registry.v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def source_bytes(source: ClassifierSource) -> bytes:
    if source.purpose.value == "development":
        return DEVELOPMENT_CSV
    return EXTERNAL_CSV


def test_collect_sources_writes_reuses_and_validates_immutable_files(tmp_path: Path) -> None:
    registry = write_test_registry(tmp_path)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    calls: list[tuple[str, str]] = []

    def downloader(source: ClassifierSource, source_file: str) -> Path:
        calls.append((source.source_id, source_file))
        path = downloads / f"{source.source_id}.csv"
        path.write_bytes(source_bytes(source))
        return path

    receipt = collect_registered_sources(
        registry,
        tmp_path,
        downloader=downloader,
        clock=lambda: COLLECTED_AT,
    )

    assert len(calls) == 2
    assert sum(item.row_count for item in receipt.files) == 4
    assert receipt.files[1].empty_text_count == 1
    assert all((tmp_path / item.raw_path).is_file() for item in receipt.files)
    validate_collection_receipt(receipt, tmp_path, require_files=True)

    repeated = collect_registered_sources(
        registry,
        tmp_path,
        downloader=lambda source, source_file: pytest.fail("existing files must be reused"),
        clock=lambda: COLLECTED_AT,
    )
    assert repeated == receipt


def test_collection_receipt_round_trip_and_optional_local_file_check(tmp_path: Path) -> None:
    registry = write_test_registry(tmp_path)
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    def downloader(source: ClassifierSource, source_file: str) -> Path:
        path = downloads / f"{source.source_id}.csv"
        path.write_bytes(source_bytes(source))
        return path

    receipt = collect_registered_sources(
        registry,
        tmp_path,
        downloader=downloader,
        clock=lambda: COLLECTED_AT,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(render_collection_receipt(receipt), encoding="utf-8")
    loaded = load_collection_receipt(receipt_path)
    assert loaded == receipt

    for item in receipt.files:
        (tmp_path / item.raw_path).unlink()
    validate_collection_receipt(loaded, tmp_path, require_files=False)
    with pytest.raises(ClassifierCollectionError, match="unavailable"):
        validate_collection_receipt(loaded, tmp_path, require_files=True)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff\xfe", "valid UTF-8"),
        (b"prompt,type\nquestion,unknown\n", "labels do not match"),
        (b"prompt,type,extra\nquestion,benign,one,two\n", "extra CSV fields"),
    ],
)
def test_collection_rejects_invalid_source_content(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    registry = write_test_registry(tmp_path)

    def downloader(source: ClassifierSource, source_file: str) -> Path:
        path = tmp_path / "download.csv"
        path.write_bytes(content if source.purpose.value == "development" else EXTERNAL_CSV)
        return path

    with pytest.raises(ClassifierCollectionError, match=message):
        collect_registered_sources(
            registry,
            tmp_path,
            downloader=downloader,
            clock=lambda: COLLECTED_AT,
        )


def test_collection_rejects_wrong_checksum_row_count_size_and_clock(tmp_path: Path) -> None:
    registry = write_test_registry(tmp_path)
    payload = load_json(registry)
    sources = payload["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["lfs_sha256"] = {"balanced/development.csv": "0" * 64}
    registry.write_text(json.dumps(payload), encoding="utf-8")

    def downloader(source: ClassifierSource, source_file: str) -> Path:
        path = tmp_path / "download.csv"
        path.write_bytes(source_bytes(source))
        return path

    with pytest.raises(ClassifierCollectionError, match="checksum"):
        collect_registered_sources(registry, tmp_path, downloader=downloader)

    registry = write_test_registry(tmp_path)
    payload = load_json(registry)
    sources = payload["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["expected_rows"] = 3
    registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ClassifierCollectionError, match="row count"):
        collect_registered_sources(registry, tmp_path, downloader=downloader)

    registry = write_test_registry(tmp_path)
    with pytest.raises(ClassifierCollectionError, match="exceeds"):
        collect_registered_sources(
            registry,
            tmp_path,
            max_file_bytes=4,
            downloader=downloader,
        )
    with pytest.raises(ClassifierCollectionError, match="timezone-aware"):
        collect_registered_sources(
            registry,
            tmp_path,
            downloader=downloader,
            clock=lambda: datetime(2026, 10, 4),
        )
    with pytest.raises(ValueError, match="greater than zero"):
        collect_registered_sources(registry, tmp_path, max_file_bytes=0)


def test_collection_requires_current_registry_path(tmp_path: Path) -> None:
    old = tmp_path / "data" / "classifier" / "source-registry.v1.json"
    old.parent.mkdir(parents=True)
    old.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ClassifierCollectionError, match="source-registry.v2.json"):
        collect_registered_sources(old, tmp_path)


def test_collection_models_reject_inconsistent_paths_and_counts() -> None:
    base = {
        "source_id": "source-one",
        "dataset_id": "owner/dataset",
        "revision": "a" * 40,
        "source_file": "file.csv",
        "raw_path": "data/classifier/raw/source-one/file.csv",
        "sha256": "b" * 64,
        "size_bytes": 10,
        "row_count": 2,
        "columns": ["text", "label"],
        "label_counts": {"benign": 1, "adversarial": 1},
        "empty_text_count": 0,
    }
    assert CollectedClassifierFile.model_validate(base).row_count == 2

    invalid = dict(base, raw_path="data/classifier/raw/other/file.csv")
    with pytest.raises(ValidationError, match="source directory"):
        CollectedClassifierFile.model_validate(invalid)
    invalid = dict(base, label_counts={"benign": 1, "adversarial": 2})
    with pytest.raises(ValidationError, match="sum"):
        CollectedClassifierFile.model_validate(invalid)
    invalid = dict(base, columns=["text", "text"])
    with pytest.raises(ValidationError, match="unique"):
        CollectedClassifierFile.model_validate(invalid)

    receipt = {
        "receipt_version": 1,
        "collected_at": COLLECTED_AT,
        "source_registry_path": "data/classifier/source-registry.v2.json",
        "source_registry_sha256": "c" * 64,
        "files": [base, base],
    }
    with pytest.raises(ValidationError, match="duplicate source files"):
        ClassifierCollectionReceipt.model_validate(receipt)


def test_collection_cli_checks_schema_and_committed_receipt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = REPOSITORY_ROOT / "data" / "classifier" / "source-collection.v1.json"
    assert main(["schema", "collection", "--check"]) == 0
    assert (
        main(["validate-collection", str(receipt), "--repository-root", str(REPOSITORY_ROOT)]) == 0
    )
    assert "Validated classifier source collection" in capsys.readouterr().out


def test_collection_cli_refuses_registry_overwrite_and_existing_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = write_test_registry(tmp_path)
    assert (
        main(
            [
                "collect-sources",
                str(registry),
                "--output",
                str(registry),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "must differ" in capsys.readouterr().err

    receipt = tmp_path / "receipt.json"
    receipt.write_text("preserve me", encoding="utf-8")
    assert (
        main(
            [
                "collect-sources",
                str(registry),
                "--output",
                str(receipt),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "already exists" in capsys.readouterr().err
    assert receipt.read_text(encoding="utf-8") == "preserve me"
