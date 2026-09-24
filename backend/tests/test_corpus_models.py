from typing import Any

import pytest
from pydantic import ValidationError

from deepshield_api.corpus.models import (
    CollectionStatus,
    SourceManifest,
    SourcePublisher,
    SourceRecord,
)


def source_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_id": "synthetic-uscis-guidance",
        "document_version": 1,
        "title": "Synthetic USCIS guidance fixture",
        "publisher": "uscis",
        "canonical_url": "https://www.uscis.gov/synthetic-test-fixture",
        "document_type": "guidance-page",
        "media_type": "text/html",
        "scope_tags": ["status", "employment"],
        "rights_notes": "Synthetic metadata fixture; not an approved corpus source.",
    }
    data.update(overrides)
    return data


def collected_source_data(**overrides: Any) -> dict[str, Any]:
    data = source_data(
        collection_status="collected",
        retrieved_at="2026-09-21T12:00:00Z",
        raw_path="data/raw/uscis/synthetic-test-fixture.html",
        sha256="a" * 64,
    )
    data.update(overrides)
    return data


def manifest_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "manifest_version": 1,
        "created_at": "2026-09-21T12:00:00Z",
        "sources": [source_data()],
    }
    data.update(overrides)
    return data


def test_planned_source_is_normalized() -> None:
    record = SourceRecord.model_validate(source_data())

    assert record.publisher is SourcePublisher.USCIS
    assert record.collection_status is CollectionStatus.PLANNED
    assert record.scope_tags == ["employment", "status"]


def test_collected_source_requires_complete_artifacts() -> None:
    record = SourceRecord.model_validate(collected_source_data())

    assert record.collection_status is CollectionStatus.COLLECTED
    assert record.raw_path == "data/raw/uscis/synthetic-test-fixture.html"
    assert record.retrieved_at is not None
    assert record.retrieved_at.utcoffset() is not None


@pytest.mark.parametrize(
    ("canonical_url", "message"),
    [
        ("http://www.uscis.gov/example", "must use HTTPS"),
        ("https://user:secret@www.uscis.gov/example", "must not contain credentials"),
        ("https://www.uscis.gov:8443/example", "standard HTTPS port"),
        ("https://www.uscis.gov/example#section", "must not contain a URL fragment"),
        ("https://uscis.gov.example.com/example", "host is not approved"),
    ],
)
def test_source_rejects_unsafe_or_unapproved_urls(
    canonical_url: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SourceRecord.model_validate(source_data(canonical_url=canonical_url))


def test_source_rejects_duplicate_scope_tags() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        SourceRecord.model_validate(source_data(scope_tags=["status", "status"]))


def test_source_rejects_unsafe_raw_path() -> None:
    with pytest.raises(ValidationError, match="inside data/raw"):
        SourceRecord.model_validate(collected_source_data(raw_path="../private.txt"))


def test_source_rejects_naive_retrieval_timestamp() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        SourceRecord.model_validate(collected_source_data(retrieved_at="2026-09-21T12:00:00"))


def test_planned_source_rejects_collection_artifacts() -> None:
    with pytest.raises(ValidationError, match="cannot include collection artifacts"):
        SourceRecord.model_validate(
            source_data(raw_path="data/raw/uscis/synthetic-test-fixture.html")
        )


def test_collected_source_rejects_missing_artifacts() -> None:
    with pytest.raises(ValidationError, match="collected sources require"):
        SourceRecord.model_validate(source_data(collection_status="collected"))


def test_source_rejects_invalid_date_order() -> None:
    with pytest.raises(ValidationError, match="cannot be earlier"):
        SourceRecord.model_validate(
            source_data(published_on="2026-09-21", last_updated_on="2026-09-20")
        )


def test_source_rejects_invalid_checksum() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        SourceRecord.model_validate(collected_source_data(sha256="not-a-checksum"))


def test_source_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceRecord.model_validate(source_data(unreviewed=True))


def test_manifest_accepts_unique_source_versions() -> None:
    manifest = SourceManifest.model_validate(manifest_data())

    assert manifest.manifest_version == 1
    assert len(manifest.sources) == 1


def test_manifest_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        SourceManifest.model_validate(manifest_data(created_at="2026-09-21T12:00:00"))


def test_manifest_rejects_duplicate_source_versions() -> None:
    duplicated_source = source_data()
    with pytest.raises(ValidationError, match="duplicate source version"):
        SourceManifest.model_validate(manifest_data(sources=[duplicated_source, duplicated_source]))
