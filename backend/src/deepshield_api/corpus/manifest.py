"""Read, validate, and export source-manifest artifacts."""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from deepshield_api.corpus.models import SourceManifest


class ManifestError(ValueError):
    """Raised when a source manifest cannot be read or validated."""


def load_source_manifest(path: Path) -> SourceManifest:
    """Load and validate a UTF-8 JSON source manifest."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"unable to read source manifest: {path}") from exc

    try:
        payload = json.loads(content)
    except JSONDecodeError as exc:
        raise ManifestError(
            f"source manifest is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    try:
        return SourceManifest.model_validate(payload)
    except ValidationError as exc:
        raise ManifestError(f"source manifest validation failed:\n{exc}") from exc


def render_source_manifest_schema() -> str:
    """Return the canonical deterministic JSON Schema document."""

    return json.dumps(SourceManifest.model_json_schema(), indent=2, sort_keys=True) + "\n"


def write_source_manifest_schema(path: Path) -> None:
    """Write the canonical JSON Schema, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_source_manifest_schema(), encoding="utf-8", newline="\n")
