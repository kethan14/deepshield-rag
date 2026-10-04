"""Command-line validation for C2 classifier-data artifacts."""

import argparse
import sys
from pathlib import Path
from typing import Literal

from deepshield_api.classifier.manifest import (
    ClassifierManifestError,
    load_risk_dataset,
    load_source_registry,
    render_risk_dataset_schema,
    render_source_registry_schema,
    validate_dataset_references,
)
from deepshield_api.classifier.models import ArtifactStatus

SCHEMA_PATHS = {
    "sources": Path("data/classifier/source-registry.schema.json"),
    "dataset": Path("data/classifier/risk-dataset.schema.json"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage C2 classifier-data contracts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema", help="Export or check a classifier JSON Schema.")
    schema.add_argument("kind", choices=sorted(SCHEMA_PATHS))
    schema.add_argument("--output", type=Path)
    schema.add_argument("--check", action="store_true")

    sources = subparsers.add_parser("validate-sources", help="Validate a source registry.")
    sources.add_argument("registry", type=Path)

    dataset = subparsers.add_parser("validate-dataset", help="Validate a classifier dataset.")
    dataset.add_argument("dataset", type=Path)
    dataset.add_argument("--repository-root", type=Path, default=Path.cwd())
    dataset.add_argument("--require-frozen", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "schema":
            return _handle_schema(args.kind, args.output, args.check)
        if args.command == "validate-sources":
            registry = load_source_registry(args.registry)
            print(
                f"Validated {registry.status.value} source registry version "
                f"{registry.registry_version} with {len(registry.sources)} sources."
            )
            return 0
        dataset = load_risk_dataset(args.dataset)
        validate_dataset_references(dataset, args.repository_root)
        if args.require_frozen and dataset.status is not ArtifactStatus.FROZEN:
            print("classifier dataset must be frozen before training", file=sys.stderr)
            return 2
        print(
            f"Validated {dataset.status.value} classifier dataset version "
            f"{dataset.dataset_version} with {len(dataset.examples)} examples."
        )
        return 0
    except ClassifierManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _handle_schema(kind: Literal["dataset", "sources"], output: Path | None, check: bool) -> int:
    rendered = (
        render_source_registry_schema() if kind == "sources" else render_risk_dataset_schema()
    )
    target = output or SCHEMA_PATHS[kind]
    if check:
        try:
            current = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ClassifierManifestError(f"unable to read schema: {target}") from exc
        if current != rendered:
            print(f"schema drift detected: {target}", file=sys.stderr)
            return 2
        print(f"Schema is current: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote schema: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
