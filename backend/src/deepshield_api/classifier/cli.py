"""Command-line validation for C2 classifier-data artifacts."""

import argparse
import sys
from pathlib import Path
from typing import Literal

from deepshield_api.classifier.collection import (
    ClassifierCollectionError,
    collect_registered_sources,
    validate_collection_receipt,
)
from deepshield_api.classifier.manifest import (
    ClassifierManifestError,
    load_collection_receipt,
    load_risk_dataset,
    load_source_registry,
    render_collection_receipt,
    render_collection_receipt_schema,
    render_risk_dataset_schema,
    render_source_registry_schema,
    validate_dataset_references,
)
from deepshield_api.classifier.models import ArtifactStatus

SCHEMA_PATHS = {
    "collection": Path("data/classifier/source-collection.schema.json"),
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

    collect = subparsers.add_parser(
        "collect-sources",
        help="Collect every pinned source file and write a verified receipt.",
    )
    collect.add_argument("registry", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--repository-root", type=Path, default=Path.cwd())

    receipt = subparsers.add_parser(
        "validate-collection",
        help="Validate a classifier source-collection receipt.",
    )
    receipt.add_argument("receipt", type=Path)
    receipt.add_argument("--repository-root", type=Path, default=Path.cwd())
    receipt.add_argument("--require-files", action="store_true")

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
        if args.command == "collect-sources":
            if args.output.resolve() == args.registry.resolve():
                print("collection output must differ from the source registry", file=sys.stderr)
                return 2
            if args.output.exists():
                print(
                    "collection receipt already exists; use a new versioned output path",
                    file=sys.stderr,
                )
                return 2
            receipt = collect_registered_sources(args.registry, args.repository_root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                render_collection_receipt(receipt),
                encoding="utf-8",
                newline="\n",
            )
            print(
                f"Collected {len(receipt.files)} source files with "
                f"{sum(item.row_count for item in receipt.files)} rows: {args.output}"
            )
            return 0
        if args.command == "validate-collection":
            receipt = load_collection_receipt(args.receipt)
            validate_collection_receipt(
                receipt,
                args.repository_root,
                require_files=args.require_files,
            )
            print(
                f"Validated classifier source collection version {receipt.receipt_version} "
                f"with {len(receipt.files)} files."
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
    except (ClassifierCollectionError, ClassifierManifestError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _handle_schema(
    kind: Literal["collection", "dataset", "sources"], output: Path | None, check: bool
) -> int:
    renderers = {
        "collection": render_collection_receipt_schema,
        "dataset": render_risk_dataset_schema,
        "sources": render_source_registry_schema,
    }
    rendered = renderers[kind]()
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
