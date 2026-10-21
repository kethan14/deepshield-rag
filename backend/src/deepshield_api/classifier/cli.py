"""Command-line validation for C2 classifier-data artifacts."""

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Literal

from deepshield_api.classifier.collection import (
    ClassifierCollectionError,
    collect_registered_sources,
    validate_collection_receipt,
)
from deepshield_api.classifier.manifest import (
    ClassifierManifestError,
    load_collection_receipt,
    load_preprocessing_report,
    load_risk_dataset,
    load_source_registry,
    render_collection_receipt,
    render_collection_receipt_schema,
    render_risk_dataset_schema,
    render_source_registry_schema,
    validate_dataset_references,
)
from deepshield_api.classifier.models import ArtifactStatus
from deepshield_api.classifier.preprocessing import (
    ClassifierPreprocessingError,
    preprocess_classifier_sources,
    render_preprocessing_report_schema,
    validate_preprocessing_report,
    write_preprocessing_result,
)
from deepshield_api.classifier.review import (
    DEFAULT_REVIEW_WORKSHEET,
    ClassifierReviewError,
    audit_classifier_review_worksheet,
    load_selected_dataset_for_review,
    prepare_classifier_review_worksheet,
    write_classifier_review_worksheet,
)
from deepshield_api.classifier.selection import (
    ClassifierSelectionError,
    load_selection_report,
    render_selection_report_schema,
    select_classifier_review_candidates,
    validate_selection_report,
    write_selection_result,
)

SCHEMA_PATHS = {
    "collection": Path("data/classifier/source-collection.schema.json"),
    "sources": Path("data/classifier/source-registry.schema.json"),
    "dataset": Path("data/classifier/risk-dataset.schema.json"),
    "preprocessing": Path("data/classifier/preprocessing-report.schema.json"),
    "selection": Path("data/classifier/selection-report.schema.json"),
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

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Normalize, deduplicate, group, and split collected classifier sources.",
    )
    preprocess.add_argument("collection", type=Path)
    preprocess.add_argument(
        "--output",
        type=Path,
        default=Path("data/classifier/processed/risk-dataset.v1.json"),
    )
    preprocess.add_argument(
        "--report",
        type=Path,
        default=Path("data/classifier/preprocessing-report.v1.json"),
    )
    preprocess.add_argument("--repository-root", type=Path, default=Path.cwd())

    preprocessing_report = subparsers.add_parser(
        "validate-preprocessing",
        help="Validate a classifier preprocessing report and its checksums.",
    )
    preprocessing_report.add_argument("report", type=Path)
    preprocessing_report.add_argument("--repository-root", type=Path, default=Path.cwd())
    preprocessing_report.add_argument("--require-dataset", action="store_true")

    select = subparsers.add_parser(
        "select-review-candidates",
        help="Exclude benchmark overlap and create the balanced classifier review set.",
    )
    select.add_argument(
        "--report",
        type=Path,
        default=Path("data/classifier/selection-report.v1.json"),
    )
    select.add_argument("--repository-root", type=Path, default=Path.cwd())

    selection_report = subparsers.add_parser(
        "validate-selection",
        help="Validate benchmark-exclusion and classifier-selection evidence.",
    )
    selection_report.add_argument("report", type=Path)
    selection_report.add_argument("--repository-root", type=Path, default=Path.cwd())
    selection_report.add_argument("--require-artifacts", action="store_true")

    prepare_review = subparsers.add_parser(
        "prepare-review",
        help="Create a separate editable human-review worksheet.",
    )
    prepare_review.add_argument("--repository-root", type=Path, default=Path.cwd())
    prepare_review.add_argument("--output", type=Path, default=DEFAULT_REVIEW_WORKSHEET)

    audit_review = subparsers.add_parser(
        "audit-review",
        help="Validate reviewer decisions and report completion progress.",
    )
    audit_review.add_argument("worksheet", type=Path, nargs="?", default=DEFAULT_REVIEW_WORKSHEET)
    audit_review.add_argument("--repository-root", type=Path, default=Path.cwd())
    audit_review.add_argument("--require-complete", action="store_true")

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
        if args.command == "preprocess":
            output_path = PurePosixPath(args.output.as_posix())
            preprocessing_result = preprocess_classifier_sources(
                args.collection,
                args.repository_root,
                output_dataset_path=output_path,
            )
            write_preprocessing_result(preprocessing_result, args.repository_root, args.report)
            print(
                f"Preprocessed {len(preprocessing_result.dataset.examples)} classifier candidates "
                "into "
                f"{args.output}; audit report: {args.report}"
            )
            return 0
        if args.command == "validate-preprocessing":
            report = load_preprocessing_report(args.report)
            validate_preprocessing_report(
                report,
                args.repository_root,
                require_dataset=args.require_dataset,
            )
            print(
                f"Validated classifier preprocessing report version {report.report_version} "
                f"for {sum(source.output_rows for source in report.sources)} candidates."
            )
            return 0
        if args.command == "select-review-candidates":
            selection_result = select_classifier_review_candidates(args.repository_root)
            selection_report_path = args.report
            if not selection_report_path.is_absolute():
                selection_report_path = args.repository_root / selection_report_path
            write_selection_result(
                selection_result,
                args.repository_root,
                selection_report_path,
            )
            print(
                f"Selected {len(selection_result.dataset.examples)} benchmark-safe classifier "
                f"candidates; review packet: {selection_result.report.review_packet_path}"
            )
            return 0
        if args.command == "validate-selection":
            selection_report_value = load_selection_report(args.report)
            validate_selection_report(
                selection_report_value,
                args.repository_root,
                require_artifacts=args.require_artifacts,
            )
            selected_count = sum(
                summary.selected_count for summary in selection_report_value.splits
            )
            print(
                "Validated classifier selection report version "
                f"{selection_report_value.report_version} "
                f"for {selected_count} review candidates."
            )
            return 0
        if args.command == "prepare-review":
            output_path = args.output
            if not output_path.is_absolute():
                output_path = args.repository_root / output_path
            expected_path = (args.repository_root / DEFAULT_REVIEW_WORKSHEET).resolve()
            if output_path.resolve() != expected_path:
                print(
                    f"review worksheet must use {DEFAULT_REVIEW_WORKSHEET.as_posix()}",
                    file=sys.stderr,
                )
                return 2
            content = prepare_classifier_review_worksheet(args.repository_root)
            write_classifier_review_worksheet(output_path, content)
            print(f"Prepared classifier review worksheet: {output_path}")
            return 0
        if args.command == "audit-review":
            worksheet_path = args.worksheet
            if not worksheet_path.is_absolute():
                worksheet_path = args.repository_root / worksheet_path
            selected_dataset = load_selected_dataset_for_review(args.repository_root)
            progress = audit_classifier_review_worksheet(
                worksheet_path,
                selected_dataset,
                require_complete=args.require_complete,
            )
            print(
                f"Classifier review: {progress.total} total, {progress.pending} pending, "
                f"{progress.approved} approved, {progress.corrected} corrected, "
                f"{progress.excluded} excluded."
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
    except (
        ClassifierCollectionError,
        ClassifierManifestError,
        ClassifierPreprocessingError,
        ClassifierReviewError,
        ClassifierSelectionError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _handle_schema(
    kind: Literal["collection", "dataset", "preprocessing", "selection", "sources"],
    output: Path | None,
    check: bool,
) -> int:
    renderers = {
        "collection": render_collection_receipt_schema,
        "dataset": render_risk_dataset_schema,
        "preprocessing": render_preprocessing_report_schema,
        "selection": render_selection_report_schema,
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
