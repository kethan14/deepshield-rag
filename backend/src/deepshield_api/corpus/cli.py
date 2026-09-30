"""Command-line tools for corpus manifest validation and schema export."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepshield_api.corpus.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_MIN_CHUNK_CHARACTERS,
    DEFAULT_OVERLAP_CHARACTERS,
    ChunkingError,
    ChunkingPolicy,
    chunk_extraction_report,
    write_chunk_dataset,
    write_chunking_report,
)
from deepshield_api.corpus.collection import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    Clock,
    CollectionError,
    CollectionPolicy,
    SourceFetcher,
    collect_manifest_source,
    download_source,
    import_manifest_source_file,
    utc_now,
)
from deepshield_api.corpus.extraction import (
    DEFAULT_MIN_CHARACTERS,
    ExtractionError,
    ExtractionPolicy,
    extract_manifest,
    write_extraction_report,
)
from deepshield_api.corpus.manifest import (
    ManifestError,
    load_source_manifest,
    render_source_manifest_schema,
    write_source_manifest,
    write_source_manifest_schema,
)

DEFAULT_SCHEMA_PATH = Path("data/manifests/source-manifest.schema.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the corpus command-line parser."""

    parser = argparse.ArgumentParser(description="Manage DeepShield-RAG corpus manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Export the source-manifest schema.")
    schema_parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH)
    schema_parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the exported schema is missing or out of date.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a source manifest.")
    validate_parser.add_argument("manifest", type=Path)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect one approved manifest source into an immutable raw artifact.",
    )
    collect_parser.add_argument("manifest", type=Path)
    collect_parser.add_argument("source_id")
    collect_parser.add_argument("--document-version", type=int, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    collect_parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    collect_parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)

    import_parser = subparsers.add_parser(
        "import-file",
        help="Validate and register a browser-downloaded source file.",
    )
    import_parser.add_argument("manifest", type=Path)
    import_parser.add_argument("source_id")
    import_parser.add_argument("--document-version", type=int, required=True)
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    import_parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Verify and extract every collected source into normalized interim text.",
    )
    extract_parser.add_argument("manifest", type=Path)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--report-version", type=int, default=1)
    extract_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    extract_parser.add_argument(
        "--min-characters",
        type=int,
        default=DEFAULT_MIN_CHARACTERS,
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Build deterministic retrieval chunks from a verified extraction report.",
    )
    chunk_parser.add_argument("extraction_report", type=Path)
    chunk_parser.add_argument("--output-chunks", type=Path, required=True)
    chunk_parser.add_argument("--output-report", type=Path, required=True)
    chunk_parser.add_argument("--report-version", type=int, default=1)
    chunk_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    chunk_parser.add_argument("--max-characters", type=int, default=DEFAULT_MAX_CHARACTERS)
    chunk_parser.add_argument(
        "--overlap-characters",
        type=int,
        default=DEFAULT_OVERLAP_CHARACTERS,
    )
    chunk_parser.add_argument(
        "--min-characters",
        type=int,
        default=DEFAULT_MIN_CHUNK_CHARACTERS,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fetcher: SourceFetcher = download_source,
    clock: Clock = utc_now,
) -> int:
    """Run a corpus manifest command."""

    args = build_parser().parse_args(argv)

    if args.command == "schema":
        output = Path(args.output)
        expected = render_source_manifest_schema()
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != expected:
                print(f"Schema is missing or out of date: {output}", file=sys.stderr)
                return 1
            print(f"Schema is current: {output}")
            return 0

        write_source_manifest_schema(output)
        print(f"Wrote schema: {output}")
        return 0

    if args.command == "chunk":
        chunks_path = Path(args.output_chunks)
        report_path = Path(args.output_report)
        if chunks_path.resolve() == report_path.resolve():
            print("chunk dataset and report outputs must differ", file=sys.stderr)
            return 1
        if report_path.exists():
            print(
                f"refusing to replace an existing chunking report: {report_path}", file=sys.stderr
            )
            return 1
        try:
            chunks, chunking_report = chunk_extraction_report(
                Path(args.extraction_report),
                chunks_path,
                Path(args.repository_root),
                report_version=args.report_version,
                policy=ChunkingPolicy(
                    max_characters=args.max_characters,
                    overlap_characters=args.overlap_characters,
                    min_characters=args.min_characters,
                ),
            )
            write_chunk_dataset(chunks_path, chunks)
            write_chunking_report(report_path, chunking_report)
        except (ChunkingError, ManifestError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(
            f"Chunked {chunking_report.total_sources} source(s) into "
            f"{chunking_report.total_chunks} chunks; wrote report version "
            f"{chunking_report.report_version}: {report_path}"
        )
        return 0

    manifest_path = Path(args.manifest)
    try:
        manifest = load_source_manifest(manifest_path)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "extract":
        output_path = Path(args.output)
        try:
            report = extract_manifest(
                manifest,
                manifest_path,
                Path(args.repository_root),
                report_version=args.report_version,
                policy=ExtractionPolicy(min_characters=args.min_characters),
            )
            write_extraction_report(output_path, report)
        except (ExtractionError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        characters = sum(record.character_count for record in report.records)
        print(
            f"Extracted {len(report.records)} source(s) with {characters} characters; "
            f"wrote report version {report.report_version}: {output_path}"
        )
        return 0

    if args.command in {"collect", "import-file"}:
        output_path = Path(args.output)
        if output_path.resolve() == manifest_path.resolve():
            print("collection output must differ from the input manifest", file=sys.stderr)
            return 1
        try:
            if args.command == "collect":
                policy = CollectionPolicy(
                    max_bytes=args.max_bytes,
                    timeout_seconds=args.timeout_seconds,
                )
                collected_manifest = collect_manifest_source(
                    manifest,
                    args.source_id,
                    args.document_version,
                    Path(args.repository_root),
                    policy=policy,
                    fetcher=fetcher,
                    clock=clock,
                )
            else:
                policy = CollectionPolicy(max_bytes=args.max_bytes)
                collected_manifest = import_manifest_source_file(
                    manifest,
                    args.source_id,
                    args.document_version,
                    Path(args.input),
                    Path(args.repository_root),
                    policy=policy,
                    clock=clock,
                )
            write_source_manifest(output_path, collected_manifest)
        except (CollectionError, ManifestError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(
            f"Registered {args.source_id} v{args.document_version}; "
            f"wrote manifest version {collected_manifest.manifest_version}: {output_path}"
        )
        return 0

    print(
        f"Validated manifest version {manifest.manifest_version} "
        f"with {len(manifest.sources)} source(s): {manifest_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
