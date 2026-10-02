"""Command-line validation for evaluation benchmark artifacts."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepshield_api.evaluation.manifest import (
    BenchmarkManifestError,
    load_benchmark_manifest,
    render_benchmark_manifest_schema,
    validate_corpus_references,
    write_benchmark_manifest_schema,
)
from deepshield_api.evaluation.models import BenchmarkStatus

DEFAULT_SCHEMA_PATH = Path("data/evaluation/benchmark-manifest.schema.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""

    parser = argparse.ArgumentParser(description="Manage DeepShield-RAG benchmark manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Export the benchmark schema.")
    schema_parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH)
    schema_parser.add_argument("--check", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a benchmark manifest.")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    validate_parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Reject draft benchmarks that are not ready for final experiments.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an evaluation benchmark command."""

    args = build_parser().parse_args(argv)
    if args.command == "schema":
        output = Path(args.output)
        expected = render_benchmark_manifest_schema()
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != expected:
                print(f"Schema is missing or out of date: {output}", file=sys.stderr)
                return 1
            print(f"Schema is current: {output}")
            return 0
        write_benchmark_manifest_schema(output)
        print(f"Wrote schema: {output}")
        return 0

    try:
        manifest = load_benchmark_manifest(Path(args.manifest))
        validate_corpus_references(manifest, Path(args.repository_root))
    except BenchmarkManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.require_frozen and manifest.status is not BenchmarkStatus.FROZEN:
        print("benchmark must be frozen before final experiments", file=sys.stderr)
        return 1
    print(
        f"Validated {manifest.status.value} benchmark version {manifest.manifest_version} "
        f"with {len(manifest.cases)} case(s): {args.manifest}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
