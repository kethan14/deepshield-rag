"""Command-line tools for corpus manifest validation and schema export."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepshield_api.corpus.manifest import (
    ManifestError,
    load_source_manifest,
    render_source_manifest_schema,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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

    manifest_path = Path(args.manifest)
    try:
        manifest = load_source_manifest(manifest_path)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Validated manifest version {manifest.manifest_version} "
        f"with {len(manifest.sources)} source(s): {manifest_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
