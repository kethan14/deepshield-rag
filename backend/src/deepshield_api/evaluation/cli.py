"""Command-line validation for evaluation benchmark artifacts."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepshield_api.evaluation.audit import (
    audit_benchmark,
    render_audit_report,
    render_review_csv,
    write_text_artifact,
)
from deepshield_api.evaluation.generation import build_draft_benchmark
from deepshield_api.evaluation.manifest import (
    BenchmarkManifestError,
    load_benchmark_manifest,
    render_benchmark_manifest,
    render_benchmark_manifest_schema,
    validate_corpus_references,
    write_benchmark_manifest,
    write_benchmark_manifest_schema,
)
from deepshield_api.evaluation.models import BenchmarkStatus
from deepshield_api.evaluation.runs import (
    ExperimentPurpose,
    ExperimentRunError,
    execute_frozen_experiment,
    prepare_experiment_inputs,
    validate_experiment_run_id,
    write_completed_run,
)
from deepshield_api.evaluation.scoring import (
    ExperimentScoringError,
    render_score_report_schema,
    score_completed_experiment,
    write_completed_score,
    write_score_report_schema,
)
from deepshield_api.evaluation.synthetic import (
    SyntheticEnvironmentError,
    build_synthetic_environment,
    load_synthetic_environment,
    render_synthetic_environment,
    write_synthetic_environment,
)
from deepshield_api.rag.runtime import (
    RagRuntimeError,
    RagSettings,
    build_experiment_comparison_backends,
)

DEFAULT_SCHEMA_PATH = Path("data/evaluation/benchmark-manifest.schema.json")
DEFAULT_BENCHMARK_PATH = Path("data/evaluation/f1-security-benchmark.v1.json")
DEFAULT_ENVIRONMENT_PATH = Path("data/evaluation/synthetic-safety-environment.v1.json")
DEFAULT_AUDIT_PATH = Path("data/evaluation/f1-security-benchmark-audit.v1.json")
DEFAULT_REVIEW_PATH = Path("data/evaluation/f1-security-benchmark-review.v1.csv")
DEFAULT_GENERATION_CONFIGURATION_PATH = Path("config/experiments/generation-model.v1.json")
DEFAULT_RUNS_ROOT = Path("data/evaluation/runs")
DEFAULT_SCORE_SCHEMA_PATH = Path("data/evaluation/score-report.schema.json")


def _resolve_from_root(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""

    parser = argparse.ArgumentParser(description="Manage DeepShield-RAG benchmark manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Export the benchmark schema.")
    schema_parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH)
    schema_parser.add_argument("--check", action="store_true")

    score_schema_parser = subparsers.add_parser(
        "score-schema", help="Export the experiment scoring-report schema."
    )
    score_schema_parser.add_argument("--output", type=Path, default=DEFAULT_SCORE_SCHEMA_PATH)
    score_schema_parser.add_argument("--check", action="store_true")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate the deterministic draft benchmark and safety fixtures."
    )
    generate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    generate_parser.add_argument("--benchmark-output", type=Path, default=DEFAULT_BENCHMARK_PATH)
    generate_parser.add_argument(
        "--environment-output", type=Path, default=DEFAULT_ENVIRONMENT_PATH
    )
    generate_parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    generate_parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_PATH)
    generate_parser.add_argument("--check", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit", help="Audit a generated benchmark and its synthetic environment."
    )
    audit_parser.add_argument("benchmark", type=Path, nargs="?", default=DEFAULT_BENCHMARK_PATH)
    audit_parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT_PATH)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_PATH)
    audit_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    audit_parser.add_argument("--check", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a benchmark manifest.")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    validate_parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Reject draft benchmarks that are not ready for final experiments.",
    )

    run_parser = subparsers.add_parser(
        "run", help="Execute one frozen benchmark split through C0, C1, and C2."
    )
    run_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    run_parser.add_argument(
        "--generation-configuration",
        type=Path,
        default=DEFAULT_GENERATION_CONFIGURATION_PATH,
    )
    run_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    run_parser.add_argument(
        "--purpose", choices=[item.value for item in ExperimentPurpose], required=True
    )
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Acknowledge that the configured generator may incur provider charges.",
    )
    run_parser.add_argument(
        "--confirm-held-out-test",
        action="store_true",
        help="Explicitly authorize the one-time held-out final evaluation.",
    )

    score_parser = subparsers.add_parser(
        "score", help="Score one completed run without making model calls."
    )
    score_parser.add_argument("run_directory", type=Path)
    score_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    score_parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    score_parser.add_argument(
        "--generation-configuration",
        type=Path,
        default=DEFAULT_GENERATION_CONFIGURATION_PATH,
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

    if args.command == "score-schema":
        output = Path(args.output)
        expected = render_score_report_schema()
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != expected:
                print(f"Score schema is missing or out of date: {output}", file=sys.stderr)
                return 1
            print(f"Score schema is current: {output}")
            return 0
        write_score_report_schema(output)
        print(f"Wrote score schema: {output}")
        return 0

    if args.command == "generate":
        repository_root = Path(args.repository_root).resolve()
        try:
            environment = build_synthetic_environment()
            manifest = build_draft_benchmark(repository_root, environment)
            validate_corpus_references(manifest, repository_root)
            audit = audit_benchmark(manifest, environment)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not audit.passed:
            print(
                f"generated benchmark failed audit with {audit.issue_count} issue(s)",
                file=sys.stderr,
            )
            return 1
        artifacts = {
            _resolve_from_root(Path(args.benchmark_output), repository_root): (
                render_benchmark_manifest(manifest)
            ),
            _resolve_from_root(Path(args.environment_output), repository_root): (
                render_synthetic_environment(environment)
            ),
            _resolve_from_root(Path(args.audit_output), repository_root): (
                render_audit_report(audit)
            ),
            _resolve_from_root(Path(args.review_output), repository_root): (
                render_review_csv(manifest)
            ),
        }
        if args.check:
            stale = [
                path
                for path, expected in artifacts.items()
                if not path.is_file() or path.read_text(encoding="utf-8") != expected
            ]
            if stale:
                print(
                    "Generated evaluation artifacts are missing or out of date: "
                    + ", ".join(str(path) for path in stale),
                    file=sys.stderr,
                )
                return 1
            print("Generated evaluation artifacts are current.")
            return 0
        existing = [path for path in artifacts if path.exists()]
        if existing:
            print(
                "Refusing to replace existing generated artifacts: "
                + ", ".join(str(path) for path in existing),
                file=sys.stderr,
            )
            return 1
        try:
            write_benchmark_manifest(
                _resolve_from_root(Path(args.benchmark_output), repository_root), manifest
            )
            write_synthetic_environment(
                _resolve_from_root(Path(args.environment_output), repository_root), environment
            )
            write_text_artifact(
                _resolve_from_root(Path(args.audit_output), repository_root),
                render_audit_report(audit),
            )
            write_text_artifact(
                _resolve_from_root(Path(args.review_output), repository_root),
                render_review_csv(manifest),
            )
        except (BenchmarkManifestError, SyntheticEnvironmentError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Generated {len(manifest.cases)} draft cases, {len(environment.records)} "
            f"fictional records, and {len(environment.tools)} inert mock tools."
        )
        return 0

    if args.command == "audit":
        repository_root = Path(args.repository_root).resolve()
        benchmark_path = _resolve_from_root(Path(args.benchmark), repository_root)
        environment_path = _resolve_from_root(Path(args.environment), repository_root)
        try:
            manifest = load_benchmark_manifest(benchmark_path)
            environment = load_synthetic_environment(environment_path)
            validate_corpus_references(manifest, repository_root)
        except (BenchmarkManifestError, SyntheticEnvironmentError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        report = audit_benchmark(manifest, environment)
        expected = render_audit_report(report)
        output = _resolve_from_root(Path(args.output), repository_root)
        if args.check and (not output.is_file() or output.read_text(encoding="utf-8") != expected):
            print(f"Benchmark audit report is missing or out of date: {output}", file=sys.stderr)
            return 1
        if not args.check:
            if output.exists():
                output.write_text(expected, encoding="utf-8", newline="\n")
            else:
                write_text_artifact(output, expected)
        if not report.passed:
            print(f"Benchmark audit failed with {report.issue_count} issue(s).", file=sys.stderr)
            return 1
        print(
            f"Benchmark audit passed for {report.total_cases} cases; maximum cross-split "
            f"similarity {report.maximum_cross_split_similarity:.4f}."
        )
        return 0

    if args.command == "run":
        repository_root = Path(args.repository_root).resolve()
        purpose = ExperimentPurpose(args.purpose)
        benchmark_path = Path(args.benchmark)
        generation_path = Path(args.generation_configuration)
        runs_root = _resolve_from_root(Path(args.runs_root), repository_root)
        try:
            validate_experiment_run_id(args.run_id)
            prepared = prepare_experiment_inputs(
                repository_root=repository_root,
                benchmark_path=benchmark_path,
                generation_configuration_path=generation_path,
                purpose=purpose,
            )
            if not args.confirm_paid_run:
                raise ExperimentRunError(
                    "refusing to execute without --confirm-paid-run; preflight made no model calls"
                )
            if purpose is ExperimentPurpose.FINAL and not args.confirm_held_out_test:
                raise ExperimentRunError(
                    "final evaluation requires --confirm-held-out-test to protect the held-out split"
                )
            if (runs_root / args.run_id).exists():
                raise ExperimentRunError(
                    f"refusing to replace existing experiment run: {runs_root / args.run_id}"
                )
            settings = RagSettings(
                repository_root=repository_root,
                llm_base_url=str(prepared.generation.base_url),
                llm_model=prepared.generation.model,
                llm_max_output_tokens=prepared.generation.max_completion_tokens,
                llm_reasoning_effort=prepared.generation.reasoning_effort,
                llm_max_token_parameter="max_completion_tokens",
            )
            if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
                raise ExperimentRunError(
                    "DEEPSHIELD_LLM_API_KEY is required for the frozen hosted-model experiment"
                )
            backends = build_experiment_comparison_backends(settings)
            completed = execute_frozen_experiment(
                repository_root=repository_root,
                benchmark_path=benchmark_path,
                generation_configuration_path=generation_path,
                purpose=purpose,
                run_id=args.run_id,
                backends=backends,
            )
            run_directory = write_completed_run(runs_root, completed)
        except (ExperimentRunError, RagRuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Completed {purpose.value} experiment with {completed.manifest.case_count} cases "
            f"and {completed.manifest.trial_count} trials: {run_directory}"
        )
        return 0

    if args.command == "score":
        repository_root = Path(args.repository_root).resolve()
        run_directory = _resolve_from_root(Path(args.run_directory), repository_root)
        try:
            completed_score = score_completed_experiment(
                repository_root=repository_root,
                run_directory=run_directory,
                benchmark_path=Path(args.benchmark),
                generation_configuration_path=Path(args.generation_configuration),
            )
            analysis_directory = write_completed_score(run_directory, completed_score)
        except (ExperimentScoringError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Scored {len(completed_score.trial_scores)} trials with "
            f"{len(completed_score.report.paired_comparisons)} paired comparisons: "
            f"{analysis_directory}"
        )
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
