import csv
import io
from collections import Counter
from pathlib import Path

import pytest

from deepshield_api.evaluation.audit import audit_benchmark, render_review_csv
from deepshield_api.evaluation.cli import main
from deepshield_api.evaluation.generation import build_draft_benchmark
from deepshield_api.evaluation.manifest import (
    render_benchmark_manifest,
    validate_corpus_references,
)
from deepshield_api.evaluation.models import AttackCategory, BenchmarkKind
from deepshield_api.evaluation.synthetic import (
    SyntheticEnvironmentError,
    build_synthetic_environment,
    load_synthetic_environment,
    render_synthetic_environment,
    write_synthetic_environment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPOSITORY_ROOT / "data/evaluation/f1-security-benchmark.v1.json"
ENVIRONMENT_PATH = REPOSITORY_ROOT / "data/evaluation/synthetic-safety-environment.v1.json"
AUDIT_PATH = REPOSITORY_ROOT / "data/evaluation/f1-security-benchmark-audit.v1.json"
REVIEW_PATH = REPOSITORY_ROOT / "data/evaluation/f1-security-benchmark-review.v1.csv"


def test_synthetic_environment_is_safe_deterministic_and_non_operational() -> None:
    first = build_synthetic_environment()
    second = build_synthetic_environment()

    assert render_synthetic_environment(first) == render_synthetic_environment(second)
    assert len(first.records) == 24
    assert len(first.tools) == 4
    assert all(record.fictional for record in first.records)
    assert all(record.email.endswith("@example.test") for record in first.records)
    assert all(not tool.executable for tool in first.tools)


def test_synthetic_environment_round_trip_refuses_overwrite(tmp_path: Path) -> None:
    environment = build_synthetic_environment()
    output = tmp_path / "nested" / "environment.json"

    write_synthetic_environment(output, environment)

    assert load_synthetic_environment(output) == environment
    with pytest.raises(SyntheticEnvironmentError, match="refusing to replace"):
        write_synthetic_environment(output, environment)


def test_generated_benchmark_has_frozen_design_counts_and_valid_references() -> None:
    environment = build_synthetic_environment()
    manifest = build_draft_benchmark(REPOSITORY_ROOT, environment)

    validate_corpus_references(manifest, REPOSITORY_ROOT)
    report = audit_benchmark(manifest, environment)
    split_kind_counts = Counter((case.split.value, case.kind.value) for case in manifest.cases)
    category_counts = Counter(
        case.attack_category for case in manifest.cases if case.kind is BenchmarkKind.ADVERSARIAL
    )

    assert len(manifest.cases) == 600
    assert split_kind_counts == {
        ("development", "benign"): 100,
        ("development", "adversarial"): 100,
        ("test", "benign"): 200,
        ("test", "adversarial"): 200,
    }
    assert all(category_counts[category] == 50 for category in AttackCategory)
    assert report.passed
    assert report.issue_count == 0
    assert report.maximum_cross_split_similarity < report.near_duplicate_threshold


def test_committed_generated_artifacts_are_current() -> None:
    environment = build_synthetic_environment()
    manifest = build_draft_benchmark(REPOSITORY_ROOT, environment)

    assert BENCHMARK_PATH.read_text(encoding="utf-8") == render_benchmark_manifest(manifest)
    assert ENVIRONMENT_PATH.read_text(encoding="utf-8") == render_synthetic_environment(environment)
    assert main(["generate", "--repository-root", str(REPOSITORY_ROOT), "--check"]) == 0
    assert (
        main(
            [
                "audit",
                str(BENCHMARK_PATH),
                "--environment",
                str(ENVIRONMENT_PATH),
                "--output",
                str(AUDIT_PATH),
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--check",
            ]
        )
        == 0
    )


def test_review_packet_contains_all_cases_without_secret_values() -> None:
    environment = build_synthetic_environment()
    manifest = build_draft_benchmark(REPOSITORY_ROOT, environment)
    content = render_review_csv(manifest)
    rows = list(csv.DictReader(io.StringIO(content)))

    assert len(rows) == 600
    assert REVIEW_PATH.read_text(encoding="utf-8") == content
    assert all(row["review_decision"] == "" for row in rows)
    assert all(record.canary_token not in content for record in environment.records)


def test_audit_detects_a_canary_assigned_to_the_wrong_fixture() -> None:
    environment = build_synthetic_environment()
    manifest = build_draft_benchmark(REPOSITORY_ROOT, environment)
    cases = list(manifest.cases)
    index = next(
        index
        for index, case in enumerate(cases)
        if case.attack_category is AttackCategory.SYNTHETIC_SECRET_EXFILTRATION
    )
    cases[index] = cases[index].model_copy(
        update={"canary_tokens": [environment.records[1].canary_token]}
    )
    tampered = manifest.model_copy(update={"cases": cases})

    report = audit_benchmark(tampered, environment)

    assert not report.passed
    assert any(issue.code == "incorrect-canary-label" for issue in report.issues)
