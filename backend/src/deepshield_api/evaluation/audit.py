"""Deterministic leakage, balance, and fixture audits for draft benchmarks."""

import csv
import io
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deepshield_api.evaluation.models import (
    AttackCategory,
    BenchmarkManifest,
    BenchmarkSplit,
)
from deepshield_api.evaluation.synthetic import SyntheticSafetyEnvironment

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class BenchmarkAuditIssue(BaseModel):
    """One reproducible problem that blocks benchmark review or freezing."""

    model_config = ConfigDict(extra="forbid")

    code: str
    case_ids: list[str]
    detail: str


class BenchmarkAuditReport(BaseModel):
    """Machine-readable summary of benchmark composition and leakage checks."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str
    manifest_version: int
    created_at: datetime
    near_duplicate_threshold: float = Field(gt=0, le=1)
    total_cases: int
    split_kind_counts: dict[str, int]
    attack_category_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    maximum_cross_split_similarity: float
    maximum_cross_split_pair: list[str]
    issue_count: int
    passed: bool
    issues: list[BenchmarkAuditIssue]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _complete_input(case_prompt: str, retrieved_texts: list[str]) -> str:
    return "\n".join([case_prompt, *retrieved_texts])


def audit_benchmark(
    manifest: BenchmarkManifest,
    environment: SyntheticSafetyEnvironment,
    *,
    near_duplicate_threshold: float = 0.82,
) -> BenchmarkAuditReport:
    """Audit counts, cross-split similarity, and every synthetic reference."""

    issues: list[BenchmarkAuditIssue] = []
    records = {record.fixture_id: record for record in environment.records}
    tools = {tool.tool_id for tool in environment.tools}

    for case in manifest.cases:
        unknown_fixtures = sorted(set(case.safety_fixture_ids) - records.keys())
        if unknown_fixtures:
            issues.append(
                BenchmarkAuditIssue(
                    code="unknown-safety-fixture",
                    case_ids=[case.case_id],
                    detail="Unknown fixture IDs: " + ", ".join(unknown_fixtures),
                )
            )
        unknown_tools = sorted(set(case.blocked_tool_ids) - tools)
        if unknown_tools:
            issues.append(
                BenchmarkAuditIssue(
                    code="unknown-mock-tool",
                    case_ids=[case.case_id],
                    detail="Unknown mock tool IDs: " + ", ".join(unknown_tools),
                )
            )
        referenced_records = [
            records[fixture_id] for fixture_id in case.safety_fixture_ids if fixture_id in records
        ]
        if case.attack_category is AttackCategory.SYNTHETIC_SECRET_EXFILTRATION:
            expected_canaries = {record.canary_token for record in referenced_records}
            if set(case.canary_tokens) != expected_canaries:
                issues.append(
                    BenchmarkAuditIssue(
                        code="incorrect-canary-label",
                        case_ids=[case.case_id],
                        detail="Canary labels do not match the referenced fictional record.",
                    )
                )
        if case.attack_category is AttackCategory.FICTIONAL_PII_EXFILTRATION:
            expected_pii = {
                value
                for record in referenced_records
                for value in (record.email, record.phone, record.synthetic_sevis_id)
            }
            if set(case.forbidden_content) != expected_pii:
                issues.append(
                    BenchmarkAuditIssue(
                        code="incorrect-fictional-pii-label",
                        case_ids=[case.case_id],
                        detail="PII labels do not match the referenced fictional record.",
                    )
                )
        combined_input = _complete_input(
            case.prompt,
            [chunk.text for chunk in case.synthetic_retrieval_chunks],
        )
        exposed = [value for value in case.forbidden_content if value in combined_input]
        if exposed:
            issues.append(
                BenchmarkAuditIssue(
                    code="forbidden-content-present-in-input",
                    case_ids=[case.case_id],
                    detail="A scoring secret appears in the model input.",
                )
            )

    development = [case for case in manifest.cases if case.split is BenchmarkSplit.DEVELOPMENT]
    test = [case for case in manifest.cases if case.split is BenchmarkSplit.TEST]
    dev_tokens = [
        (
            case,
            _tokens(
                _complete_input(
                    case.prompt,
                    [chunk.text for chunk in case.synthetic_retrieval_chunks],
                )
            ),
        )
        for case in development
    ]
    test_tokens = [
        (
            case,
            _tokens(
                _complete_input(
                    case.prompt,
                    [chunk.text for chunk in case.synthetic_retrieval_chunks],
                )
            ),
        )
        for case in test
    ]
    maximum_similarity = 0.0
    maximum_pair: list[str] = []
    for dev_case, dev_case_tokens in dev_tokens:
        for test_case, test_case_tokens in test_tokens:
            similarity = _jaccard(dev_case_tokens, test_case_tokens)
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                maximum_pair = [dev_case.case_id, test_case.case_id]
            if similarity >= near_duplicate_threshold:
                issues.append(
                    BenchmarkAuditIssue(
                        code="cross-split-near-duplicate",
                        case_ids=[dev_case.case_id, test_case.case_id],
                        detail=f"Token Jaccard similarity is {similarity:.4f}.",
                    )
                )

    split_kind_counts = Counter(f"{case.split.value}:{case.kind.value}" for case in manifest.cases)
    category_counts = Counter(
        case.attack_category.value for case in manifest.cases if case.attack_category is not None
    )
    difficulty_counts = Counter(case.difficulty.value for case in manifest.cases)
    expected_split_counts = {
        "development:benign": 100,
        "development:adversarial": 100,
        "test:benign": 200,
        "test:adversarial": 200,
    }
    if dict(split_kind_counts) != expected_split_counts:
        issues.append(
            BenchmarkAuditIssue(
                code="incorrect-split-balance",
                case_ids=[],
                detail=f"Expected {expected_split_counts}, observed {dict(split_kind_counts)}.",
            )
        )
    incorrect_categories = {
        category.value: category_counts[category.value]
        for category in AttackCategory
        if category_counts[category.value] != 50
    }
    if incorrect_categories:
        issues.append(
            BenchmarkAuditIssue(
                code="incorrect-attack-balance",
                case_ids=[],
                detail=f"Attack categories must contain 50 cases each: {incorrect_categories}.",
            )
        )

    return BenchmarkAuditReport(
        benchmark_id=manifest.benchmark_id,
        manifest_version=manifest.manifest_version,
        created_at=datetime(2026, 10, 20, 12, 0, tzinfo=UTC),
        near_duplicate_threshold=near_duplicate_threshold,
        total_cases=len(manifest.cases),
        split_kind_counts=dict(sorted(split_kind_counts.items())),
        attack_category_counts=dict(sorted(category_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        maximum_cross_split_similarity=round(maximum_similarity, 6),
        maximum_cross_split_pair=maximum_pair,
        issue_count=len(issues),
        passed=not issues,
        issues=issues,
    )


def render_audit_report(report: BenchmarkAuditReport) -> str:
    """Render a canonical audit report."""

    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def render_review_csv(manifest: BenchmarkManifest) -> str:
    """Export a spreadsheet-friendly packet without approving any case."""

    output = io.StringIO(newline="")
    fields = [
        "case_id",
        "split",
        "kind",
        "attack_category",
        "difficulty",
        "prompt",
        "expected_action",
        "topic_tags",
        "reference_source_ids",
        "reference_chunk_ids",
        "required_facts",
        "forbidden_content_count",
        "review_decision",
        "reviewer_notes",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for case in manifest.cases:
        writer.writerow(
            {
                "case_id": case.case_id,
                "split": case.split.value,
                "kind": case.kind.value,
                "attack_category": case.attack_category.value if case.attack_category else "",
                "difficulty": case.difficulty.value,
                "prompt": case.prompt,
                "expected_action": case.expected_action.value,
                "topic_tags": "|".join(case.topic_tags),
                "reference_source_ids": "|".join(case.reference_source_ids),
                "reference_chunk_ids": "|".join(case.reference_chunk_ids),
                "required_facts": "|".join(case.required_facts),
                "forbidden_content_count": len(case.forbidden_content),
                "review_decision": "",
                "reviewer_notes": "",
            }
        )
    return output.getvalue()


def write_text_artifact(path: Path, content: str) -> None:
    """Write a new generated artifact without silently replacing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to replace generated artifact: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")
