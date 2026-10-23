import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.adjudication import (
    AdjudicationManifest,
    AdjudicationPacketItem,
)
from deepshield_api.evaluation.agreement import (
    AgreementAnalysisError,
    analyze_adjudication_agreement,
    render_agreement_report_schema,
    write_agreement_analysis,
)
from deepshield_api.evaluation.models import (
    AttackCategory,
    BenchmarkKind,
    BenchmarkSplit,
    ExpectedAction,
)
from deepshield_api.evaluation.runs import ExperimentPurpose

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "data/evaluation/adjudication-agreement.schema.json"
LABEL_FIELDS = [
    "blinded_item_id",
    "reviewer_id",
    "attack_success",
    "factual_correctness",
    "citation_support",
    "helpfulness",
    "refusal_appropriateness",
    "confidence",
    "notes",
]


def test_agreement_analysis_is_blinded_and_reproducible(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    primary_labels = tmp_path / "primary-labels.csv"
    secondary_labels = tmp_path / "secondary-labels.csv"
    primary_labels.write_text(
        _labels("primary-reviewer", attack=("no", "no", "no"), refusal=("yes", "yes", "yes")),
        encoding="utf-8",
    )
    secondary_labels.write_text(
        _labels("secondary-reviewer", attack=("no", "yes", "no"), refusal=("yes", "yes", "no")),
        encoding="utf-8",
    )

    first = analyze_adjudication_agreement(
        package_directory=package,
        primary_labels_path=primary_labels,
        secondary_labels_path=secondary_labels,
    )
    second = analyze_adjudication_agreement(
        package_directory=package,
        primary_labels_path=primary_labels,
        secondary_labels_path=secondary_labels,
    )

    assert first == second
    assert first.report.overlap_item_count == 3
    assert first.report.applicable_decision_count == 6
    assert first.report.agreement_count == 4
    assert first.report.disagreement_count == 2
    assert first.report.overall_percent_agreement == pytest.approx(2 / 3)
    assert first.report.dimensions[0].cohen_kappa == pytest.approx(0)
    assert first.report.dimensions[1].percent_agreement is None
    assert first.report.dimensions[4].cohen_kappa == pytest.approx(0)
    assert "configuration" not in first.disagreements_csv
    assert "review-000000000000000000000002,attack_success,no,yes" in (first.disagreements_csv)


def test_agreement_writer_and_cli_are_no_overwrite(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    primary_labels = tmp_path / "primary-labels.csv"
    secondary_labels = tmp_path / "secondary-labels.csv"
    primary_labels.write_text(
        _labels("primary-reviewer", attack=("no", "no", "no"), refusal=("yes", "yes", "yes")),
        encoding="utf-8",
    )
    secondary_labels.write_text(
        _labels("secondary-reviewer", attack=("no", "no", "no"), refusal=("yes", "yes", "yes")),
        encoding="utf-8",
    )
    arguments = [
        "adjudication-agreement",
        str(package),
        str(primary_labels),
        str(secondary_labels),
    ]

    assert evaluation_cli.main(arguments) == 0
    assert (package / "agreement/COMPLETE").is_file()
    assert evaluation_cli.main(arguments) == 1
    completed = analyze_adjudication_agreement(
        package_directory=package,
        primary_labels_path=primary_labels,
        secondary_labels_path=secondary_labels,
    )
    with pytest.raises(AgreementAnalysisError, match="refusing to replace"):
        write_agreement_analysis(package, completed)


def test_agreement_rejects_same_reviewer_and_package_drift(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    primary_labels = tmp_path / "primary-labels.csv"
    secondary_labels = tmp_path / "secondary-labels.csv"
    content = _labels("same-reviewer", attack=("no", "no", "no"), refusal=("yes", "yes", "yes"))
    primary_labels.write_text(content, encoding="utf-8")
    secondary_labels.write_text(content, encoding="utf-8")

    with pytest.raises(AgreementAnalysisError, match="different reviewer IDs"):
        analyze_adjudication_agreement(
            package_directory=package,
            primary_labels_path=primary_labels,
            secondary_labels_path=secondary_labels,
        )

    (package / "primary-packet.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AgreementAnalysisError, match="checksum mismatch"):
        analyze_adjudication_agreement(
            package_directory=package,
            primary_labels_path=primary_labels,
            secondary_labels_path=secondary_labels,
        )


def test_agreement_schema_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_agreement_report_schema()
    assert (
        evaluation_cli.main(
            ["adjudication-agreement-schema", "--check", "--output", str(SCHEMA_PATH)]
        )
        == 0
    )
    assert "Agreement schema is current" in capsys.readouterr().out


def _write_package(root: Path) -> Path:
    package = root / "adjudication"
    package.mkdir()
    packet = "".join(item.model_dump_json() + "\n" for item in _packet_items())
    template = "registered,label,template\n"
    key = '{"private":"sealed"}\n'
    (package / "primary-packet.jsonl").write_text(packet, encoding="utf-8", newline="\n")
    (package / "secondary-packet.jsonl").write_text(packet, encoding="utf-8", newline="\n")
    (package / "primary-label-template.csv").write_text(template, encoding="utf-8", newline="\n")
    (package / "secondary-label-template.csv").write_text(template, encoding="utf-8", newline="\n")
    (package / "blinding-key.json").write_text(key, encoding="utf-8", newline="\n")
    manifest = AdjudicationManifest(
        run_id="test-run",
        purpose=ExperimentPurpose.DEVELOPMENT,
        benchmark_split=BenchmarkSplit.DEVELOPMENT,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_manifest_sha256="1" * 64,
        results_sha256="2" * 64,
        benchmark_sha256="3" * 64,
        chunking_report_sha256="4" * 64,
        chunks_sha256="5" * 64,
        item_count=3,
        case_count=1,
        secondary_review_fraction=0.2,
        secondary_case_count=1,
        secondary_item_count=3,
        primary_packet_sha256=_sha256(packet),
        secondary_packet_sha256=_sha256(packet),
        primary_label_template_sha256=_sha256(template),
        secondary_label_template_sha256=_sha256(template),
        blinding_key_sha256=_sha256(key),
        dimensions=[
            "attack-success",
            "factual-correctness",
            "citation-support",
            "helpfulness",
            "refusal-appropriateness",
        ],
        instructions=["first instruction", "second instruction", "third instruction"],
        limitations=["configuration blinding is imperfect"],
    )
    manifest_text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (package / "adjudication-manifest.json").write_text(
        manifest_text, encoding="utf-8", newline="\n"
    )
    (package / "COMPLETE").write_text(_sha256(manifest_text) + "\n", encoding="utf-8", newline="\n")
    return package


def _packet_items() -> list[AdjudicationPacketItem]:
    return [
        AdjudicationPacketItem(
            blinded_item_id=f"review-{index:024x}",
            kind=BenchmarkKind.ADVERSARIAL,
            attack_category=AttackCategory.DIRECT_PROMPT_INJECTION,
            expected_action=ExpectedAction.REFUSE,
            prompt=f"Controlled attack {index}",
            assistant_response=f"Safe response {index}",
            required_facts=[],
            retrieved_evidence=[],
            attack_success_applicable=True,
            factual_correctness_applicable=False,
            citation_support_applicable=False,
            helpfulness_applicable=False,
            refusal_appropriateness_applicable=True,
            rubric=["criterion one", "criterion two", "criterion three"],
        )
        for index in range(1, 4)
    ]


def _labels(
    reviewer_id: str,
    *,
    attack: tuple[str, str, str],
    refusal: tuple[str, str, str],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=LABEL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, (attack_decision, refusal_decision) in enumerate(
        zip(attack, refusal, strict=True), 1
    ):
        writer.writerow(
            {
                "blinded_item_id": f"review-{index:024x}",
                "reviewer_id": reviewer_id,
                "attack_success": attack_decision,
                "factual_correctness": "not-applicable",
                "citation_support": "not-applicable",
                "helpfulness": "not-applicable",
                "refusal_appropriateness": refusal_decision,
                "confidence": "high",
                "notes": "",
            }
        )
    return output.getvalue()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
