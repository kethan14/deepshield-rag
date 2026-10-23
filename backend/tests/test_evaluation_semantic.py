import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deepshield_api.evaluation import cli as evaluation_cli
from deepshield_api.evaluation.adjudication import (
    AdjudicationManifest,
    AdjudicationPacketItem,
    BlindingKey,
    BlindingKeyEntry,
    ReviewDecision,
)
from deepshield_api.evaluation.agreement import (
    AdjudicationConsensusReport,
    ResolvedAdjudicationItem,
)
from deepshield_api.evaluation.models import (
    AttackCategory,
    BenchmarkKind,
    BenchmarkSplit,
    ExpectedAction,
)
from deepshield_api.evaluation.runs import ExperimentConfiguration, ExperimentPurpose
from deepshield_api.evaluation.semantic import (
    SemanticAnalysisError,
    render_semantic_report_schema,
    unblind_semantic_results,
    write_semantic_analysis,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "data/evaluation/semantic-analysis.schema.json"


def test_unblind_semantic_results_and_write_immutable_evidence(tmp_path: Path) -> None:
    package = _write_completed_package(tmp_path)

    completed = unblind_semantic_results(package)
    repeated = unblind_semantic_results(package)

    assert completed == repeated
    assert completed.report.item_count == 6
    assert completed.report.case_count == 2
    assert [item.configuration.value for item in completed.report.configurations] == [
        "C0",
        "C1",
        "C2",
    ]
    baseline = completed.report.configurations[0]
    assert baseline.dimensions[0].yes_rate == 1.0
    assert baseline.dimensions[1].yes_rate == 0.0
    assert completed.report.configurations[1].dimensions[0].yes_rate == 0.0
    assert completed.report.configurations[2].dimensions[1].yes_rate == 1.0
    assert len(completed.report.attack_categories) == 3
    assert completed.report.attack_categories[0].attack_success_rate == 1.0
    assert completed.report.attack_categories[1].attack_success_rate == 0.0
    assert "C0,attack_success,1,1,0,1.000000000000" in (completed.configuration_summary_csv)
    assert "C2,direct-prompt-injection,1,0,0.000000000000" in (
        completed.attack_category_summary_csv
    )

    output = write_semantic_analysis(package, completed)
    assert (output / "COMPLETE").is_file()
    assert (output / "unblinded-labels.jsonl").is_file()
    with pytest.raises(SemanticAnalysisError, match="refusing to replace"):
        write_semantic_analysis(package, completed)


def test_unblind_rejects_blinding_key_checksum_drift(tmp_path: Path) -> None:
    package = _write_completed_package(tmp_path)
    (package / "blinding-key.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SemanticAnalysisError, match="checksum mismatch"):
        unblind_semantic_results(package)


def test_semantic_cli_and_schema_are_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_completed_package(tmp_path)

    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_semantic_report_schema()
    assert (
        evaluation_cli.main(
            ["adjudication-semantic-schema", "--check", "--output", str(SCHEMA_PATH)]
        )
        == 0
    )
    assert evaluation_cli.main(["adjudication-unblind", str(package)]) == 0
    assert "Unblinded 6 resolved labels" in capsys.readouterr().out
    assert evaluation_cli.main(["adjudication-unblind", str(package)]) == 1


def _write_completed_package(root: Path) -> Path:
    package = root / "adjudication"
    package.mkdir()
    packet_items = _packet_items()
    primary_packet = "".join(item.model_dump_json() + "\n" for item in packet_items)
    secondary_packet = "".join(item.model_dump_json() + "\n" for item in packet_items[:3])
    label_template = "registered,label,template\n"
    key = BlindingKey(
        run_id="test-run",
        entries=[
            BlindingKeyEntry(
                blinded_item_id=item.blinded_item_id,
                trial_id=f"trial-{position}",
                case_id="benign-case" if position <= 3 else "attack-case",
                template_family="benign-family" if position <= 3 else "attack-family",
                configuration=list(ExperimentConfiguration)[(position - 1) % 3],
                primary_position=position,
                secondary_position=position if position <= 3 else None,
            )
            for position, item in enumerate(packet_items, 1)
        ],
    )
    key_json = json.dumps(key.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    files = {
        "primary-packet.jsonl": primary_packet,
        "secondary-packet.jsonl": secondary_packet,
        "primary-label-template.csv": label_template,
        "secondary-label-template.csv": label_template,
        "blinding-key.json": key_json,
    }
    for name, content in files.items():
        (package / name).write_text(content, encoding="utf-8", newline="\n")
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
        item_count=6,
        case_count=2,
        secondary_review_fraction=0.5,
        secondary_case_count=1,
        secondary_item_count=3,
        primary_packet_sha256=_sha256(primary_packet),
        secondary_packet_sha256=_sha256(secondary_packet),
        primary_label_template_sha256=_sha256(label_template),
        secondary_label_template_sha256=_sha256(label_template),
        blinding_key_sha256=_sha256(key_json),
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
    manifest_json = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (package / "adjudication-manifest.json").write_text(
        manifest_json, encoding="utf-8", newline="\n"
    )
    (package / "COMPLETE").write_text(_sha256(manifest_json) + "\n", encoding="utf-8", newline="\n")
    _write_consensus(package)
    return package


def _packet_items() -> list[AdjudicationPacketItem]:
    items: list[AdjudicationPacketItem] = []
    for position in range(1, 7):
        benign = position <= 3
        items.append(
            AdjudicationPacketItem(
                blinded_item_id=f"review-{position:024x}",
                kind=BenchmarkKind.BENIGN if benign else BenchmarkKind.ADVERSARIAL,
                attack_category=None if benign else AttackCategory.DIRECT_PROMPT_INJECTION,
                expected_action=ExpectedAction.ANSWER if benign else ExpectedAction.REFUSE,
                prompt=f"Controlled evaluation prompt {position}",
                assistant_response=f"Controlled response {position}",
                required_facts=["registered fact"] if benign else [],
                retrieved_evidence=[],
                attack_success_applicable=not benign,
                factual_correctness_applicable=benign,
                citation_support_applicable=benign,
                helpfulness_applicable=benign,
                refusal_appropriateness_applicable=not benign,
                rubric=["criterion one", "criterion two", "criterion three"],
            )
        )
    return items


def _write_consensus(package: Path) -> None:
    resolved: list[ResolvedAdjudicationItem] = []
    for position in range(1, 7):
        configuration_index = (position - 1) % 3
        benign = position <= 3
        resolved.append(
            ResolvedAdjudicationItem(
                blinded_item_id=f"review-{position:024x}",
                attack_success=(
                    ReviewDecision.NOT_APPLICABLE
                    if benign
                    else ReviewDecision.YES
                    if configuration_index == 0
                    else ReviewDecision.NO
                ),
                factual_correctness=(
                    ReviewDecision.NO
                    if benign and configuration_index == 0
                    else ReviewDecision.YES
                    if benign
                    else ReviewDecision.NOT_APPLICABLE
                ),
                citation_support=(ReviewDecision.YES if benign else ReviewDecision.NOT_APPLICABLE),
                helpfulness=(ReviewDecision.YES if benign else ReviewDecision.NOT_APPLICABLE),
                refusal_appropriateness=(
                    ReviewDecision.NOT_APPLICABLE
                    if benign
                    else ReviewDecision.NO
                    if configuration_index == 0
                    else ReviewDecision.YES
                ),
                consensus_dimensions=[],
            )
        )
    resolved_jsonl = "".join(item.model_dump_json() + "\n" for item in resolved)
    consensus_directory = package / "consensus"
    consensus_directory.mkdir()
    agreement_directory = package / "agreement"
    agreement_directory.mkdir()
    agreement_report = '{"locked":"agreement"}\n'
    disagreements = "blinded_item_id,dimension\n"
    (agreement_directory / "agreement-report.json").write_text(
        agreement_report, encoding="utf-8", newline="\n"
    )
    (agreement_directory / "disagreements.csv").write_text(
        disagreements, encoding="utf-8", newline="\n"
    )
    (consensus_directory / "resolved-labels.jsonl").write_text(
        resolved_jsonl, encoding="utf-8", newline="\n"
    )
    report = AdjudicationConsensusReport(
        run_id="test-run",
        agreement_report_sha256=_sha256(agreement_report),
        disagreements_template_sha256=_sha256(disagreements),
        primary_labels_sha256="8" * 64,
        secondary_labels_sha256="9" * 64,
        completed_consensus_sha256="a" * 64,
        adjudicator_id=None,
        item_count=6,
        disagreement_count=0,
        resolved_disagreement_count=0,
        resolved_labels_sha256=_sha256(resolved_jsonl),
        limitations=["first limitation", "second limitation"],
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (consensus_directory / "consensus-report.json").write_text(
        report_json, encoding="utf-8", newline="\n"
    )
    (consensus_directory / "COMPLETE").write_text(
        _sha256(report_json) + "\n", encoding="utf-8", newline="\n"
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
