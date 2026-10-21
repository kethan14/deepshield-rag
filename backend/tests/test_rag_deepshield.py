import json
from collections.abc import Sequence
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from deepshield_api.classifier.training import ClassifierTrainingError
from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.evaluation.synthetic import build_synthetic_environment
from deepshield_api.main import create_app
from deepshield_api.rag.c2_guardrails import DeterministicC2Guardrails
from deepshield_api.rag.deepshield import (
    SYSTEM_POLICY_MARKER,
    DeepShieldRagService,
)
from deepshield_api.rag.generation import GenerationResult
from deepshield_api.rag.models import DeepShieldQuestion, GuardrailAction
from deepshield_api.rag.router import deepshield_service_dependency
from deepshield_api.rag.service import RagServiceError


def make_hit(rank: int, *, text: str | None = None) -> RetrievalHit:
    return RetrievalHit.model_validate(
        {
            "rank": rank,
            "score": 0.9 - rank / 100,
            "chunk_id": f"uscis-source-v1-c{rank:05d}",
            "source_id": "uscis-source",
            "title": f"Official policy source {rank}",
            "publisher": "uscis",
            "canonical_url": f"https://www.uscis.gov/example-{rank}",
            "char_start": (rank - 1) * 100,
            "char_end": rank * 100,
            "text": text or f"Official F-1 policy passage number {rank}.",
        }
    )


class FakeRetriever:
    model_name = "synthetic-retrieval-model"

    def __init__(self, hits: Sequence[RetrievalHit] | None = None) -> None:
        self.hits = list(hits if hits is not None else [make_hit(1), make_hit(2)])
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        self.calls.append((question, top_k))
        return self.hits[:top_k]


class FakeGenerator:
    model = "synthetic-generator"

    def __init__(self, text: str = "CPT requires prior authorization [S1].") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        self.calls.append((system_prompt, user_prompt))
        return GenerationResult(
            text=self.text,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )


class FakeClassifier:
    model_id: Literal["deepshield-c2-risk-classifier"] = "deepshield-c2-risk-classifier"
    model_version = 1
    threshold = 0.5

    def score(self, text: str) -> float:
        return 0.9 if "classifier-risk" in text else 0.1


def make_service(
    *,
    hits: Sequence[RetrievalHit] | None = None,
    generated_text: str = "CPT requires prior authorization [S1].",
) -> tuple[DeepShieldRagService, FakeRetriever, FakeGenerator]:
    retriever = FakeRetriever(hits)
    generator = FakeGenerator(generated_text)
    service = DeepShieldRagService(
        retriever,
        generator,
        FakeClassifier(),
        DeterministicC2Guardrails(build_synthetic_environment()),
    )
    return service, retriever, generator


def test_c2_answers_benign_question_with_complete_layered_evidence() -> None:
    service, _, generator = make_service()

    answer = service.answer(DeepShieldQuestion(question="Can an F-1 student use CPT?", top_k=2))

    assert answer.configuration == "C2"
    assert answer.outcome == "answered"
    assert answer.cited_reference_ids == ["S1"]
    assert len(answer.risk_assessments) == 3
    assert all(not assessment.risky for assessment in answer.risk_assessments)
    assert answer.risk_assessments[0].chunk_id is None
    assert answer.risk_assessments[1].chunk_id == "uscis-source-v1-c00001"
    assert any(event.code == "c2-output-clean" for event in answer.guardrail_events)
    assert SYSTEM_POLICY_MARKER in generator.calls[0][0]


def test_c2_blocks_classifier_attack_only_input_before_retrieval() -> None:
    service, retriever, generator = make_service()

    answer = service.answer(DeepShieldQuestion(question="Perform classifier-risk behavior now."))

    assert answer.outcome == "refused"
    assert retriever.calls == []
    assert generator.calls == []
    assert answer.risk_assessments[0].risky is True
    assert any(event.code == "classifier-input-risk" for event in answer.guardrail_events)


def test_c2_preserves_policy_utility_for_mixed_classifier_risk() -> None:
    service, _, generator = make_service()

    answer = service.answer(
        DeepShieldQuestion(question="What are the F-1 CPT rules with classifier-risk text?")
    )

    assert answer.outcome == "answered"
    event = next(
        event for event in answer.guardrail_events if event.code == "classifier-input-risk"
    )
    assert event.action is GuardrailAction.WARN
    payload = json.loads(generator.calls[0][1].split("\n", maxsplit=1)[1])
    assert "classifier-input-risk" in payload["input_risk_codes"]


def test_c2_excludes_retrieval_detected_only_by_classifier() -> None:
    risky = make_hit(1, text="A subtle classifier-risk payload without rule keywords.")
    safe = make_hit(2)
    service, _, generator = make_service(
        hits=[risky, safe],
        generated_text="The safe official passage supports this answer [S2].",
    )

    answer = service.answer(DeepShieldQuestion(question="What are F-1 travel rules?"))

    assert answer.outcome == "answered"
    assert answer.excluded_chunk_ids == [risky.chunk_id]
    assert [passage.reference_id for passage in answer.retrieved_passages] == ["S2"]
    assert any(event.code == "classifier-retrieval-risk" for event in answer.guardrail_events)
    payload = json.loads(generator.calls[0][1].split("\n", maxsplit=1)[1])
    assert [item["reference_id"] for item in payload["official_passages"]] == ["S2"]


def test_c2_refuses_when_classifier_excludes_every_retrieved_passage() -> None:
    risky = make_hit(1, text="A subtle classifier-risk payload without rule keywords.")
    service, _, generator = make_service(hits=[risky])

    answer = service.answer(DeepShieldQuestion(question="What are F-1 travel rules?"))

    assert answer.outcome == "refused"
    assert generator.calls == []
    assert answer.excluded_chunk_ids == [risky.chunk_id]
    assert any(event.code == "no-safe-retrieval-context" for event in answer.guardrail_events)


def test_c2_rejects_empty_retrieval_as_service_failure() -> None:
    service, _, _ = make_service(hits=[])

    with pytest.raises(RagServiceError, match="no policy passages"):
        service.answer(DeepShieldQuestion(question="What are F-1 travel rules?"))


def test_c2_blocks_registered_fictional_pii_from_generated_output() -> None:
    record = build_synthetic_environment().records[0]
    service, _, _ = make_service(generated_text=f"The private email is {record.email} [S1].")

    answer = service.answer(DeepShieldQuestion(question="Can an F-1 student use CPT?"))

    assert answer.outcome == "refused"
    assert record.email not in answer.answer
    assert answer.generated_output_sha256 is not None
    assert any(event.code == "fictional-email-detected" for event in answer.guardrail_events)


def test_c2_blocks_its_system_policy_marker() -> None:
    service, _, _ = make_service(generated_text=f"Marker: {SYSTEM_POLICY_MARKER} [S1].")

    answer = service.answer(DeepShieldQuestion(question="Can an F-1 student use CPT?"))

    assert answer.outcome == "refused"
    assert any(event.code == "c2-system-policy-disclosure" for event in answer.guardrail_events)


def test_c2_blocks_generated_output_without_valid_citation() -> None:
    service, _, _ = make_service(generated_text="An unsupported answer without evidence.")

    answer = service.answer(DeepShieldQuestion(question="Can an F-1 student use CPT?"))

    assert answer.outcome == "refused"
    assert any(event.code == "missing-citation" for event in answer.guardrail_events)


def test_c2_api_returns_layered_evidence_with_dependency_override() -> None:
    service, _, _ = make_service()
    app = create_app()
    app.dependency_overrides[deepshield_service_dependency] = lambda: service

    response = TestClient(app).post(
        "/api/v1/assistant/query/c2",
        json={"question": "Can an F-1 student use CPT?", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["configuration"] == "C2"
    assert len(response.json()["risk_assessments"]) == 3


def test_c2_dependency_hides_missing_unreviewed_classifier_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepshield_api.rag.router.get_deepshield_rag_service",
        lambda: (_ for _ in ()).throw(ClassifierTrainingError("private artifact path")),
    )

    response = TestClient(create_app()).post(
        "/api/v1/assistant/query/c2",
        json={"question": "Can an F-1 student use CPT?"},
    )

    assert response.status_code == 503
    assert "private artifact path" not in response.text


def test_openapi_includes_c2_query_contract() -> None:
    operation = create_app().openapi()["paths"]["/api/v1/assistant/query/c2"]["post"]

    assert operation["tags"] == ["assistant"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DeepShieldAnswer"
    }
