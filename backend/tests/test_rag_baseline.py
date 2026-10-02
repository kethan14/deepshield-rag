import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.main import create_app
from deepshield_api.rag.generation import (
    MAX_RESPONSE_BYTES,
    GenerationError,
    GenerationResult,
    OpenAICompatibleGenerator,
)
from deepshield_api.rag.models import BaselineQuestion
from deepshield_api.rag.router import baseline_service_dependency
from deepshield_api.rag.runtime import RagRuntimeError, RagSettings, build_baseline_rag_service
from deepshield_api.rag.service import BaselineRagService, RagServiceError


def make_hit(rank: int, *, source_id: str = "uscis-policy") -> RetrievalHit:
    return RetrievalHit.model_validate(
        {
            "rank": rank,
            "score": 0.9 - rank / 100,
            "chunk_id": f"{source_id}-v1-c{rank:05d}",
            "source_id": source_id,
            "title": f"Official policy source {rank}",
            "publisher": "uscis",
            "canonical_url": f"https://www.uscis.gov/example-{rank}",
            "char_start": (rank - 1) * 100,
            "char_end": rank * 100,
            "text": f"Official policy passage number {rank}.",
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

    def __init__(
        self,
        text: str = "CPT requires authorization [S1]. Travel rules differ [S2].",
    ) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        self.calls.append((system_prompt, user_prompt))
        return GenerationResult(
            text=self.text,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=120,
            completion_tokens=20,
            total_tokens=140,
        )


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return self.content


def test_baseline_service_returns_complete_retrieval_and_citation_evidence() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator("Use the first rule [S1], repeat [S1], and ignore [S9].")
    service = BaselineRagService(retriever, generator)

    answer = service.answer(BaselineQuestion(question="Can I work through CPT?", top_k=2))

    assert answer.configuration == "C0"
    assert answer.prompt_version == "c0-v1"
    assert answer.cited_reference_ids == ["S1"]
    assert answer.invalid_reference_ids == ["S9"]
    assert [passage.reference_id for passage in answer.retrieved_passages] == ["S1", "S2"]
    assert answer.generator_model == generator.model
    assert answer.retrieval_model == retriever.model_name
    assert answer.usage.total_tokens == 140
    assert retriever.calls == [("Can I work through CPT?", 2)]
    assert "using only the retrieved official passages" in generator.calls[0][0]
    assert "Question:\nCan I work through CPT?" in generator.calls[0][1]
    assert "[S1]\nTitle: Official policy source 1" in generator.calls[0][1]
    assert "https://www.uscis.gov/example-2" in generator.calls[0][1]


def test_baseline_service_records_no_citations_without_inventing_them() -> None:
    service = BaselineRagService(FakeRetriever(), FakeGenerator("The sources are insufficient."))

    answer = service.answer(BaselineQuestion(question="An unsupported policy question"))

    assert answer.cited_reference_ids == []
    assert answer.invalid_reference_ids == []


def test_baseline_service_rejects_empty_retrieval() -> None:
    service = BaselineRagService(FakeRetriever([]), FakeGenerator())

    with pytest.raises(RagServiceError, match="no policy passages"):
        service.answer(BaselineQuestion(question="Can I work through CPT?"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"question": "  "}, "at least 3 characters"),
        ({"question": "valid question", "top_k": 0}, "greater than or equal to 1"),
        ({"question": "valid question", "top_k": 11}, "less than or equal to 10"),
        ({"question": "valid question", "unexpected": True}, "Extra inputs are not permitted"),
    ],
)
def test_baseline_question_validation(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        BaselineQuestion.model_validate(payload)


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://example.com/v1",
        "http://example.com/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?secret=value",
        "https://example.com/v1#fragment",
    ],
)
def test_generator_rejects_unsafe_or_invalid_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleGenerator(base_url=base_url, model="test-model")


def test_generator_accepts_local_http_and_hosted_https() -> None:
    local = OpenAICompatibleGenerator(
        base_url="http://127.0.0.1:11434/v1/",
        model=" local-model ",
    )
    hosted = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        model="hosted-model",
        api_key=" secret ",
    )

    assert local.model == "local-model"
    assert hosted.model == "hosted-model"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0.5},
        {"timeout_seconds": 301.0},
        {"max_output_tokens": 0},
        {"max_output_tokens": 4_097},
    ],
)
def test_generator_rejects_invalid_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleGenerator(
            base_url="http://localhost:11434/v1",
            model="test-model",
            **kwargs,
        )


def test_generator_sends_expected_request_and_parses_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    response = {
        "model": "provider-model-snapshot",
        "choices": [{"message": {"content": " Grounded answer [S1]. "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(response).encode())

    monkeypatch.setattr("deepshield_api.rag.generation.urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1/",
        model="requested-model",
        api_key="top-secret",
        timeout_seconds=12,
        max_output_tokens=345,
    )

    result = generator.generate("system text", "user text")
    request = captured["request"]
    body = json.loads(request.data)

    assert request.full_url == "https://api.example.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer top-secret"
    assert captured["timeout"] == 12
    assert body == {
        "model": "requested-model",
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        "temperature": 0,
        "max_tokens": 345,
        "stream": False,
    }
    assert result.text == "Grounded answer [S1]."
    assert result.model == "provider-model-snapshot"
    assert result.total_tokens == 15


def test_generator_supports_frozen_reasoning_model_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    response = {"choices": [{"message": {"content": "Answer [S1]."}}]}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        if not isinstance(request.data, bytes):
            raise AssertionError("expected a byte request body")
        captured["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(response).encode())

    monkeypatch.setattr("deepshield_api.rag.generation.urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.openai.com/v1",
        model="gpt-5.4-mini-2026-03-17",
        reasoning_effort="none",
        max_token_parameter="max_completion_tokens",
        max_output_tokens=700,
    )

    generator.generate("system", "user")

    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_completion_tokens"] == 700
    assert "max_tokens" not in captured["body"]


def test_generator_omits_temperature_for_positive_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    response = {"choices": [{"message": {"content": "Answer [S1]."}}]}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        if not isinstance(request.data, bytes):
            raise AssertionError("expected a byte request body")
        captured["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(response).encode())

    monkeypatch.setattr("deepshield_api.rag.generation.urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        model="reasoning-model",
        reasoning_effort="low",
        max_token_parameter="max_completion_tokens",
    )

    generator.generate("system", "user")

    assert captured["body"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["body"]


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b"{}",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": 7}}]}',
        b'{"choices": [{"message": {"content": "  "}}]}',
    ],
)
def test_generator_rejects_invalid_provider_responses(
    content: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepshield_api.rag.generation.urlopen",
        lambda request, timeout: FakeResponse(content),
    )
    generator = OpenAICompatibleGenerator(
        base_url="http://localhost:11434/v1",
        model="test-model",
    )

    with pytest.raises(GenerationError):
        generator.generate("system", "user")


def test_generator_rejects_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepshield_api.rag.generation.urlopen",
        lambda request, timeout: FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1)),
    )
    generator = OpenAICompatibleGenerator(
        base_url="http://localhost:11434/v1",
        model="test-model",
    )

    with pytest.raises(GenerationError, match="size limit"):
        generator.generate("system", "user")


def test_generator_hides_transport_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(request: Request, timeout: float) -> FakeResponse:
        raise URLError("top-secret transport detail")

    monkeypatch.setattr("deepshield_api.rag.generation.urlopen", fail)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="top-secret",
    )

    with pytest.raises(GenerationError, match="request failed") as error:
        generator.generate("system", "user")
    assert "top-secret" not in str(error.value)


def test_runtime_requires_explicit_generation_model(tmp_path: Path) -> None:
    settings = RagSettings(repository_root=tmp_path, llm_model=None)

    with pytest.raises(RagRuntimeError, match="DEEPSHIELD_LLM_MODEL"):
        build_baseline_rag_service(settings)


def test_runtime_rejects_artifact_path_outside_repository(tmp_path: Path) -> None:
    settings = RagSettings(
        repository_root=tmp_path,
        index_report=tmp_path.parent / "outside.json",
        llm_model="test-model",
    )

    with pytest.raises(RagRuntimeError, match="inside the repository"):
        build_baseline_rag_service(settings)


def test_api_returns_c0_answer_with_dependency_override() -> None:
    app = create_app()
    app.dependency_overrides[baseline_service_dependency] = lambda: BaselineRagService(
        FakeRetriever(), FakeGenerator()
    )

    response = TestClient(app).post(
        "/api/v1/assistant/query",
        json={"question": "Can an F-1 student use CPT?", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["configuration"] == "C0"
    assert payload["cited_reference_ids"] == ["S1", "S2"]
    assert len(payload["retrieved_passages"]) == 2


def test_api_rejects_invalid_request_before_running_service() -> None:
    app = create_app()
    app.dependency_overrides[baseline_service_dependency] = lambda: BaselineRagService(
        FakeRetriever(), FakeGenerator()
    )

    response = TestClient(app).post(
        "/api/v1/assistant/query",
        json={"question": "x", "top_k": 99},
    )

    assert response.status_code == 422


def test_api_reports_unconfigured_runtime_without_exposing_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepshield_api.rag.router.get_baseline_rag_service",
        lambda: (_ for _ in ()).throw(RagRuntimeError("sensitive local path")),
    )

    response = TestClient(create_app()).post(
        "/api/v1/assistant/query",
        json={"question": "Can an F-1 student use CPT?"},
    )

    assert response.status_code == 503
    assert "sensitive local path" not in response.text


def test_api_maps_generation_failure_to_bad_gateway() -> None:
    class FailingGenerator(FakeGenerator):
        def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
            raise GenerationError("provider secret")

    app = create_app()
    app.dependency_overrides[baseline_service_dependency] = lambda: BaselineRagService(
        FakeRetriever(), FailingGenerator()
    )

    response = TestClient(app).post(
        "/api/v1/assistant/query",
        json={"question": "Can an F-1 student use CPT?"},
    )

    assert response.status_code == 502
    assert "provider secret" not in response.text


def test_openapi_includes_baseline_query_contract() -> None:
    schema = create_app().openapi()

    operation = schema["paths"]["/api/v1/assistant/query"]["post"]
    assert operation["tags"] == ["assistant"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/BaselineAnswer"
    }
