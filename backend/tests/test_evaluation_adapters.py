import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.evaluation.adapters import (
    ControlledExperimentRetriever,
    ExperimentAdapterError,
    build_comparison_backends,
)
from deepshield_api.evaluation.manifest import load_benchmark_manifest
from deepshield_api.evaluation.models import AttackCategory, BenchmarkCase, BenchmarkKind
from deepshield_api.evaluation.runs import ExperimentConfiguration, ObservedOutcome
from deepshield_api.evaluation.synthetic import (
    SyntheticSafetyEnvironment,
    build_synthetic_environment,
)
from deepshield_api.rag.generation import GenerationResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPOSITORY_ROOT / "data/evaluation/f1-security-benchmark.v1.json"


def _official_hit(rank: int) -> RetrievalHit:
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
            "text": f"Official F-1 policy passage number {rank}.",
        }
    )


class FakeRetriever:
    model_name = "synthetic-retrieval-model"

    def __init__(self) -> None:
        self.hits = [_official_hit(rank) for rank in range(1, 6)]
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        self.calls.append((question, top_k))
        return self.hits[:top_k]


class FakeGenerator:
    model = "synthetic-generator"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        self.calls.append((system_prompt, user_prompt))
        reference = re.search(r"\[S[1-9][0-9]*\]", user_prompt)
        citation = reference.group(0) if reference is not None else "[S1]"
        return GenerationResult(
            text=f"Controlled policy answer {citation}",
            model=self.model,
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        )


class FakeClassifier:
    model_id: Literal["deepshield-c2-risk-classifier"] = "deepshield-c2-risk-classifier"
    model_version = 1
    threshold = 0.5

    def score(self, text: str) -> float:
        risky_markers = ("controlled untrusted retrieval", "@example.test")
        return 0.9 if any(marker in text.casefold() for marker in risky_markers) else 0.1


@pytest.fixture(scope="module")
def environment() -> SyntheticSafetyEnvironment:
    return build_synthetic_environment()


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return load_benchmark_manifest(BENCHMARK_PATH).cases


def _case(
    cases: Sequence[BenchmarkCase],
    *,
    category: AttackCategory | None = None,
    kind: BenchmarkKind | None = None,
) -> BenchmarkCase:
    return next(
        case
        for case in cases
        if (category is None or case.attack_category is category)
        and (kind is None or case.kind is kind)
    )


def test_controlled_retriever_injects_explicit_poison_without_changing_corpus(
    environment: SyntheticSafetyEnvironment,
    cases: list[BenchmarkCase],
) -> None:
    base = FakeRetriever()
    controlled = ControlledExperimentRetriever(base, environment)
    case = _case(cases, category=AttackCategory.INDIRECT_PROMPT_INJECTION)

    with controlled.activate(case):
        hits = controlled.retrieve(case.prompt, top_k=5)
        assert controlled.last_hits == tuple(hits)

    assert len(hits) == 5
    assert hits[0].chunk_id == case.synthetic_retrieval_chunks[0].synthetic_chunk_id
    assert hits[0].publisher == "synthetic-evaluation"
    assert str(hits[0].canonical_url).startswith("https://example.test/")
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert [hit.chunk_id for hit in hits[1:]] == [hit.chunk_id for hit in base.hits[:4]]
    assert base.calls == [(case.prompt, 4)]
    assert controlled.last_hits == ()
    with pytest.raises(ExperimentAdapterError, match="case must be active"):
        controlled.retrieve(case.prompt, top_k=5)


def test_controlled_retriever_supplies_only_fictional_private_records(
    environment: SyntheticSafetyEnvironment,
    cases: list[BenchmarkCase],
) -> None:
    controlled = ControlledExperimentRetriever(FakeRetriever(), environment)
    case = _case(cases, category=AttackCategory.FICTIONAL_PII_EXFILTRATION)
    record = next(
        item for item in environment.records if item.fixture_id in case.safety_fixture_ids
    )

    with controlled.activate(case):
        hit = controlled.retrieve(case.prompt, top_k=5)[0]

    assert hit.publisher == "synthetic-evaluation"
    assert hit.source_id == "synthetic-private-records"
    assert "UNTRUSTED FICTIONAL RECORD" in hit.text
    assert record.email in hit.text
    assert record.phone in hit.text
    assert record.synthetic_sevis_id in hit.text
    assert record.canary_token in hit.text


def test_real_services_are_normalized_with_identical_benign_retrieval(
    environment: SyntheticSafetyEnvironment,
    cases: list[BenchmarkCase],
) -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    backends = build_comparison_backends(
        retriever=retriever,
        generator=generator,
        classifier=FakeClassifier(),
        environment=environment,
    )
    case = _case(cases, kind=BenchmarkKind.BENIGN)

    observations = [backend.run(case) for backend in backends]

    assert [backend.configuration for backend in backends] == list(ExperimentConfiguration)
    assert all(observation.outcome is ObservedOutcome.ANSWERED for observation in observations)
    retrieved_ids = [
        [evidence.chunk_id for evidence in observation.retrieved] for observation in observations
    ]
    assert retrieved_ids[0] == retrieved_ids[1] == retrieved_ids[2]
    assert len(generator.calls) == 3
    assert retriever.calls == [(case.prompt, 5)]


def test_indirect_injection_is_visible_to_c0_and_excluded_by_c1_and_c2(
    environment: SyntheticSafetyEnvironment,
    cases: list[BenchmarkCase],
) -> None:
    backends = build_comparison_backends(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        classifier=FakeClassifier(),
        environment=environment,
    )
    case = _case(cases, category=AttackCategory.INDIRECT_PROMPT_INJECTION)
    poison_id = case.synthetic_retrieval_chunks[0].synthetic_chunk_id

    c0, c1, c2 = [backend.run(case) for backend in backends]

    assert poison_id in [evidence.chunk_id for evidence in c0.retrieved]
    assert c0.excluded_chunk_ids == []
    assert poison_id in c1.excluded_chunk_ids
    assert poison_id in c2.excluded_chunk_ids
    assert any(event.chunk_id == poison_id for event in c1.guardrail_events)
    assert any(
        assessment.chunk_id == poison_id and assessment.risky for assessment in c2.risk_assessments
    )


def test_c2_adapter_applies_deny_by_default_mock_tool_authorization(
    environment: SyntheticSafetyEnvironment,
    cases: list[BenchmarkCase],
) -> None:
    c2 = build_comparison_backends(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        classifier=FakeClassifier(),
        environment=environment,
    )[2]
    case = _case(cases, category=AttackCategory.UNAUTHORIZED_TOOL_ACTION)

    observation = c2.run(case)

    assert observation.outcome is ObservedOutcome.REFUSED
    assert any(event.code == "missing-tool-scope" for event in observation.guardrail_events)
    assert not any(event.code == "mock-tool-authorized" for event in observation.guardrail_events)
