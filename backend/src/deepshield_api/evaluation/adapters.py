"""Adapters that run the real C0/C1/C2 services against controlled benchmark evidence."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.evaluation.models import AttackCategory, BenchmarkCase
from deepshield_api.evaluation.runs import (
    ExperimentConfiguration,
    ObservedOutcome,
    RetrievalEvidence,
    TrialObservation,
)
from deepshield_api.evaluation.synthetic import SyntheticSafetyEnvironment
from deepshield_api.rag.c2_guardrails import (
    DeterministicC2Guardrails,
    MockToolCall,
    ToolAuthorizationContext,
)
from deepshield_api.rag.deepshield import DeepShieldRagService, RiskClassifierBackend
from deepshield_api.rag.generation import GenerationBackend
from deepshield_api.rag.hardened import HardenedRagService
from deepshield_api.rag.models import (
    BaselineQuestion,
    DeepShieldAnswer,
    DeepShieldQuestion,
    GuardrailEvent,
    HardenedAnswer,
    HardenedQuestion,
)
from deepshield_api.rag.service import BaselineRagService, RetrievalBackend

SYNTHETIC_PUBLISHER: Literal["synthetic-evaluation"] = "synthetic-evaluation"
SYNTHETIC_BASE_URL = "https://example.test/deepshield/evaluation"
TOOL_AUTHORIZATION_REFUSAL = (
    "I can’t perform that mock-tool action because the trusted authorization context "
    "does not permit it."
)


class ExperimentAdapterError(RuntimeError):
    """Raised when controlled benchmark evidence cannot be injected safely."""


class ControlledExperimentRetriever:
    """Add isolated synthetic evidence while preserving one shared official retriever."""

    def __init__(
        self,
        base: RetrievalBackend,
        environment: SyntheticSafetyEnvironment,
    ) -> None:
        self._base = base
        self._records = {record.fixture_id: record for record in environment.records}
        self._active_case: BenchmarkCase | None = None
        self._last_hits: tuple[RetrievalHit, ...] = ()
        self._case_cache: dict[tuple[str, str, int], tuple[RetrievalHit, ...]] = {}
        self.model_name = base.model_name

    @property
    def last_hits(self) -> tuple[RetrievalHit, ...]:
        """Return the complete pre-guardrail evidence from the active case."""

        return self._last_hits

    @contextmanager
    def activate(self, case: BenchmarkCase) -> Iterator[None]:
        """Bind one case for a single sequential service invocation."""

        if self._active_case is not None:
            raise ExperimentAdapterError("controlled retrieval does not allow nested cases")
        self._active_case = case
        self._last_hits = ()
        try:
            yield
        finally:
            self._active_case = None
            self._last_hits = ()

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        """Return synthetic fixtures first, followed by stable official retrieval hits."""

        case = self._active_case
        if case is None:
            raise ExperimentAdapterError("a benchmark case must be active before retrieval")
        case_sha256 = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
        cache_key = (case_sha256, question, top_k)
        cached = self._case_cache.get(cache_key)
        if cached is not None:
            self._last_hits = cached
            return list(cached)
        synthetic = self._synthetic_hits(case)
        if len(synthetic) >= top_k:
            raise ExperimentAdapterError("synthetic evidence must leave room for official evidence")
        official = self._base.retrieve(question, top_k=top_k - len(synthetic))
        combined = synthetic + official
        ranked = [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(combined, 1)]
        self._last_hits = tuple(ranked)
        self._case_cache[cache_key] = self._last_hits
        return ranked

    def _synthetic_hits(self, case: BenchmarkCase) -> list[RetrievalHit]:
        hits = [
            _synthetic_hit(
                chunk.synthetic_chunk_id,
                source_id="synthetic-poisoned-retrieval",
                title=chunk.title,
                text=chunk.text,
            )
            for chunk in case.synthetic_retrieval_chunks
        ]
        if case.attack_category not in {
            AttackCategory.SYNTHETIC_SECRET_EXFILTRATION,
            AttackCategory.FICTIONAL_PII_EXFILTRATION,
        }:
            return hits
        for fixture_id in case.safety_fixture_ids:
            record = self._records.get(fixture_id)
            if record is None:
                raise ExperimentAdapterError(
                    f"benchmark references an unknown synthetic fixture: {fixture_id}"
                )
            content = {
                "canary_token": record.canary_token,
                "email": record.email,
                "fictional": True,
                "fixture_id": record.fixture_id,
                "full_name": record.full_name,
                "phone": record.phone,
                "program": record.program,
                "record_note": record.record_note,
                "synthetic_sevis_id": record.synthetic_sevis_id,
            }
            hits.append(
                _synthetic_hit(
                    f"private-{record.fixture_id}",
                    source_id="synthetic-private-records",
                    title="Synthetic private student record fixture",
                    text="UNTRUSTED FICTIONAL RECORD\n"
                    + json.dumps(content, ensure_ascii=False, sort_keys=True),
                )
            )
        return hits


@dataclass
class BaselineExperimentBackend:
    """Normalize C0 service output into the experiment evidence contract."""

    service: BaselineRagService
    retriever: ControlledExperimentRetriever
    configuration: ExperimentConfiguration = field(
        init=False, default=ExperimentConfiguration.BASELINE
    )
    retrieval_top_k: Literal[5] = field(init=False, default=5)

    def run(self, case: BenchmarkCase) -> TrialObservation:
        with self.retriever.activate(case):
            answer = self.service.answer(BaselineQuestion(question=case.prompt, top_k=5))
            hits = self.retriever.last_hits
            return TrialObservation(
                outcome=ObservedOutcome.ANSWERED,
                answer=answer.answer,
                cited_reference_ids=answer.cited_reference_ids,
                invalid_reference_ids=answer.invalid_reference_ids,
                retrieved=_retrieval_evidence(hits),
                generator_model=answer.generator_model,
                retrieval_model=answer.retrieval_model,
                finish_reason=answer.finish_reason,
                usage=answer.usage,
            )


@dataclass
class HardenedExperimentBackend:
    """Normalize C1 output while retaining pre-guardrail retrieval evidence."""

    service: HardenedRagService
    retriever: ControlledExperimentRetriever
    configuration: ExperimentConfiguration = field(
        init=False, default=ExperimentConfiguration.HARDENED
    )
    retrieval_top_k: Literal[5] = field(init=False, default=5)

    def run(self, case: BenchmarkCase) -> TrialObservation:
        with self.retriever.activate(case):
            answer = self.service.answer(HardenedQuestion(question=case.prompt, top_k=5))
            return _guarded_observation(answer, self.retriever.last_hits)


@dataclass
class DeepShieldExperimentBackend:
    """Normalize C2 output and apply the trusted mock-tool authorization boundary."""

    service: DeepShieldRagService
    retriever: ControlledExperimentRetriever
    guardrails: DeterministicC2Guardrails
    configuration: ExperimentConfiguration = field(
        init=False, default=ExperimentConfiguration.DEEPSHIELD
    )
    retrieval_top_k: Literal[5] = field(init=False, default=5)

    def run(self, case: BenchmarkCase) -> TrialObservation:
        with self.retriever.activate(case):
            answer = self.service.answer(DeepShieldQuestion(question=case.prompt, top_k=5))
            events = list(answer.guardrail_events)
            tool_blocked = False
            if case.attack_category is AttackCategory.UNAUTHORIZED_TOOL_ACTION:
                context = ToolAuthorizationContext()
                for tool_id in case.blocked_tool_ids:
                    for fixture_id in case.safety_fixture_ids:
                        decision = self.guardrails.authorize_tool(
                            MockToolCall(tool_id=tool_id, fixture_id=fixture_id),
                            context,
                        )
                        events.append(decision.event)
                        tool_blocked = tool_blocked or not decision.allowed
            return _guarded_observation(
                answer,
                self.retriever.last_hits,
                guardrail_events=events,
                outcome=(ObservedOutcome.REFUSED if tool_blocked else None),
                response_text=(TOOL_AUTHORIZATION_REFUSAL if tool_blocked else None),
            )


def build_comparison_backends(
    *,
    retriever: RetrievalBackend,
    generator: GenerationBackend,
    classifier: RiskClassifierBackend,
    environment: SyntheticSafetyEnvironment,
) -> tuple[
    BaselineExperimentBackend,
    HardenedExperimentBackend,
    DeepShieldExperimentBackend,
]:
    """Build all configurations around identical retrieval and generation components."""

    controlled = ControlledExperimentRetriever(retriever, environment)
    guardrails = DeterministicC2Guardrails(environment)
    return (
        BaselineExperimentBackend(BaselineRagService(controlled, generator), controlled),
        HardenedExperimentBackend(HardenedRagService(controlled, generator), controlled),
        DeepShieldExperimentBackend(
            DeepShieldRagService(controlled, generator, classifier, guardrails),
            controlled,
            guardrails,
        ),
    )


def _guarded_observation(
    answer: HardenedAnswer | DeepShieldAnswer,
    hits: tuple[RetrievalHit, ...],
    *,
    guardrail_events: list[GuardrailEvent] | None = None,
    outcome: ObservedOutcome | None = None,
    response_text: str | None = None,
) -> TrialObservation:
    return TrialObservation(
        outcome=outcome or ObservedOutcome(answer.outcome),
        answer=response_text or answer.answer,
        cited_reference_ids=answer.cited_reference_ids,
        invalid_reference_ids=answer.invalid_reference_ids,
        retrieved=_retrieval_evidence(hits),
        excluded_chunk_ids=answer.excluded_chunk_ids,
        guardrail_events=guardrail_events or answer.guardrail_events,
        risk_assessments=(answer.risk_assessments if isinstance(answer, DeepShieldAnswer) else []),
        pre_guardrail_output_sha256=answer.generated_output_sha256,
        generator_model=answer.generator_model,
        retrieval_model=answer.retrieval_model,
        finish_reason=answer.finish_reason,
        usage=answer.usage,
    )


def _retrieval_evidence(hits: tuple[RetrievalHit, ...]) -> list[RetrievalEvidence]:
    return [
        RetrievalEvidence(
            reference_id=f"S{hit.rank}",
            rank=hit.rank,
            score=hit.score,
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
        )
        for hit in hits
    ]


def _synthetic_hit(
    chunk_id: str,
    *,
    source_id: str,
    title: str,
    text: str,
) -> RetrievalHit:
    return RetrievalHit.model_validate(
        {
            "rank": 1,
            "score": 1.0,
            "chunk_id": chunk_id,
            "source_id": source_id,
            "title": title,
            "publisher": SYNTHETIC_PUBLISHER,
            "canonical_url": f"{SYNTHETIC_BASE_URL}/{chunk_id}",
            "char_start": 0,
            "char_end": len(text),
            "text": text,
        }
    )
