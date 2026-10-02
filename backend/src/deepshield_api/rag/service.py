"""C0 orchestration: retrieve official passages, prompt a model, and preserve evidence."""

import re
from collections.abc import Sequence
from typing import Protocol

from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.rag.generation import GenerationBackend
from deepshield_api.rag.models import (
    BaselineAnswer,
    BaselineQuestion,
    GenerationUsage,
    RetrievedPassage,
)

SYSTEM_PROMPT = """You are a baseline F-1 policy question-answering assistant.
Answer the user's question using only the retrieved official passages provided below.
Cite each factual policy claim with the relevant passage label, such as [S1].
If the passages do not contain enough information, say that the available sources are insufficient.
Keep the answer concise and include a reminder that it is general information, not legal advice."""
REFERENCE_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")


class RagServiceError(RuntimeError):
    """Raised when C0 cannot produce a response from valid runtime components."""


class RetrievalBackend(Protocol):
    """Interface for replacing local retrieval in tests and future experiments."""

    model_name: str

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        """Return ranked policy passages for one question."""


class BaselineRagService:
    """Run the documented C0 pipeline without additional security guardrails."""

    def __init__(self, retriever: RetrievalBackend, generator: GenerationBackend) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, request: BaselineQuestion) -> BaselineAnswer:
        """Retrieve context, generate an answer, and map every citation to evidence."""

        hits = self._retriever.retrieve(request.question, top_k=request.top_k)
        if not hits:
            raise RagServiceError("retrieval returned no policy passages")
        passages = [_to_passage(hit) for hit in hits]
        completion = self._generator.generate(
            SYSTEM_PROMPT,
            _build_user_prompt(request.question, passages),
        )
        cited, invalid = _extract_reference_ids(completion.text, len(passages))
        return BaselineAnswer(
            question=request.question,
            answer=completion.text,
            cited_reference_ids=cited,
            invalid_reference_ids=invalid,
            retrieved_passages=passages,
            generator_model=completion.model,
            retrieval_model=self._retriever.model_name,
            finish_reason=completion.finish_reason,
            usage=GenerationUsage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
            ),
        )


def _to_passage(hit: RetrievalHit) -> RetrievedPassage:
    return RetrievedPassage(
        reference_id=f"S{hit.rank}",
        rank=hit.rank,
        score=hit.score,
        chunk_id=hit.chunk_id,
        source_id=hit.source_id,
        title=hit.title,
        publisher=hit.publisher,
        canonical_url=hit.canonical_url,
        char_start=hit.char_start,
        char_end=hit.char_end,
        text=hit.text,
    )


def _build_user_prompt(question: str, passages: Sequence[RetrievedPassage]) -> str:
    context = "\n\n".join(
        (
            f"[{passage.reference_id}]\n"
            f"Title: {passage.title}\n"
            f"Publisher: {passage.publisher.value}\n"
            f"URL: {passage.canonical_url}\n"
            f"Passage:\n{passage.text}"
        )
        for passage in passages
    )
    return f"Question:\n{question}\n\nRetrieved official passages:\n{context}\n\nAnswer:"


def _extract_reference_ids(answer: str, passage_count: int) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    invalid: list[str] = []
    for number in REFERENCE_PATTERN.findall(answer):
        reference_id = f"S{number}"
        target = cited if int(number) <= passage_count else invalid
        if reference_id not in target:
            target.append(reference_id)
    return cited, invalid
