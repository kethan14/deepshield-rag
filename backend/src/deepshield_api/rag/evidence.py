"""Shared evidence mapping for comparable RAG configurations."""

import re

from deepshield_api.corpus.indexing import RetrievalHit
from deepshield_api.rag.generation import GenerationResult
from deepshield_api.rag.models import GenerationUsage, RetrievedPassage

REFERENCE_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")


def retrieval_hit_to_passage(hit: RetrievalHit) -> RetrievedPassage:
    """Convert one ranked retrieval result into API evidence."""

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


def extract_reference_ids(
    answer: str, valid_reference_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Return unique valid and invalid citations in first-occurrence order."""

    cited: list[str] = []
    invalid: list[str] = []
    for number in REFERENCE_PATTERN.findall(answer):
        reference_id = f"S{number}"
        target = cited if reference_id in valid_reference_ids else invalid
        if reference_id not in target:
            target.append(reference_id)
    return cited, invalid


def generation_usage(completion: GenerationResult | None) -> GenerationUsage:
    """Map optional provider usage without fabricating unavailable measurements."""

    if completion is None:
        return GenerationUsage()
    return GenerationUsage(
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
    )
