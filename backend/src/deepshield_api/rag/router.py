"""FastAPI routes for the comparable C0 and C1 policy assistants."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from deepshield_api.corpus.indexing import IndexingError
from deepshield_api.rag.generation import GenerationError
from deepshield_api.rag.hardened import HardenedRagService
from deepshield_api.rag.models import (
    BaselineAnswer,
    BaselineQuestion,
    HardenedAnswer,
    HardenedQuestion,
)
from deepshield_api.rag.runtime import (
    RagRuntimeError,
    get_baseline_rag_service,
    get_hardened_rag_service,
)
from deepshield_api.rag.service import BaselineRagService, RagServiceError

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


def baseline_service_dependency() -> BaselineRagService:
    """Resolve the cached service while keeping configuration errors server-side."""

    try:
        return get_baseline_rag_service()
    except (RagRuntimeError, IndexingError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The baseline assistant is not configured or its index is unavailable.",
        ) from exc


def hardened_service_dependency() -> HardenedRagService:
    """Resolve the cached C1 service without exposing configuration details."""

    try:
        return get_hardened_rag_service()
    except (RagRuntimeError, IndexingError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The hardened assistant is not configured or its index is unavailable.",
        ) from exc


@router.post(
    "/query",
    response_model=BaselineAnswer,
    summary="Answer an F-1 policy question with the C0 baseline RAG pipeline",
)
async def answer_policy_question(
    request: BaselineQuestion,
    service: Annotated[BaselineRagService, Depends(baseline_service_dependency)],
) -> BaselineAnswer:
    """Run blocking local retrieval and generation outside the async event loop."""

    try:
        return await run_in_threadpool(service.answer, request)
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The configured generation endpoint failed.",
        ) from exc
    except (IndexingError, RagServiceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The baseline assistant could not complete the request.",
        ) from exc


@router.post(
    "/query/c1",
    response_model=HardenedAnswer,
    summary="Answer an F-1 policy question with the C1 hardened RAG pipeline",
)
async def answer_hardened_policy_question(
    request: HardenedQuestion,
    service: Annotated[HardenedRagService, Depends(hardened_service_dependency)],
) -> HardenedAnswer:
    """Run C1 screening, retrieval, generation, and output validation off the event loop."""

    try:
        return await run_in_threadpool(service.answer, request)
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The configured generation endpoint failed.",
        ) from exc
    except (IndexingError, RagServiceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The hardened assistant could not complete the request.",
        ) from exc
