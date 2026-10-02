from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from deepshield_api import __version__
from deepshield_api.rag.router import router as assistant_router


class HealthResponse(BaseModel):
    """Response returned by the service health endpoint."""

    status: Literal["ok"]
    service: str
    version: str


def create_app() -> FastAPI:
    """Create and configure the DeepShield-RAG FastAPI application."""

    application = FastAPI(
        title="DeepShield-RAG API",
        description="Backend API for RAG security evaluation and guardrail benchmarking.",
        version=__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    application.include_router(assistant_router)

    @application.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Check API health",
    )
    async def health_check() -> HealthResponse:
        """Return basic service availability and version information."""

        return HealthResponse(
            status="ok",
            service="deepshield-rag-api",
            version=__version__,
        )

    return application


app = create_app()
