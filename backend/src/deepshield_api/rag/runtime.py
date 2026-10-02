"""Runtime configuration and cached construction of the local C0 RAG service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deepshield_api.corpus.indexing import (
    DEFAULT_MODEL_NAME,
    FastEmbedBackend,
    IndexingError,
    RetrievalHit,
    search_vector_index,
)
from deepshield_api.rag.generation import (
    MaxTokenParameter,
    OpenAICompatibleGenerator,
    ReasoningEffort,
)
from deepshield_api.rag.service import BaselineRagService


class RagRuntimeError(RuntimeError):
    """Raised when the C0 service cannot be configured from the local environment."""


class RagSettings(BaseSettings):
    """Server-only configuration for retrieval and model generation."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSHIELD_",
        env_file=".env",
        extra="ignore",
    )

    repository_root: Path = Path.cwd()
    index_report: Path = Path("data/manifests/f1-policy-vector-index-report.v1.json")
    model_cache: Path = Path("data/indexes/models")
    embedding_threads: int = Field(default=1, ge=1, le=32)
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    llm_max_output_tokens: int = Field(default=700, ge=1, le=4_096)
    llm_reasoning_effort: ReasoningEffort | None = None
    llm_max_token_parameter: MaxTokenParameter = "max_tokens"


class LocalCorpusRetriever:
    """Adapter from the verified vector index to the RAG retrieval interface."""

    model_name = DEFAULT_MODEL_NAME

    def __init__(self, settings: RagSettings) -> None:
        self._repository_root = settings.repository_root.resolve()
        self._index_report = _under_root(settings.index_report, self._repository_root)
        model_cache = _under_root(settings.model_cache, self._repository_root)
        self._embedder = FastEmbedBackend(model_cache, threads=settings.embedding_threads)

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        return search_vector_index(
            question,
            self._index_report,
            self._repository_root,
            self._embedder,
            top_k=top_k,
        )


def build_baseline_rag_service(settings: RagSettings | None = None) -> BaselineRagService:
    """Build one reusable C0 service from validated server-side settings."""

    resolved = settings or RagSettings()
    if resolved.llm_model is None or not resolved.llm_model.strip():
        raise RagRuntimeError("DEEPSHIELD_LLM_MODEL is required to enable answer generation")
    generator = OpenAICompatibleGenerator(
        base_url=resolved.llm_base_url,
        model=resolved.llm_model,
        api_key=(
            resolved.llm_api_key.get_secret_value() if resolved.llm_api_key is not None else None
        ),
        timeout_seconds=resolved.llm_timeout_seconds,
        max_output_tokens=resolved.llm_max_output_tokens,
        reasoning_effort=resolved.llm_reasoning_effort,
        max_token_parameter=resolved.llm_max_token_parameter,
    )
    try:
        retriever = LocalCorpusRetriever(resolved)
    except RagRuntimeError:
        raise
    except (IndexingError, OSError, ValueError) as exc:
        raise RagRuntimeError("unable to initialize the verified local retriever") from exc
    return BaselineRagService(retriever, generator)


@lru_cache(maxsize=1)
def get_baseline_rag_service() -> BaselineRagService:
    """Reuse model and index resources instead of loading them for every request."""

    return build_baseline_rag_service()


def clear_baseline_rag_service_cache() -> None:
    """Reset runtime state for tests or deliberate local configuration changes."""

    get_baseline_rag_service.cache_clear()


def _under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise RagRuntimeError("runtime artifact paths must stay inside the repository root")
    return resolved
