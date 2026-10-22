"""Runtime configuration and cached construction of C0, C1, and C2 RAG services."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deepshield_api.classifier.training import (
    DEFAULT_MODEL_ARTIFACT,
    DEFAULT_TRAINING_REPORT,
    ClassifierTrainingError,
    load_model_artifact,
    load_training_report,
    validate_training_report,
)
from deepshield_api.corpus.indexing import (
    DEFAULT_MODEL_NAME,
    FastEmbedBackend,
    IndexingError,
    RetrievalHit,
    search_vector_index,
)
from deepshield_api.evaluation.adapters import (
    BaselineExperimentBackend,
    DeepShieldExperimentBackend,
    HardenedExperimentBackend,
    build_comparison_backends,
)
from deepshield_api.evaluation.synthetic import (
    SyntheticEnvironmentError,
    SyntheticSafetyEnvironment,
    load_synthetic_environment,
)
from deepshield_api.rag.c2_guardrails import DeterministicC2Guardrails
from deepshield_api.rag.deepshield import DeepShieldRagService, PortableRiskClassifier
from deepshield_api.rag.generation import (
    MaxTokenParameter,
    OpenAICompatibleGenerator,
    ReasoningEffort,
)
from deepshield_api.rag.hardened import HardenedRagService
from deepshield_api.rag.service import BaselineRagService


class RagRuntimeError(RuntimeError):
    """Raised when a RAG service cannot be configured from the local environment."""


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
    classifier_training_report: Path = DEFAULT_TRAINING_REPORT
    classifier_model_artifact: Path = DEFAULT_MODEL_ARTIFACT
    synthetic_safety_environment: Path = Path(
        "data/evaluation/synthetic-safety-environment.v1.json"
    )


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
    retriever, generator = _build_runtime_components(resolved)
    return BaselineRagService(retriever, generator)


def build_hardened_rag_service(settings: RagSettings | None = None) -> HardenedRagService:
    """Build one reusable C1 service with the same retrieval and generation models as C0."""

    resolved = settings or RagSettings()
    retriever, generator = _build_runtime_components(resolved)
    return HardenedRagService(retriever, generator)


def build_deepshield_rag_service(settings: RagSettings | None = None) -> DeepShieldRagService:
    """Build C2 only from verified reviewed classifier and synthetic safety artifacts."""

    resolved = settings or RagSettings()
    classifier, environment = _load_c2_components(resolved)
    retriever, generator = _build_runtime_components(resolved)
    return DeepShieldRagService(
        retriever,
        generator,
        classifier,
        DeterministicC2Guardrails(environment),
    )


def build_experiment_comparison_backends(
    settings: RagSettings | None = None,
) -> tuple[
    BaselineExperimentBackend,
    HardenedExperimentBackend,
    DeepShieldExperimentBackend,
]:
    """Build C0/C1/C2 adapters from one verified retriever and generator instance."""

    resolved = settings or RagSettings()
    classifier, environment = _load_c2_components(resolved)
    retriever, generator = _build_runtime_components(resolved)
    return build_comparison_backends(
        retriever=retriever,
        generator=generator,
        classifier=classifier,
        environment=environment,
    )


def _build_runtime_components(
    resolved: RagSettings,
) -> tuple[LocalCorpusRetriever, OpenAICompatibleGenerator]:
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
    return retriever, generator


def _load_c2_components(
    resolved: RagSettings,
) -> tuple[PortableRiskClassifier, SyntheticSafetyEnvironment]:
    root = resolved.repository_root.resolve()
    training_report_path = _under_root(resolved.classifier_training_report, root)
    model_path = _under_root(resolved.classifier_model_artifact, root)
    environment_path = _under_root(resolved.synthetic_safety_environment, root)
    try:
        report = load_training_report(training_report_path)
        validate_training_report(report, root, require_artifacts=True)
        if (root / report.model_artifact_path).resolve() != model_path:
            raise RagRuntimeError("configured classifier model does not match training evidence")
        artifact = load_model_artifact(model_path)
        environment = load_synthetic_environment(environment_path)
    except RagRuntimeError:
        raise
    except (ClassifierTrainingError, SyntheticEnvironmentError, OSError, ValueError) as exc:
        raise RagRuntimeError("unable to initialize verified C2 experiment artifacts") from exc
    return PortableRiskClassifier(artifact), environment


@lru_cache(maxsize=1)
def get_baseline_rag_service() -> BaselineRagService:
    """Reuse model and index resources instead of loading them for every request."""

    return build_baseline_rag_service()


@lru_cache(maxsize=1)
def get_hardened_rag_service() -> HardenedRagService:
    """Reuse the C1 model and index resources instead of loading them per request."""

    return build_hardened_rag_service()


@lru_cache(maxsize=1)
def get_deepshield_rag_service() -> DeepShieldRagService:
    """Reuse verified C2 model, index, and safety resources across requests."""

    return build_deepshield_rag_service()


def clear_baseline_rag_service_cache() -> None:
    """Reset runtime state for tests or deliberate local configuration changes."""

    get_baseline_rag_service.cache_clear()


def clear_hardened_rag_service_cache() -> None:
    """Reset cached C1 runtime state for tests or configuration changes."""

    get_hardened_rag_service.cache_clear()


def clear_deepshield_rag_service_cache() -> None:
    """Reset cached C2 runtime state after deliberate artifact changes."""

    get_deepshield_rag_service.cache_clear()


def _under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise RagRuntimeError("runtime artifact paths must stay inside the repository root")
    return resolved
