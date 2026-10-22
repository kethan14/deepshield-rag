"""Frozen-benchmark execution records for reproducible C0/C1/C2 comparisons."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from deepshield_api.corpus.models import Sha256Digest, Slug
from deepshield_api.evaluation.manifest import (
    BenchmarkManifestError,
    load_benchmark_manifest,
    validate_corpus_references,
)
from deepshield_api.evaluation.models import (
    AttackCategory,
    BenchmarkCase,
    BenchmarkDifficulty,
    BenchmarkKind,
    BenchmarkManifest,
    BenchmarkSplit,
    BenchmarkStatus,
    ExpectedAction,
)
from deepshield_api.rag.experiment import (
    ExperimentConfigurationError,
    FixedGenerationConfiguration,
    load_fixed_generation_configuration,
)
from deepshield_api.rag.models import (
    ClassifierRiskAssessment,
    GenerationUsage,
    GuardrailEvent,
)


class ExperimentRunError(RuntimeError):
    """Raised when an experiment cannot be executed or its evidence is invalid."""


class ExperimentConfiguration(StrEnum):
    """Comparable system configurations in the preregistered execution order."""

    BASELINE = "C0"
    HARDENED = "C1"
    DEEPSHIELD = "C2"


class ExperimentPurpose(StrEnum):
    """Run purpose that prevents development work from touching the held-out test split."""

    DEVELOPMENT = "development"
    FINAL = "final"

    @property
    def benchmark_split(self) -> BenchmarkSplit:
        return (
            BenchmarkSplit.DEVELOPMENT
            if self is ExperimentPurpose.DEVELOPMENT
            else BenchmarkSplit.TEST
        )


class ObservedOutcome(StrEnum):
    """Coarse behavior directly observable from one assistant configuration."""

    ANSWERED = "answered"
    REFUSED = "refused"


class RetrievalEvidence(BaseModel):
    """Compact retrieval evidence sufficient for recall and ranking analysis."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    rank: int = Field(ge=1)
    score: float = Field(ge=-1.000001, le=1.000001, allow_inf_nan=False)
    chunk_id: str = Field(min_length=1)
    source_id: Slug


class TrialObservation(BaseModel):
    """Normalized raw evidence returned by one configuration adapter."""

    model_config = ConfigDict(extra="forbid")

    outcome: ObservedOutcome
    answer: str = Field(min_length=1)
    cited_reference_ids: list[str] = Field(default_factory=list)
    invalid_reference_ids: list[str] = Field(default_factory=list)
    retrieved: list[RetrievalEvidence] = Field(default_factory=list)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    risk_assessments: list[ClassifierRiskAssessment] = Field(default_factory=list)
    pre_guardrail_output_sha256: Sha256Digest | None = None
    generator_model: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: GenerationUsage = Field(default_factory=GenerationUsage)

    @field_validator(
        "cited_reference_ids",
        "invalid_reference_ids",
        "excluded_chunk_ids",
    )
    @classmethod
    def require_unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("trial evidence lists cannot contain duplicates")
        return value

    @field_validator("answer", "generator_model", "retrieval_model")
    @classmethod
    def reject_blank_text_without_altering_raw_evidence(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trial text values cannot be blank")
        return value

    @model_validator(mode="after")
    def require_consistent_evidence(self) -> Self:
        reference_ids = [item.reference_id for item in self.retrieved]
        chunk_ids = [item.chunk_id for item in self.retrieved]
        ranks = [item.rank for item in self.retrieved]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("retrieved reference IDs must be unique")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieved chunk IDs must be unique")
        if len(ranks) != len(set(ranks)) or ranks != sorted(ranks):
            raise ValueError("retrieved evidence ranks must be unique and ordered")
        if set(self.cited_reference_ids) - set(reference_ids):
            raise ValueError("cited references must exist in retrieved evidence")
        if set(self.cited_reference_ids) & set(self.invalid_reference_ids):
            raise ValueError("a citation cannot be both valid and invalid")
        return self


class ExperimentTrial(BaseModel):
    """One immutable case/configuration result with benchmark labels and raw evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    trial_id: str = Field(
        min_length=1,
        max_length=180,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    sequence: int = Field(ge=1)
    case_id: Slug
    configuration: ExperimentConfiguration
    split: BenchmarkSplit
    kind: BenchmarkKind
    attack_category: AttackCategory | None = None
    difficulty: BenchmarkDifficulty
    expected_action: ExpectedAction
    prompt_sha256: Sha256Digest
    output_sha256: Sha256Digest
    duration_ms: int = Field(ge=0)
    observation: TrialObservation

    @model_validator(mode="after")
    def require_matching_output_hash(self) -> Self:
        if self.output_sha256 != _sha256_text(self.observation.answer):
            raise ValueError("output_sha256 does not match the recorded answer")
        return self


class ExperimentRunManifest(BaseModel):
    """Provenance and integrity metadata for one complete local experiment run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: Slug
    status: Literal["completed"] = "completed"
    purpose: ExperimentPurpose
    benchmark_split: BenchmarkSplit
    benchmark_id: str = Field(min_length=1)
    benchmark_version: int = Field(ge=1)
    benchmark_sha256: Sha256Digest
    generation_configuration_id: str = Field(min_length=1)
    generation_configuration_sha256: Sha256Digest
    generator_model: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    retrieval_top_k: Literal[5] = 5
    configurations: list[ExperimentConfiguration]
    started_at: datetime
    completed_at: datetime
    case_count: int = Field(ge=1)
    trial_count: int = Field(ge=1)
    results_file: str = Field(default="trials.jsonl", pattern=r"^trials\.jsonl$")
    results_sha256: Sha256Digest

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("experiment timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def require_complete_comparison(self) -> Self:
        expected = list(ExperimentConfiguration)
        if self.configurations != expected:
            raise ValueError("configurations must be C0, C1, and C2 in comparison order")
        if self.benchmark_split is not self.purpose.benchmark_split:
            raise ValueError("benchmark split must match the declared run purpose")
        if self.trial_count != self.case_count * len(expected):
            raise ValueError("trial count must contain one result per case and configuration")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        return self


class ExperimentBackend(Protocol):
    """Adapter contract for executing exactly one system configuration."""

    configuration: ExperimentConfiguration
    retrieval_top_k: Literal[5]

    def run(self, case: BenchmarkCase) -> TrialObservation:
        """Execute one reviewed benchmark case and return normalized evidence."""


@dataclass(frozen=True)
class CompletedExperimentRun:
    """In-memory complete run ready for checksum-backed persistence."""

    manifest: ExperimentRunManifest
    trials: tuple[ExperimentTrial, ...]
    results_jsonl: str


@dataclass(frozen=True)
class PreparedExperimentInputs:
    """Validated immutable inputs that are safe to inspect before runtime construction."""

    repository_root: Path
    benchmark_path: Path
    generation_configuration_path: Path
    benchmark: BenchmarkManifest
    generation: FixedGenerationConfiguration
    purpose: ExperimentPurpose
    selected_case_count: int


Clock = Callable[[], datetime]
Timer = Callable[[], float]


def execute_frozen_experiment(
    *,
    repository_root: Path,
    benchmark_path: Path,
    generation_configuration_path: Path,
    purpose: ExperimentPurpose,
    run_id: str,
    backends: Sequence[ExperimentBackend],
    clock: Clock | None = None,
    timer: Timer | None = None,
) -> CompletedExperimentRun:
    """Execute one split only after all benchmark and configuration gates pass."""

    prepared = prepare_experiment_inputs(
        repository_root=repository_root,
        benchmark_path=benchmark_path,
        generation_configuration_path=generation_configuration_path,
        purpose=purpose,
    )
    benchmark_file = prepared.benchmark_path
    generation_file = prepared.generation_configuration_path
    benchmark = prepared.benchmark
    generation = prepared.generation

    validate_experiment_run_id(run_id)
    ordered_backends = _validate_backends(backends, generation)
    selected_cases = [case for case in benchmark.cases if case.split is purpose.benchmark_split]

    now = clock or (lambda: datetime.now(UTC))
    monotonic = timer or perf_counter
    started_at = now()
    if started_at.utcoffset() is None:
        raise ExperimentRunError("experiment clock must return timezone-aware timestamps")
    trials: list[ExperimentTrial] = []
    sequence = 1
    for case in selected_cases:
        for backend in ordered_backends:
            began = monotonic()
            observation = backend.run(case)
            elapsed_ms = max(0, round((monotonic() - began) * 1_000))
            if observation.generator_model != generation.model:
                raise ExperimentRunError(
                    "an experiment observation used a generator other than the frozen model"
                )
            trials.append(
                ExperimentTrial(
                    trial_id=f"{run_id}-{case.case_id}-{backend.configuration.value.lower()}",
                    sequence=sequence,
                    case_id=case.case_id,
                    configuration=backend.configuration,
                    split=case.split,
                    kind=case.kind,
                    attack_category=case.attack_category,
                    difficulty=case.difficulty,
                    expected_action=case.expected_action,
                    prompt_sha256=_sha256_text(case.prompt),
                    output_sha256=_sha256_text(observation.answer),
                    duration_ms=elapsed_ms,
                    observation=observation,
                )
            )
            sequence += 1

    results_jsonl = render_trials(trials)
    completed_at = now()
    retrieval_models = {trial.observation.retrieval_model for trial in trials}
    if len(retrieval_models) != 1:
        raise ExperimentRunError("all configurations must use the same retrieval model")
    manifest = ExperimentRunManifest(
        run_id=run_id,
        purpose=purpose,
        benchmark_split=purpose.benchmark_split,
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.manifest_version,
        benchmark_sha256=_sha256_file(benchmark_file),
        generation_configuration_id=generation.configuration_id,
        generation_configuration_sha256=_sha256_file(generation_file),
        generator_model=generation.model,
        retrieval_model=retrieval_models.pop(),
        configurations=list(ExperimentConfiguration),
        started_at=started_at,
        completed_at=completed_at,
        case_count=len(selected_cases),
        trial_count=len(trials),
        results_sha256=_sha256_text(results_jsonl),
    )
    _validate_trial_matrix(manifest, trials)
    return CompletedExperimentRun(manifest, tuple(trials), results_jsonl)


def prepare_experiment_inputs(
    *,
    repository_root: Path,
    benchmark_path: Path,
    generation_configuration_path: Path,
    purpose: ExperimentPurpose,
) -> PreparedExperimentInputs:
    """Validate all static experiment gates before models or indexes are initialized."""

    root = repository_root.resolve()
    benchmark_file = _resolve(root, benchmark_path)
    generation_file = _resolve(root, generation_configuration_path)
    try:
        benchmark = load_benchmark_manifest(benchmark_file)
        generation = load_fixed_generation_configuration(generation_file)
        validate_corpus_references(benchmark, root)
    except (BenchmarkManifestError, ExperimentConfigurationError) as exc:
        raise ExperimentRunError(str(exc)) from exc
    if benchmark.status is not BenchmarkStatus.FROZEN:
        raise ExperimentRunError("benchmark must be frozen before any experiment is executed")
    selected_case_count = sum(case.split is purpose.benchmark_split for case in benchmark.cases)
    if selected_case_count == 0:
        raise ExperimentRunError("the selected benchmark split contains no cases")
    return PreparedExperimentInputs(
        repository_root=root,
        benchmark_path=benchmark_file,
        generation_configuration_path=generation_file,
        benchmark=benchmark,
        generation=generation,
        purpose=purpose,
        selected_case_count=selected_case_count,
    )


def render_trials(trials: Sequence[ExperimentTrial]) -> str:
    """Render canonical JSON Lines without silently reordering trial execution."""

    return "".join(
        json.dumps(trial.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for trial in trials
    )


def render_run_manifest(manifest: ExperimentRunManifest) -> str:
    """Render one canonical human-readable run manifest."""

    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_completed_run(runs_root: Path, completed: CompletedExperimentRun) -> Path:
    """Write a complete run to a new directory and never replace prior evidence."""

    run_directory = runs_root.resolve() / completed.manifest.run_id
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
        (run_directory / completed.manifest.results_file).write_text(
            completed.results_jsonl,
            encoding="utf-8",
            newline="\n",
        )
        (run_directory / "run-manifest.json").write_text(
            render_run_manifest(completed.manifest),
            encoding="utf-8",
            newline="\n",
        )
    except FileExistsError as exc:
        raise ExperimentRunError(
            f"refusing to replace existing experiment run: {run_directory}"
        ) from exc
    except OSError as exc:
        raise ExperimentRunError(f"unable to write experiment run: {run_directory}") from exc
    return run_directory


def load_completed_run(run_directory: Path) -> CompletedExperimentRun:
    """Load a completed run and verify its result checksum, count, and ordering."""

    try:
        manifest = ExperimentRunManifest.model_validate_json(
            (run_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        results_jsonl = (run_directory / manifest.results_file).read_text(encoding="utf-8")
        trials = tuple(
            ExperimentTrial.model_validate_json(line)
            for line in results_jsonl.splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as exc:
        raise ExperimentRunError(f"unable to load experiment run: {run_directory}") from exc
    if _sha256_text(results_jsonl) != manifest.results_sha256:
        raise ExperimentRunError("experiment result checksum does not match the run manifest")
    if len(trials) != manifest.trial_count:
        raise ExperimentRunError("experiment result count does not match the run manifest")
    if [trial.sequence for trial in trials] != list(range(1, len(trials) + 1)):
        raise ExperimentRunError("experiment trials are not in contiguous execution order")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ExperimentRunError("experiment results contain duplicate trial IDs")
    _validate_trial_matrix(manifest, trials)
    return CompletedExperimentRun(manifest, trials, results_jsonl)


def _validate_backends(
    backends: Sequence[ExperimentBackend],
    generation: FixedGenerationConfiguration,
) -> tuple[ExperimentBackend, ...]:
    expected = list(ExperimentConfiguration)
    received = [backend.configuration for backend in backends]
    configured = [ExperimentConfiguration(item) for item in generation.configurations]
    if configured != expected or received != expected:
        raise ExperimentRunError("backends must contain C0, C1, and C2 exactly once in order")
    if any(backend.retrieval_top_k != 5 for backend in backends):
        raise ExperimentRunError("every experiment backend must use retrieval top_k=5")
    return tuple(backends)


def validate_experiment_run_id(run_id: str) -> None:
    """Reject unsafe run-directory identifiers before any model is initialized."""

    try:
        TypeAdapter(Slug).validate_python(run_id)
    except ValidationError as exc:
        raise ExperimentRunError("run_id must be a lowercase hyphenated identifier") from exc


def _validate_trial_matrix(
    manifest: ExperimentRunManifest,
    trials: Sequence[ExperimentTrial],
) -> None:
    configurations = list(ExperimentConfiguration)
    case_ids: list[str] = []
    for offset in range(0, len(trials), len(configurations)):
        group = trials[offset : offset + len(configurations)]
        if len(group) != len(configurations):
            raise ExperimentRunError("experiment results contain an incomplete case comparison")
        if [trial.configuration for trial in group] != configurations:
            raise ExperimentRunError("experiment configurations are not in C0, C1, C2 order")
        if len({trial.case_id for trial in group}) != 1:
            raise ExperimentRunError("a case comparison contains mismatched case IDs")
        if any(trial.split is not manifest.benchmark_split for trial in group):
            raise ExperimentRunError("a trial belongs to the wrong benchmark split")
        case_ids.append(group[0].case_id)
    if len(case_ids) != manifest.case_count or len(set(case_ids)) != len(case_ids):
        raise ExperimentRunError(
            "experiment results do not contain each selected case exactly once"
        )


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ExperimentRunError(f"unable to hash experiment input: {path}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
