"""Train and validate the lightweight C2 text-risk classifier."""

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from deepshield_api.classifier.freezing import load_and_validate_review_report
from deepshield_api.classifier.manifest import (
    load_risk_dataset,
    validate_dataset_references,
)
from deepshield_api.classifier.models import (
    ArtifactStatus,
    BinaryClassificationMetrics,
    ClassifierTrainingConfig,
    ClassifierTrainingReport,
    RiskClassifierArtifact,
    RiskDatasetManifest,
    RiskExample,
    RiskLabel,
    RiskSplit,
)

DEFAULT_REVIEW_REPORT = Path("data/classifier/review-report.v1.json")
DEFAULT_FROZEN_DATASET = Path("data/classifier/processed/risk-dataset-reviewed.v1.json")
DEFAULT_MODEL_ARTIFACT = Path("data/classifier/models/risk-classifier.v1.json")
DEFAULT_TRAINING_REPORT = Path("data/classifier/training-report.v1.json")
ModelType = TypeVar("ModelType", bound=BaseModel)


class ClassifierTrainingError(ValueError):
    """Raised when training inputs or evidence are unsafe or inconsistent."""


@dataclass(frozen=True)
class ClassifierTrainingResult:
    """Local portable model plus aggregate, committable training evidence."""

    artifact: RiskClassifierArtifact
    report: ClassifierTrainingReport


def train_risk_classifier(
    repository_root: Path,
    *,
    created_at: datetime | None = None,
) -> ClassifierTrainingResult:
    """Train only from the reviewed frozen dataset and validation-selected threshold."""

    root = repository_root.resolve()
    review_path = root / DEFAULT_REVIEW_REPORT
    dataset_path = root / DEFAULT_FROZEN_DATASET
    try:
        review_bytes = review_path.read_bytes()
        dataset_bytes = dataset_path.read_bytes()
    except OSError as exc:
        raise ClassifierTrainingError(
            f"unable to read classifier training input: {exc.filename}"
        ) from exc

    review = load_and_validate_review_report(review_path, root, require_artifacts=True)
    if review.output_dataset_path != DEFAULT_FROZEN_DATASET.as_posix():
        raise ClassifierTrainingError(
            "review report does not reference the frozen training dataset"
        )
    dataset_sha256 = _sha256(dataset_bytes)
    if review.output_dataset_sha256 != dataset_sha256:
        raise ClassifierTrainingError("frozen dataset checksum does not match review evidence")
    dataset = load_risk_dataset(dataset_path)
    validate_dataset_references(dataset, root)
    if dataset.status is not ArtifactStatus.FROZEN:
        raise ClassifierTrainingError("classifier training requires a frozen reviewed dataset")

    result = _fit_classifier(
        dataset,
        dataset_sha256=dataset_sha256,
        review_report_sha256=_sha256(review_bytes),
        created_at=created_at or datetime.now(UTC),
    )
    if review_path.read_bytes() != review_bytes or dataset_path.read_bytes() != dataset_bytes:
        raise ClassifierTrainingError("classifier training inputs changed during fitting")
    return result


def _fit_classifier(
    dataset: RiskDatasetManifest,
    *,
    dataset_sha256: str,
    review_report_sha256: str,
    created_at: datetime,
) -> ClassifierTrainingResult:
    config = ClassifierTrainingConfig()
    train = _examples_for_split(dataset, RiskSplit.TRAIN)
    validation = _examples_for_split(dataset, RiskSplit.VALIDATION)
    test = _examples_for_split(dataset, RiskSplit.TEST)
    external = _examples_for_split(dataset, RiskSplit.EXTERNAL_TEST)

    vectorizer = _new_vectorizer(config)
    train_matrix = vectorizer.fit_transform([item.text for item in train])
    train_labels = _labels(train)
    classifier = LogisticRegression(
        C=config.regularization_c,
        class_weight=config.class_weight,
        l1_ratio=config.l1_ratio,
        random_state=config.random_seed,
        solver=config.solver,
        max_iter=config.maximum_iterations,
    )
    classifier.fit(train_matrix, train_labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise ClassifierTrainingError("classifier learned an unexpected class order")

    validation_probabilities = _positive_probabilities(
        classifier,
        vectorizer.transform([item.text for item in validation]),
    )
    threshold = _select_threshold(_labels(validation), validation_probabilities, config)
    test_probabilities = _positive_probabilities(
        classifier,
        vectorizer.transform([item.text for item in test]),
    )
    external_probabilities = _positive_probabilities(
        classifier,
        vectorizer.transform([item.text for item in external]),
    )

    features = vectorizer.get_feature_names_out().tolist()
    artifact = RiskClassifierArtifact(
        model_version=1,
        created_at=created_at,
        training_dataset_sha256=dataset_sha256,
        config=config,
        threshold=threshold,
        features=features,
        inverse_document_frequency=vectorizer.idf_.tolist(),
        coefficients=classifier.coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]),
    )
    artifact_content = render_model_artifact(artifact)
    report = ClassifierTrainingReport(
        report_version=1,
        created_at=created_at,
        review_report_path="data/classifier/review-report.v1.json",
        review_report_sha256=review_report_sha256,
        input_dataset_path="data/classifier/processed/risk-dataset-reviewed.v1.json",
        input_dataset_sha256=dataset_sha256,
        model_artifact_path="data/classifier/models/risk-classifier.v1.json",
        model_artifact_sha256=_sha256(artifact_content.encode()),
        config=config,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scikit_learn_version=_scikit_learn_version(),
        training_count=len(train),
        validation_count=len(validation),
        selected_threshold=threshold,
        validation_metrics=_metrics(
            RiskSplit.VALIDATION,
            _labels(validation),
            validation_probabilities,
            threshold,
        ),
        test_metrics=_metrics(RiskSplit.TEST, _labels(test), test_probabilities, threshold),
        external_test_metrics=_metrics(
            RiskSplit.EXTERNAL_TEST,
            _labels(external),
            external_probabilities,
            threshold,
        ),
    )
    return ClassifierTrainingResult(artifact=artifact, report=report)


def predict_risk_probabilities(
    artifact: RiskClassifierArtifact,
    texts: list[str],
) -> NDArray[np.float64]:
    """Score text using the portable JSON artifact without executable serialization."""

    if not texts:
        return np.asarray([], dtype=np.float64)
    vocabulary = {feature: index for index, feature in enumerate(artifact.features)}
    vectorizer = _new_vectorizer(artifact.config, vocabulary=vocabulary)
    vectorizer.fit(["deepshield portable model bootstrap"])
    vectorizer.idf_ = np.asarray(artifact.inverse_document_frequency, dtype=np.float64)
    matrix = vectorizer.transform(texts)
    coefficients = np.asarray(artifact.coefficients, dtype=np.float64)
    scores = np.asarray(matrix @ coefficients).reshape(-1) + artifact.intercept
    return _sigmoid(scores)


def write_training_result(
    result: ClassifierTrainingResult,
    repository_root: Path,
    report_path: Path,
    model_path: Path,
) -> None:
    """Write versioned model and evidence with no overwrite or partial replacement."""

    root = repository_root.resolve()
    expected_report = (root / DEFAULT_TRAINING_REPORT).resolve()
    expected_model = (root / DEFAULT_MODEL_ARTIFACT).resolve()
    if report_path.resolve() != expected_report or model_path.resolve() != expected_model:
        raise ClassifierTrainingError("training outputs must use the registered versioned paths")
    model_content = render_model_artifact(result.artifact)
    report_content = render_training_report(result.report)
    if _sha256(model_content.encode()) != result.report.model_artifact_sha256:
        raise ClassifierTrainingError("model artifact checksum does not match training report")
    _require_new_target(expected_model)
    _require_new_target(expected_report)
    expected_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        with expected_model.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(model_content)
        with expected_report.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report_content)
    except OSError as exc:
        if expected_model.exists() and not expected_report.exists():
            expected_model.unlink()
        raise ClassifierTrainingError(
            f"unable to write classifier training output: {exc.filename}"
        ) from exc


def validate_training_report(
    report: ClassifierTrainingReport,
    repository_root: Path,
    *,
    require_artifacts: bool = False,
) -> None:
    """Validate report provenance and, when available, recompute every reported metric."""

    root = repository_root.resolve()
    review_path = root / report.review_report_path
    try:
        review_bytes = review_path.read_bytes()
    except OSError as exc:
        raise ClassifierTrainingError(f"unable to read review report: {review_path}") from exc
    if _sha256(review_bytes) != report.review_report_sha256:
        raise ClassifierTrainingError("review report checksum does not match training report")
    review = load_and_validate_review_report(review_path, root, require_artifacts=require_artifacts)
    if (
        review.output_dataset_path != report.input_dataset_path
        or review.output_dataset_sha256 != report.input_dataset_sha256
    ):
        raise ClassifierTrainingError("training input does not match review evidence")

    dataset_path = root / report.input_dataset_path
    model_path = root / report.model_artifact_path
    artifacts_exist = dataset_path.exists() and model_path.exists()
    if dataset_path.exists() != model_path.exists():
        raise ClassifierTrainingError("local classifier training artifacts are incomplete")
    if not artifacts_exist:
        if require_artifacts:
            raise ClassifierTrainingError(
                "required local classifier training artifacts are missing"
            )
        return
    dataset_bytes = dataset_path.read_bytes()
    model_bytes = model_path.read_bytes()
    if _sha256(dataset_bytes) != report.input_dataset_sha256:
        raise ClassifierTrainingError("training dataset checksum does not match")
    if _sha256(model_bytes) != report.model_artifact_sha256:
        raise ClassifierTrainingError("model artifact checksum does not match")
    dataset = load_risk_dataset(dataset_path)
    validate_dataset_references(dataset, root)
    artifact = load_model_artifact(model_path)
    if artifact.training_dataset_sha256 != report.input_dataset_sha256:
        raise ClassifierTrainingError("model artifact references a different training dataset")
    if artifact.config != report.config or artifact.threshold != report.selected_threshold:
        raise ClassifierTrainingError("model configuration does not match training report")
    if artifact.created_at != report.created_at:
        raise ClassifierTrainingError("model timestamp does not match training report")
    if report.training_count != len(_examples_for_split(dataset, RiskSplit.TRAIN)):
        raise ClassifierTrainingError("training count does not match frozen dataset")
    _validate_recomputed_metrics(report, artifact, dataset)
    if dataset_path.read_bytes() != dataset_bytes or model_path.read_bytes() != model_bytes:
        raise ClassifierTrainingError("classifier artifacts changed during validation")


def load_training_report(path: Path) -> ClassifierTrainingReport:
    return _load_json_model(path, ClassifierTrainingReport, "training report")


def load_model_artifact(path: Path) -> RiskClassifierArtifact:
    return _load_json_model(path, RiskClassifierArtifact, "model artifact")


def render_training_report(report: ClassifierTrainingReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def render_model_artifact(artifact: RiskClassifierArtifact) -> str:
    return artifact.model_dump_json(indent=2) + "\n"


def render_training_report_schema() -> str:
    return json.dumps(ClassifierTrainingReport.model_json_schema(), indent=2, sort_keys=True) + "\n"


def _new_vectorizer(
    config: ClassifierTrainingConfig,
    *,
    vocabulary: dict[str, int] | None = None,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        lowercase=config.lowercase,
        min_df=config.minimum_document_frequency,
        max_features=config.maximum_features,
        sublinear_tf=config.sublinear_term_frequency,
        norm=config.norm,
        dtype=np.float64,
        vocabulary=vocabulary,
    )


def _select_threshold(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    config: ClassifierTrainingConfig,
) -> float:
    minimum = round(config.threshold_minimum * 100)
    maximum = round(config.threshold_maximum * 100)
    step = round(config.threshold_step * 100)
    candidates = [round(value / 100, 2) for value in range(minimum, maximum + step, step)]
    if len(candidates) != 91:
        raise ClassifierTrainingError("threshold grid does not match the frozen configuration")

    def score(threshold: float) -> tuple[float, float, float, float]:
        predictions = (probabilities >= threshold).astype(np.int64)
        return (
            float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
            float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
            float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
            -threshold,
        )

    return max(candidates, key=score)


def _metrics(
    split: RiskSplit,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> BinaryClassificationMetrics:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_benign, false_alarm, missed_attack, detected_attack = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return BinaryClassificationMetrics(
        split=split,
        example_count=len(labels),
        threshold=threshold,
        true_benign=int(true_benign),
        false_alarm=int(false_alarm),
        missed_attack=int(missed_attack),
        detected_attack=int(detected_attack),
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        adversarial_precision=float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        adversarial_recall=float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        adversarial_f1=float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        average_precision=float(average_precision_score(labels, probabilities)),
    )


def _validate_recomputed_metrics(
    report: ClassifierTrainingReport,
    artifact: RiskClassifierArtifact,
    dataset: RiskDatasetManifest,
) -> None:
    split_reports = {
        RiskSplit.VALIDATION: report.validation_metrics,
        RiskSplit.TEST: report.test_metrics,
        RiskSplit.EXTERNAL_TEST: report.external_test_metrics,
    }
    for split, recorded in split_reports.items():
        examples = _examples_for_split(dataset, split)
        probabilities = predict_risk_probabilities(artifact, [item.text for item in examples])
        actual = _metrics(split, _labels(examples), probabilities, artifact.threshold)
        if not _metrics_close(recorded, actual):
            raise ClassifierTrainingError(f"recomputed {split.value} metrics do not match report")


def _metrics_close(
    expected: BinaryClassificationMetrics,
    actual: BinaryClassificationMetrics,
) -> bool:
    exact_fields = (
        "split",
        "example_count",
        "threshold",
        "true_benign",
        "false_alarm",
        "missed_attack",
        "detected_attack",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in exact_fields):
        return False
    float_fields = (
        "accuracy",
        "balanced_accuracy",
        "adversarial_precision",
        "adversarial_recall",
        "adversarial_f1",
        "roc_auc",
        "average_precision",
    )
    return all(
        math.isclose(getattr(expected, field), getattr(actual, field), rel_tol=1e-12, abs_tol=1e-12)
        for field in float_fields
    )


def _examples_for_split(dataset: RiskDatasetManifest, split: RiskSplit) -> list[RiskExample]:
    examples = [item for item in dataset.examples if item.split is split]
    if not examples:
        raise ClassifierTrainingError(f"classifier dataset has no {split.value} examples")
    return examples


def _labels(examples: list[RiskExample]) -> NDArray[np.int64]:
    return np.asarray(
        [1 if item.label is RiskLabel.ADVERSARIAL else 0 for item in examples],
        dtype=np.int64,
    )


def _positive_probabilities(classifier: LogisticRegression, matrix: object) -> NDArray[np.float64]:
    return np.asarray(classifier.predict_proba(matrix)[:, 1], dtype=np.float64)


def _sigmoid(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(scores, -709.0, 709.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _load_json_model(
    path: Path,
    model_type: type[ModelType],
    label: str,
) -> ModelType:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload)
    except (OSError, ValueError) as exc:
        raise ClassifierTrainingError(f"unable to load valid {label}: {path}") from exc


def _scikit_learn_version() -> Literal["1.9.1"]:
    installed = version("scikit-learn")
    if installed != "1.9.1":
        raise ClassifierTrainingError("classifier training requires scikit-learn 1.9.1")
    return "1.9.1"


def _require_new_target(path: Path) -> None:
    if path.exists():
        raise ClassifierTrainingError(f"classifier training output already exists: {path}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
