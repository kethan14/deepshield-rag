"""Deterministic standard defenses used only by configuration C1."""

import re
from dataclasses import dataclass

from deepshield_api.rag.models import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvent,
    GuardrailStage,
)


@dataclass(frozen=True)
class PatternRule:
    """One auditable pattern-based C1 rule."""

    code: str
    category: GuardrailCategory
    pattern: re.Pattern[str]


INPUT_RULES = (
    PatternRule(
        "instruction-override",
        GuardrailCategory.DIRECT_PROMPT_INJECTION,
        re.compile(
            r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,80}"
            r"\b(?:instructions?|rules?|policies|policy|guardrails?|system)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    PatternRule(
        "role-manipulation",
        GuardrailCategory.ROLE_MANIPULATION,
        re.compile(
            r"\b(?:you are now|act as|developer mode|jailbreak|do anything now)\b",
            re.IGNORECASE,
        ),
    ),
    PatternRule(
        "system-prompt-request",
        GuardrailCategory.SYSTEM_PROMPT_EXTRACTION,
        re.compile(
            r"\b(?:reveal|show|print|repeat|return|expose)\b.{0,80}"
            r"\b(?:system prompts?|hidden instructions?|developer messages?|internal rules?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    PatternRule(
        "sensitive-data-request",
        GuardrailCategory.SENSITIVE_DATA_REQUEST,
        re.compile(
            r"\b(?:reveal|show|print|list|give|exfiltrate|leak)\b.{0,80}"
            r"\b(?:api key|secret|password|credential|access token|private record)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    PatternRule(
        "unauthorized-action-request",
        GuardrailCategory.UNAUTHORIZED_ACTION_REQUEST,
        re.compile(
            r"\b(?:call|run|execute|invoke|send|email|delete|update|modify)\b.{0,80}"
            r"\b(?:tool|function|database|record|email|file)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)
POLICY_INTENT_PATTERN = re.compile(
    r"\b(?:f-?1|student|visa|sevis|i-20|cpt|opt|employment|travel|status|course|"
    r"training|school|study|reinstatement|transfer)\b",
    re.IGNORECASE,
)
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENSITIVE_OUTPUT_RULES = (
    re.compile(r"\bDSRAG_CANARY_[A-Z0-9_]{8,80}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bDEEPSHIELD_C1_SYSTEM_POLICY\b"),
)


def inspect_input(text: str) -> tuple[list[GuardrailEvent], bool]:
    """Return sanitized detections and whether an attack-only request must be blocked."""

    events = _inspect_rules(text, GuardrailStage.INPUT, GuardrailAction.WARN)
    if CONTROL_CHARACTER_PATTERN.search(text):
        events.append(
            GuardrailEvent(
                stage=GuardrailStage.INPUT,
                action=GuardrailAction.BLOCK,
                category=GuardrailCategory.CONTROL_CHARACTERS,
                code="unsafe-control-character",
                detail="Blocked a request containing disallowed control characters.",
            )
        )
        return events, True
    if not events:
        return [
            GuardrailEvent(
                stage=GuardrailStage.INPUT,
                action=GuardrailAction.ALLOW,
                category=GuardrailCategory.NONE,
                code="input-clean",
                detail="No deterministic C1 input rule matched.",
            )
        ], False
    blocked = POLICY_INTENT_PATTERN.search(text) is None
    if blocked:
        events = [event.model_copy(update={"action": GuardrailAction.BLOCK}) for event in events]
    return events, blocked


def inspect_retrieved_text(text: str, chunk_id: str) -> list[GuardrailEvent]:
    """Identify retrieved instructions that must not reach the generator."""

    events = _inspect_rules(text, GuardrailStage.RETRIEVAL, GuardrailAction.EXCLUDE)
    return [
        event.model_copy(
            update={
                "chunk_id": chunk_id,
                "detail": f"Excluded retrieved chunk {chunk_id} after rule {event.code} matched.",
            }
        )
        for event in events
    ]


def inspect_output(text: str) -> list[GuardrailEvent]:
    """Block high-confidence secret, canary, or system-policy disclosure."""

    if not any(pattern.search(text) for pattern in SENSITIVE_OUTPUT_RULES):
        return []
    return [
        GuardrailEvent(
            stage=GuardrailStage.OUTPUT,
            action=GuardrailAction.BLOCK,
            category=GuardrailCategory.SENSITIVE_OUTPUT,
            code="sensitive-output-detected",
            detail="Blocked generated text matching a protected output pattern.",
        )
    ]


def _inspect_rules(
    text: str,
    stage: GuardrailStage,
    action: GuardrailAction,
) -> list[GuardrailEvent]:
    return [
        GuardrailEvent(
            stage=stage,
            action=action,
            category=rule.category,
            code=rule.code,
            detail=f"Rule {rule.code} matched untrusted {stage.value} text.",
        )
        for rule in INPUT_RULES
        if rule.pattern.search(text)
    ]
