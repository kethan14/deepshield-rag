"""Deterministic C2 leakage and mock-tool authorization guardrails."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deepshield_api.corpus.models import Slug
from deepshield_api.evaluation.synthetic import SyntheticSafetyEnvironment
from deepshield_api.rag.models import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailEvent,
    GuardrailStage,
)


class MockToolCall(BaseModel):
    """Untrusted mock-tool proposal produced by an assistant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_id: Slug
    fixture_id: Slug


class ToolAuthorizationContext(BaseModel):
    """Trusted server-side permissions, never inferred from prompt text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    granted_scopes: list[Slug] = Field(default_factory=list)
    permitted_fixture_ids: list[Slug] = Field(default_factory=list)
    confirmed_side_effect_calls: list[MockToolCall] = Field(default_factory=list)

    @field_validator("granted_scopes", "permitted_fixture_ids")
    @classmethod
    def require_unique_values(cls, value: list[str]) -> list[str]:
        """Reject ambiguous duplicated authorization values."""

        if len(value) != len(set(value)):
            raise ValueError("authorization context values must be unique")
        return value

    @field_validator("confirmed_side_effect_calls")
    @classmethod
    def require_unique_confirmed_calls(cls, value: list[MockToolCall]) -> list[MockToolCall]:
        """Bind each trusted confirmation to one exact tool and fixture pair."""

        call_keys = [(call.tool_id, call.fixture_id) for call in value]
        if len(call_keys) != len(set(call_keys)):
            raise ValueError("confirmed side-effect calls must be unique")
        return value


class ToolAuthorizationDecision(BaseModel):
    """Auditable allow-or-block result for an inert mock-tool proposal."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    tool_id: Slug
    fixture_id: Slug
    required_scope: Slug | None = None
    has_external_side_effect: bool | None = None
    event: GuardrailEvent


@dataclass(frozen=True)
class _ProtectedValue:
    fixture_id: str
    category: GuardrailCategory
    code: str
    label: str
    matches: Callable[[str], bool]


class DeterministicC2Guardrails:
    """Protect known synthetic fixtures and authorize only inert mock tools."""

    def __init__(self, environment: SyntheticSafetyEnvironment) -> None:
        self._records = {record.fixture_id: record for record in environment.records}
        self._tools = {tool.tool_id: tool for tool in environment.tools}
        self._protected_values = tuple(self._build_protected_values(environment))

    def inspect_output(self, text: str) -> tuple[list[GuardrailEvent], bool]:
        """Block exact known synthetic secrets and PII without logging their values."""

        events = [
            GuardrailEvent(
                stage=GuardrailStage.OUTPUT,
                action=GuardrailAction.BLOCK,
                category=value.category,
                code=value.code,
                detail=(
                    f"Blocked protected {value.label} associated with fixture {value.fixture_id}."
                ),
            )
            for value in self._protected_values
            if value.matches(text)
        ]
        if events:
            return events, True
        return [
            GuardrailEvent(
                stage=GuardrailStage.OUTPUT,
                action=GuardrailAction.ALLOW,
                category=GuardrailCategory.NONE,
                code="c2-output-clean",
                detail="No registered synthetic secret or fictional PII was detected.",
            )
        ], False

    def authorize_tool(
        self,
        call: MockToolCall,
        context: ToolAuthorizationContext,
    ) -> ToolAuthorizationDecision:
        """Deny by default unless trusted context authorizes every required boundary."""

        tool = self._tools.get(call.tool_id)
        if tool is None:
            return self._tool_decision(
                call,
                allowed=False,
                category=GuardrailCategory.UNKNOWN_MOCK_TOOL,
                code="unknown-mock-tool",
                detail="Blocked a proposal for an unregistered mock tool.",
            )

        if call.fixture_id not in self._records:
            return self._tool_decision(
                call,
                allowed=False,
                category=GuardrailCategory.UNKNOWN_SYNTHETIC_FIXTURE,
                code="unknown-synthetic-fixture",
                detail="Blocked a proposal targeting an unregistered synthetic fixture.",
                required_scope=tool.required_scope,
                has_external_side_effect=tool.has_external_side_effect,
            )

        if tool.required_scope not in context.granted_scopes:
            return self._tool_decision(
                call,
                allowed=False,
                category=GuardrailCategory.MISSING_TOOL_SCOPE,
                code="missing-tool-scope",
                detail="Blocked a mock-tool proposal because its required scope was absent.",
                required_scope=tool.required_scope,
                has_external_side_effect=tool.has_external_side_effect,
            )

        if call.fixture_id not in context.permitted_fixture_ids:
            return self._tool_decision(
                call,
                allowed=False,
                category=GuardrailCategory.UNAUTHORIZED_SYNTHETIC_FIXTURE,
                code="unauthorized-synthetic-fixture",
                detail="Blocked a mock-tool proposal outside the permitted fixture set.",
                required_scope=tool.required_scope,
                has_external_side_effect=tool.has_external_side_effect,
            )

        confirmed_calls = {
            (confirmed.tool_id, confirmed.fixture_id)
            for confirmed in context.confirmed_side_effect_calls
        }
        if tool.has_external_side_effect and (call.tool_id, call.fixture_id) not in confirmed_calls:
            return self._tool_decision(
                call,
                allowed=False,
                category=GuardrailCategory.SIDE_EFFECT_CONFIRMATION_REQUIRED,
                code="side-effect-confirmation-required",
                detail="Blocked an effectful mock-tool proposal without trusted confirmation.",
                required_scope=tool.required_scope,
                has_external_side_effect=True,
            )

        return self._tool_decision(
            call,
            allowed=True,
            category=GuardrailCategory.AUTHORIZED_MOCK_TOOL,
            code="mock-tool-authorized",
            detail="Authorized an inert mock-tool proposal using trusted server-side context.",
            required_scope=tool.required_scope,
            has_external_side_effect=tool.has_external_side_effect,
        )

    @staticmethod
    def _build_protected_values(
        environment: SyntheticSafetyEnvironment,
    ) -> list[_ProtectedValue]:
        values: list[_ProtectedValue] = []
        for record in environment.records:
            exact_fields = (
                (
                    record.canary_token,
                    GuardrailCategory.SYNTHETIC_SECRET_LEAKAGE,
                    "synthetic-canary-detected",
                    "canary",
                ),
                (
                    record.email,
                    GuardrailCategory.FICTIONAL_PII_LEAKAGE,
                    "fictional-email-detected",
                    "email",
                ),
                (
                    record.synthetic_sevis_id,
                    GuardrailCategory.FICTIONAL_PII_LEAKAGE,
                    "fictional-student-id-detected",
                    "student identifier",
                ),
                (
                    record.record_note,
                    GuardrailCategory.FICTIONAL_PII_LEAKAGE,
                    "fictional-record-note-detected",
                    "record note",
                ),
            )
            for protected, category, code, label in exact_fields:
                normalized = protected.casefold()
                values.append(
                    _ProtectedValue(
                        fixture_id=record.fixture_id,
                        category=category,
                        code=code,
                        label=label,
                        matches=_casefold_matcher(normalized),
                    )
                )

            phone_pattern = _phone_pattern(record.phone)
            values.append(
                _ProtectedValue(
                    fixture_id=record.fixture_id,
                    category=GuardrailCategory.FICTIONAL_PII_LEAKAGE,
                    code="fictional-phone-detected",
                    label="phone number",
                    matches=_regex_matcher(phone_pattern),
                )
            )
        return values

    @staticmethod
    def _tool_decision(
        call: MockToolCall,
        *,
        allowed: bool,
        category: GuardrailCategory,
        code: str,
        detail: str,
        required_scope: str | None = None,
        has_external_side_effect: bool | None = None,
    ) -> ToolAuthorizationDecision:
        return ToolAuthorizationDecision(
            allowed=allowed,
            tool_id=call.tool_id,
            fixture_id=call.fixture_id,
            required_scope=required_scope,
            has_external_side_effect=has_external_side_effect,
            event=GuardrailEvent(
                stage=GuardrailStage.AUTHORIZATION,
                action=GuardrailAction.ALLOW if allowed else GuardrailAction.BLOCK,
                category=category,
                code=code,
                detail=detail,
            ),
        )


def _phone_pattern(phone: str) -> re.Pattern[str]:
    digits = re.sub(r"\D", "", phone)
    flexible = r"[\s().-]*".join(re.escape(digit) for digit in digits)
    return re.compile(rf"(?<!\d){flexible}(?!\d)")


def _casefold_matcher(expected: str) -> Callable[[str], bool]:
    def matches(text: str) -> bool:
        return expected in text.casefold()

    return matches


def _regex_matcher(pattern: re.Pattern[str]) -> Callable[[str], bool]:
    def matches(text: str) -> bool:
        return pattern.search(text) is not None

    return matches
