import json

import pytest
from pydantic import ValidationError

from deepshield_api.evaluation.synthetic import build_synthetic_environment
from deepshield_api.rag.c2_guardrails import (
    DeterministicC2Guardrails,
    MockToolCall,
    ToolAuthorizationContext,
)
from deepshield_api.rag.models import GuardrailAction, GuardrailCategory, GuardrailStage


@pytest.fixture
def guardrails() -> DeterministicC2Guardrails:
    return DeterministicC2Guardrails(build_synthetic_environment())


def test_c2_allows_output_without_registered_protected_values(
    guardrails: DeterministicC2Guardrails,
) -> None:
    events, blocked = guardrails.inspect_output(
        "F-1 students should consult the official policy source [S1]."
    )

    assert blocked is False
    assert len(events) == 1
    assert events[0].code == "c2-output-clean"
    assert events[0].action is GuardrailAction.ALLOW


def test_c2_blocks_all_registered_secret_and_pii_types_without_logging_values(
    guardrails: DeterministicC2Guardrails,
) -> None:
    environment = build_synthetic_environment()
    record = environment.records[0]
    output = " | ".join(
        [
            record.canary_token.lower(),
            record.email.upper(),
            record.phone,
            record.synthetic_sevis_id.lower(),
            record.record_note,
        ]
    )

    events, blocked = guardrails.inspect_output(output)

    assert blocked is True
    assert {event.code for event in events} == {
        "synthetic-canary-detected",
        "fictional-email-detected",
        "fictional-phone-detected",
        "fictional-record-note-detected",
        "fictional-student-id-detected",
    }
    serialized_events = json.dumps([event.model_dump(mode="json") for event in events])
    for protected_value in (
        record.canary_token,
        record.email,
        record.phone,
        record.synthetic_sevis_id,
        record.record_note,
    ):
        assert protected_value not in serialized_events
    assert all(event.stage is GuardrailStage.OUTPUT for event in events)
    assert all(event.action is GuardrailAction.BLOCK for event in events)


def test_c2_detects_a_reformatted_registered_phone_number(
    guardrails: DeterministicC2Guardrails,
) -> None:
    events, blocked = guardrails.inspect_output("Contact: 1 (202) 555-0101")

    assert blocked is True
    assert [event.code for event in events] == ["fictional-phone-detected"]


def test_c2_does_not_block_unregistered_test_values(
    guardrails: DeterministicC2Guardrails,
) -> None:
    events, blocked = guardrails.inspect_output(
        "Contact unrelated.person@example.test or +1-202-555-0199."
    )

    assert blocked is False
    assert events[0].category is GuardrailCategory.NONE


@pytest.mark.parametrize(
    ("call", "context", "expected_code"),
    [
        (
            MockToolCall(tool_id="unregistered-tool", fixture_id="fictional-student-0001"),
            ToolAuthorizationContext(),
            "unknown-mock-tool",
        ),
        (
            MockToolCall(tool_id="student-record-lookup", fixture_id="unknown-fixture"),
            ToolAuthorizationContext(granted_scopes=["student-record-read"]),
            "unknown-synthetic-fixture",
        ),
        (
            MockToolCall(tool_id="student-record-lookup", fixture_id="fictional-student-0001"),
            ToolAuthorizationContext(permitted_fixture_ids=["fictional-student-0001"]),
            "missing-tool-scope",
        ),
        (
            MockToolCall(tool_id="student-record-lookup", fixture_id="fictional-student-0001"),
            ToolAuthorizationContext(granted_scopes=["student-record-read"]),
            "unauthorized-synthetic-fixture",
        ),
        (
            MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0001"),
            ToolAuthorizationContext(
                granted_scopes=["student-record-write"],
                permitted_fixture_ids=["fictional-student-0001"],
            ),
            "side-effect-confirmation-required",
        ),
    ],
)
def test_c2_tool_authorization_denies_each_failed_boundary(
    guardrails: DeterministicC2Guardrails,
    call: MockToolCall,
    context: ToolAuthorizationContext,
    expected_code: str,
) -> None:
    decision = guardrails.authorize_tool(call, context)

    assert decision.allowed is False
    assert decision.event.action is GuardrailAction.BLOCK
    assert decision.event.stage is GuardrailStage.AUTHORIZATION
    assert decision.event.code == expected_code


def test_c2_authorizes_read_only_mock_tool_without_confirmation(
    guardrails: DeterministicC2Guardrails,
) -> None:
    call = MockToolCall(tool_id="student-record-lookup", fixture_id="fictional-student-0001")
    context = ToolAuthorizationContext(
        granted_scopes=["student-record-read"],
        permitted_fixture_ids=["fictional-student-0001"],
    )

    decision = guardrails.authorize_tool(call, context)

    assert decision.allowed is True
    assert decision.required_scope == "student-record-read"
    assert decision.has_external_side_effect is False
    assert decision.event.code == "mock-tool-authorized"
    assert decision.event.category is GuardrailCategory.AUTHORIZED_MOCK_TOOL


def test_c2_authorizes_effectful_mock_tool_only_after_trusted_confirmation(
    guardrails: DeterministicC2Guardrails,
) -> None:
    call = MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0001")
    context = ToolAuthorizationContext(
        granted_scopes=["student-record-write"],
        permitted_fixture_ids=["fictional-student-0001"],
        confirmed_side_effect_calls=[
            MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0001")
        ],
    )

    decision = guardrails.authorize_tool(call, context)

    assert decision.allowed is True
    assert decision.has_external_side_effect is True
    assert decision.event.action is GuardrailAction.ALLOW


def test_c2_authorization_context_rejects_duplicate_trusted_values() -> None:
    with pytest.raises(ValidationError, match="authorization context values must be unique"):
        ToolAuthorizationContext(granted_scopes=["student-record-read", "student-record-read"])


def test_c2_side_effect_confirmation_is_bound_to_exact_fixture(
    guardrails: DeterministicC2Guardrails,
) -> None:
    call = MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0002")
    context = ToolAuthorizationContext(
        granted_scopes=["student-record-write"],
        permitted_fixture_ids=["fictional-student-0001", "fictional-student-0002"],
        confirmed_side_effect_calls=[
            MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0001")
        ],
    )

    decision = guardrails.authorize_tool(call, context)

    assert decision.allowed is False
    assert decision.event.code == "side-effect-confirmation-required"


def test_c2_authorization_context_rejects_duplicate_confirmed_calls() -> None:
    confirmed = MockToolCall(tool_id="student-record-update", fixture_id="fictional-student-0001")
    with pytest.raises(ValidationError, match="confirmed side-effect calls must be unique"):
        ToolAuthorizationContext(confirmed_side_effect_calls=[confirmed, confirmed])
