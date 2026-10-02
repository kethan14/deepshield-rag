"""Provider-neutral text generation through an OpenAI-compatible HTTP endpoint."""

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class GenerationError(RuntimeError):
    """Raised when the configured generation backend cannot return valid text."""


@dataclass(frozen=True)
class GenerationResult:
    """Normalized completion output and optional provider usage metadata."""

    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GenerationBackend(Protocol):
    """Interface used by RAG orchestration and deterministic test doubles."""

    model: str

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Generate one answer from the fixed system and user prompts."""


class OpenAICompatibleGenerator:
    """Small synchronous client compatible with hosted or local chat-completion servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 700,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self.model = _require_nonempty(model, "model")
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        if not 1.0 <= timeout_seconds <= 300.0:
            raise ValueError("timeout_seconds must be between 1 and 300")
        if not 1 <= max_output_tokens <= 4_096:
            raise ValueError("max_output_tokens must be between 1 and 4096")
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Request one deterministic non-streaming chat completion."""

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": self._max_output_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                content = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise GenerationError("generation endpoint request failed") from exc
        if len(content) > MAX_RESPONSE_BYTES:
            raise GenerationError("generation endpoint response exceeded the size limit")
        return _parse_chat_completion(content, self.model)


def _validate_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme == "http" and parsed.hostname.lower() not in loopback_hosts:
        raise ValueError("unencrypted HTTP is allowed only for a loopback endpoint")
    return cleaned


def _require_nonempty(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty")
    return cleaned


def _parse_chat_completion(content: bytes, requested_model: str) -> GenerationResult:
    try:
        payload = json.loads(content.decode("utf-8"))
        choice = payload["choices"][0]
        text = choice["message"]["content"].strip()
    except (
        AttributeError,
        KeyError,
        IndexError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise GenerationError("generation endpoint returned an invalid response") from exc
    if not isinstance(text, str) or not text:
        raise GenerationError("generation endpoint returned an empty answer")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return GenerationResult(
        text=text,
        model=_optional_string(payload.get("model")) or requested_model,
        finish_reason=_optional_string(choice.get("finish_reason")),
        prompt_tokens=_optional_nonnegative_int(usage, "prompt_tokens"),
        completion_tokens=_optional_nonnegative_int(usage, "completion_tokens"),
        total_tokens=_optional_nonnegative_int(usage, "total_tokens"),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: object, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None
