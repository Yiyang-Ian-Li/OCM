from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

DEFAULT_API_MAX_ATTEMPTS = 3


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int] | None
    message: Any
    response: Any
    tool_calls: list[Any]


def _get_usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(key)
    if value is None:
        return None
    return int(value)


def extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)),
    }


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: int,
        max_attempts: int = DEFAULT_API_MAX_ATTEMPTS,
        usage_callback: Callable[[dict[str, int] | None, str], None] | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.base_url = str(base_url)
        self.max_attempts = max(1, int(max_attempts))
        self._usage_callback = usage_callback

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {408, 409, 429} or exc.status_code >= 500
        return False

    def _create_with_retry(self, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_attempts or not self._should_retry(exc):
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        assert last_exc is not None
        raise last_exc

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        call_type: str = "chat",
    ) -> ChatResult:
        kwargs = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "stop": stop,
            "tools": list(tools) if tools is not None else None,
            "tool_choice": tool_choice,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if seed is not None:
            kwargs["seed"] = int(seed)
        if "openrouter.ai" in self.base_url:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        response = self._create_with_retry(**kwargs)
        message = response.choices[0].message
        usage = extract_usage(response)
        if self._usage_callback is not None:
            self._usage_callback(usage, call_type)
        return ChatResult(
            content=message.content or "",
            usage=usage,
            message=message,
            response=response,
            tool_calls=list(getattr(message, "tool_calls", None) or []),
        )
