from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import httpx

from powerhouse_ai.config import Settings
from powerhouse_ai.errors import ModelResponseError, ModelUnavailableError


@dataclass(slots=True)
class ModelReply:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class CallTracker:
    count: int = 0


def _estimate_text_tokens(text: str) -> int:
    """Conservatively estimate tokens without depending on one model's tokenizer."""
    ascii_characters = sum(character.isascii() for character in text)
    non_ascii_bytes = sum(
        len(character.encode("utf-8")) for character in text if not character.isascii()
    )
    return max(1, math.ceil(ascii_characters / 3) + non_ascii_bytes)


def _estimate_message_tokens(message: dict[str, Any]) -> int:
    tokens = 6
    for key, value in message.items():
        if key == "content" and isinstance(value, str):
            tokens += _estimate_text_tokens(value)
        else:
            tokens += _estimate_text_tokens(json.dumps(value, ensure_ascii=False, default=str))
    return tokens


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def _truncate_text(text: str, max_tokens: int) -> str:
    if _estimate_text_tokens(text) <= max_tokens:
        return text
    marker = "\n...[truncated to fit model context]...\n"
    if max_tokens <= _estimate_text_tokens(marker):
        return "[truncated]"

    low = 0
    high = len(text)
    best = marker
    while low <= high:
        keep = (low + high) // 2
        head = math.ceil(keep * 2 / 3)
        tail = keep - head
        candidate = text[:head] + marker + (text[-tail:] if tail else "")
        if _estimate_text_tokens(candidate) <= max_tokens:
            best = candidate
            low = keep + 1
        else:
            high = keep - 1
    return best


class LLMClient:
    """Small client for llama.cpp, vLLM, Ollama, or any OpenAI-compatible local server."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.model_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.model_api_key}"},
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )
        self._call_count = 0
        self._call_count_lock = threading.Lock()
        self._active_call_tracker: ContextVar[CallTracker | None] = ContextVar(
            f"powerhouse_llm_call_tracker_{id(self)}",
            default=None,
        )

    @property
    def call_count(self) -> int:
        with self._call_count_lock:
            return self._call_count

    @contextmanager
    def track_calls(self) -> Iterator[CallTracker]:
        tracker = CallTracker()
        token = self._active_call_tracker.set(tracker)
        try:
            yield tracker
        finally:
            self._active_call_tracker.reset(token)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            response = self._client.get("models", timeout=3.0)
            if not response.is_success:
                return False
            data = response.json()
            if not isinstance(data, dict):
                return False
            models = data.get("data")
            return isinstance(models, list) and any(
                isinstance(model, dict) and model.get("id") == self.settings.model_name
                for model in models
            )
        except (httpx.HTTPError, TypeError, ValueError):
            return False

    def _fit_messages(
        self,
        messages: list[dict[str, Any]],
        token_budget: int,
    ) -> list[dict[str, Any]]:
        fitted = [dict(message) for message in messages]

        while _estimate_messages_tokens(fitted) > token_budget:
            first_tool_call = next(
                (
                    index
                    for index, message in enumerate(fitted)
                    if message.get("tool_calls")
                ),
                None,
            )
            protected_from = len(fitted) - 1
            if first_tool_call is not None:
                protected_from = max(
                    (
                        index
                        for index, message in enumerate(fitted[:first_tool_call])
                        if message.get("role") == "user"
                    ),
                    default=protected_from,
                )
            droppable = next(
                (
                    index
                    for index, message in enumerate(fitted[1:protected_from], start=1)
                    if message.get("role") in {"user", "assistant"}
                    and not message.get("tool_calls")
                ),
                None,
            )
            if droppable is None:
                break
            fitted.pop(droppable)

        while _estimate_messages_tokens(fitted) > token_budget:
            candidates = [
                (index, _estimate_text_tokens(message["content"]))
                for index, message in enumerate(fitted)
                if isinstance(message.get("content"), str)
                and _estimate_text_tokens(message["content"]) > 24
            ]
            if not candidates:
                break
            index, current_tokens = max(candidates, key=lambda item: item[1])
            excess = _estimate_messages_tokens(fitted) - token_budget
            target = max(24, current_tokens - excess - 8)
            fitted[index]["content"] = _truncate_text(fitted[index]["content"], target)

        if _estimate_messages_tokens(fitted) > token_budget:
            raise ModelResponseError(
                "Request metadata cannot fit inside the configured model context."
            )
        return fitted

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelReply:
        requested_max_tokens = max_tokens or self.settings.max_output_tokens
        resolved_tool_choice = (tool_choice or "auto") if tools else None
        request_metadata: dict[str, Any] = {}
        if json_mode:
            request_metadata["response_format"] = {"type": "json_object"}
        if tools:
            request_metadata["tools"] = tools
            request_metadata["tool_choice"] = resolved_tool_choice
        metadata_tokens = (
            _estimate_text_tokens(
                json.dumps(request_metadata, ensure_ascii=False, default=str)
            )
            if request_metadata
            else 0
        )
        context_overhead = 128 + metadata_tokens
        minimum_input_tokens = max(512, self.settings.context_tokens // 4)
        max_output_capacity = max(
            1,
            self.settings.context_tokens - context_overhead - minimum_input_tokens,
        )
        effective_max_tokens = min(requested_max_tokens, max_output_capacity)
        input_budget = self.settings.context_tokens - context_overhead - effective_max_tokens
        fitted_messages = self._fit_messages(messages, input_budget)

        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": fitted_messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": (
                    self.settings.model_enable_thinking
                    if enable_thinking is None
                    else enable_thinking
                ),
            },
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = resolved_tool_choice

        try:
            response = self._client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"Local model server is unavailable at {self.settings.model_base_url}: {exc}"
            ) from exc

        with self._call_count_lock:
            self._call_count += 1
        tracker = self._active_call_tracker.get()
        if tracker is not None:
            tracker.count += 1
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            usage = data.get("usage", {})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError("Local model returned an invalid chat response.") from exc
        if not isinstance(content, str):
            raise ModelResponseError("Local model returned invalid content.")
        if not content.strip() and not tool_calls:
            raise ModelResponseError("Local model returned an empty response.")
        return ModelReply(
            content=content.strip(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            tool_calls=tool_calls,
        )

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        reply = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            enable_thinking=enable_thinking,
        )
        try:
            parsed = json.loads(reply.content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", reply.content, flags=re.DOTALL)
            if not match:
                raise ModelResponseError("Model did not return a JSON object.") from None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ModelResponseError("Model returned malformed JSON.") from exc
        if not isinstance(parsed, dict):
            raise ModelResponseError("Model JSON response was not an object.")
        return parsed
