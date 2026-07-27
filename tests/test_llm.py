import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx

from powerhouse_ai.config import Settings
from powerhouse_ai.llm import LLMClient, _estimate_messages_tokens


def test_llm_client_health_and_json_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        payload = json.loads(request.content)
        assert payload["model"] == "local-model"
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"goal":"test","domains":[]}\n```',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = LLMClient(
        Settings(),
        transport=httpx.MockTransport(handler),
    )
    assert client.health() is True
    parsed = client.chat_json([{"role": "user", "content": "route this"}])
    assert parsed["goal"] == "test"
    assert client.call_count == 1
    client.close()


def test_health_rejects_client_errors_and_missing_model() -> None:
    responses = iter(
        [
            httpx.Response(401, json={"error": "bad key"}),
            httpx.Response(200, json={"data": [{"id": "different-model"}]}),
        ]
    )
    client = LLMClient(
        Settings(),
        transport=httpx.MockTransport(lambda _: next(responses)),
    )
    assert client.health() is False
    assert client.health() is False
    client.close()


def test_chat_fits_messages_inside_configured_context() -> None:
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed.update(payload)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fitted"}}]},
        )

    settings = Settings(context_tokens=2048, max_output_tokens=1024)
    client = LLMClient(settings, transport=httpx.MockTransport(handler))
    reply = client.chat(
        [
            {"role": "system", "content": "system guidance " * 500},
            {"role": "user", "content": "old user turn " * 500},
            {"role": "assistant", "content": "old assistant turn " * 500},
            {"role": "user", "content": "current request " * 5000},
        ]
    )
    assert reply.content == "fitted"
    assert observed["max_tokens"] == 1024
    assert _estimate_messages_tokens(observed["messages"]) <= 896
    assert any(
        "[truncated to fit model context]" in message.get("content", "")
        for message in observed["messages"]
    )
    client.close()


def test_call_tracking_is_scoped_per_concurrent_request() -> None:
    barrier = Barrier(2)

    def handler(_: httpx.Request) -> httpx.Response:
        barrier.wait(timeout=5)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done"}}]},
        )

    client = LLMClient(Settings(), transport=httpx.MockTransport(handler))

    def run_request() -> int:
        with client.track_calls() as tracker:
            client.chat([{"role": "user", "content": "hello"}])
            return tracker.count

    with ThreadPoolExecutor(max_workers=2) as executor:
        counts = list(executor.map(lambda _: run_request(), range(2)))

    assert counts == [1, 1]
    assert client.call_count == 2
    client.close()
