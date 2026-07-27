from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from powerhouse_ai.config import get_settings
from powerhouse_ai.gpu import get_gpu_status

PROMPTS = [
    "Explain the practical difference between RAM and VRAM in exactly 150 words.",
    "Solve x^2 - 11x + 24 = 0, verify both roots, and explain the method.",
    (
        "Write a correct Python implementation of binary search with type hints, input "
        "validation, and three assert-based tests."
    ),
]


def sample_gpu(stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(0.1):
        status = get_gpu_status()
        if status.used_mb is not None:
            samples.append(status.used_mb)


def run_case(client: httpx.Client, prompt: str, max_tokens: int) -> dict[str, Any]:
    samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    response = client.post(
        "chat/completions",
        json={
            "model": get_settings().model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": get_settings().model_enable_thinking,
            },
        },
    )
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=2)
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "prompt": prompt,
        "latency_seconds": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "end_to_end_tokens_per_second": (
            round(completion_tokens / elapsed, 3) if completion_tokens else None
        ),
        "generation_tokens_per_second": timings.get("predicted_per_second"),
        "prompt_eval_seconds": (
            round(float(timings["prompt_ms"]) / 1000, 3)
            if timings.get("prompt_ms") is not None
            else None
        ),
        "peak_vram_mb": max(samples) if samples else None,
        "answer": data["choices"][0]["message"]["content"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/model-profile.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    with httpx.Client(
        base_url=settings.model_base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {settings.model_api_key}"},
        timeout=settings.request_timeout_seconds,
    ) as client:
        warmup_started = time.perf_counter()
        warmup = client.post(
            "chat/completions",
            json={
                "model": settings.model_name,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        warmup.raise_for_status()
        warmup_seconds = round(time.perf_counter() - warmup_started, 3)
        cases = [run_case(client, prompt, args.max_tokens) for prompt in PROMPTS]
    result = {
        "profile": args.profile,
        "context_tokens": settings.context_tokens,
        "cold_warmup_seconds": warmup_seconds,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
