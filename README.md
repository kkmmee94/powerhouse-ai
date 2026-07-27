# Powerhouse AI

[![CI](https://github.com/kkmmee94/powerhouse-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/kkmmee94/powerhouse-ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kkmmee94/powerhouse-ai/blob/main/LICENSE)

Powerhouse AI is an experimental, private local assistant for a single VRAM-constrained NVIDIA GPU.
It combines a text interface to one local language model with project memory, SQLite FTS5 retrieval,
specialist prompts, deterministic planning, safe calculation tools, synthesis, and final
verification.

Version 0.1.0 is a Windows/NVIDIA preview tested on an RTX 5060 Ti with 16 GB VRAM. The selected
Qwen3.5 model is multimodal-capable, but the Powerhouse API and CLI are currently **text-only**.
Powerhouse is intended to improve the same local model on measurable project-specific work; it is
not presented as a replacement for a frontier model or as a production multi-user service.

The implemented flow is documented in the
[architecture guide](https://github.com/kkmmee94/powerhouse-ai/blob/main/docs/ARCHITECTURE.md).

## Features

- OpenAI-compatible local-model client
- adaptive `fast`, `balanced`, `adaptive`, and `power` modes
- heuristic and model-assisted multi-domain routing
- deterministic dependency-aware planning
- eight prompt specialists using one resident model
- project messages and long-term memory in SQLite
- project-scoped and domain-scoped SQLite FTS5 retrieval
- context fitting with reserved output capacity
- restricted calculator and dimension-checked metric conversion
- disabled-by-default trusted Python subprocess tool
- Python syntax checking and repair without execution
- synthesis, evidence-aware final verification, and cited-source reporting
- FastAPI and Typer interfaces bound to localhost
- NVIDIA VRAM sampling and budget warnings
- unit, integration, benchmark, and packaging checks

## Supported environment

The automated runtime setup currently supports:

| Component | Supported release configuration |
|---|---|
| Operating system | Windows 11 |
| Python | 3.12 |
| GPU | NVIDIA CUDA-capable GPU |
| Validated VRAM | 16 GB |
| Model server | pinned CUDA build of `llama.cpp` |
| Model | Qwen3.5-9B GGUF, Q6_K default |

The Python package may work with another OpenAI-compatible endpoint, but other operating systems,
GPU backends, and hardware sizes are not release-tested.

## Quick start

Install [Git](https://git-scm.com/), [Python 3.12](https://www.python.org/downloads/), and
[`uv`](https://docs.astral.sh/uv/), then clone the source:

```powershell
git clone https://github.com/kkmmee94/powerhouse-ai.git
Set-Location powerhouse-ai
.\scripts\bootstrap.ps1
```

Install the pinned CUDA runtime and download the balanced model:

```powershell
.\scripts\install-llama-cuda.ps1
.\scripts\download-model.ps1 -Profile balanced
```

Both scripts verify pinned SHA-256 digests before the downloaded files are used. Start the model and
local API:

```powershell
.\scripts\start-powerhouse.ps1 -Profile balanced
```

Inspect the runtime and ask a question:

```powershell
uv run powerhouse status
uv run powerhouse ask "Build a mathematical model of bacterial growth."
```

The command creates and reuses a default project. For isolated work:

```powershell
uv run powerhouse create-project "My project"
uv run powerhouse chat PROJECT_ID "Summarize the current design constraints."
uv run powerhouse ingest .\my-notes research --project-id PROJECT_ID --trust-level high
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. Stop both processes with:

```powershell
.\scripts\stop-powerhouse.ps1
```

## Execution modes

- `fast`: deterministic routing and one model stage.
- `balanced`: specialist stages for material multi-domain work.
- `adaptive`: fast for simple work and heavier staging for complex work; this is the default.
- `power`: model-assisted routing, model thinking, specialist stages, synthesis, and verification.

## Security and privacy

Powerhouse is a single-user local application:

- The built-in model server and API bind to `127.0.0.1`; the API has no authentication.
- Messages, memory, document chunks, and local source paths are stored unencrypted in SQLite.
- Retrieved content is untrusted data and may contain prompt injection. Do not ingest untrusted
  material into a high-trust project.
- Python execution is off by default. `POWERHOUSE_ENABLE_PYTHON_TOOL=true` is only for code you trust;
  timeout and isolation flags do not make it a hostile-code sandbox.
- Do not expose either local service through a proxy, tunnel, port-forward, or custom network bind.

See the complete [security policy](https://github.com/kkmmee94/powerhouse-ai/blob/main/SECURITY.md).

## Validation status

The unit and integration suite, distribution build, clean-install smoke test, dependency audit, and
release benchmark evidence are recorded in the
[validation report](https://github.com/kkmmee94/powerhouse-ai/blob/main/docs/VALIDATION.md).

On the 44-case release suite, the plain local model scored 76.14%, Powerhouse Fast scored 93.75%,
and Powerhouse Adaptive scored 96.59%. Fast averaged 14.046 seconds per request; Adaptive averaged
46.502 seconds. These are project-focused, model-graded measurements on one machine, not universal
claims about model quality.

This is an experimental release. Held-out routing evaluation and execution-based generated-code
evaluation remain open acceptance criteria; syntax-only checks do not satisfy executable-code
correctness.

## Development

```powershell
uv sync --locked --extra dev
uv run ruff check .
uv run pytest --cov=powerhouse_ai
uv build
uv run python benchmarks/evaluate_memory.py
```

End-to-end model benchmarks are hardware-dependent and intentionally run separately. See
[CONTRIBUTING.md](https://github.com/kkmmee94/powerhouse-ai/blob/main/CONTRIBUTING.md) for the public
workflow.

## License and third-party components

Powerhouse AI source code is released under the [MIT License](https://github.com/kkmmee94/powerhouse-ai/blob/main/LICENSE).
Model weights and downloaded runtimes are not redistributed by this repository and remain subject to
their respective upstream licenses.
