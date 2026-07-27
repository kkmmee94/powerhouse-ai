# Contributing

Powerhouse AI is an experimental local-AI runtime. Focus contributions on measurable correctness,
privacy, resource use, and reproducibility.

## Development setup

On Windows with Python 3.12 and `uv` installed:

```powershell
uv sync --locked --extra dev
Copy-Item .env.example .env
uv run ruff check .
uv run pytest --cov=powerhouse_ai
uv build
```

Use a focused branch, include tests for behavior changes, and update documentation when interfaces,
safety boundaries, model artifacts, or benchmark methodology change. Never commit `.env`, model
weights, databases, logs, private documents, or unsanitized benchmark output.

By contributing, you agree that your contribution is licensed under the repository's MIT license.
