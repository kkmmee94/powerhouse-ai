# Reproducing validation

Powerhouse results depend on model bytes, runtime version, sampling settings, source revision, GPU,
driver, and benchmark inputs. A release result is attributable only to that complete configuration.

## Software checks

```powershell
uv sync --locked --extra dev
uv run ruff check .
uv run pytest --cov=powerhouse_ai
uv build
```

## Local deterministic checks

```powershell
uv run python benchmarks/evaluate_memory.py --output benchmarks/results/v0.1.0/memory.json
```

## End-to-end runs

Start the pinned Q6_K model, then run:

```powershell
uv run python benchmarks/run_benchmark.py --mode base --output benchmarks/results/v0.1.0/base.jsonl
uv run python benchmarks/run_benchmark.py --mode fast --output benchmarks/results/v0.1.0/fast.jsonl
uv run python benchmarks/run_benchmark.py --mode adaptive --output benchmarks/results/v0.1.0/adaptive.jsonl
uv run python benchmarks/judge_run.py benchmarks/results/v0.1.0/base.jsonl
uv run python benchmarks/judge_run.py benchmarks/results/v0.1.0/fast.jsonl
uv run python benchmarks/judge_run.py benchmarks/results/v0.1.0/adaptive.jsonl
uv run python benchmarks/compare_runs.py benchmarks/results/v0.1.0/base.scored.jsonl benchmarks/results/v0.1.0/fast.scored.jsonl benchmarks/results/v0.1.0/adaptive.scored.jsonl
```

The local model is also the rubric judge, so these scores are repeatable project diagnostics rather
than an independent human evaluation. Preserve raw and scored JSONL, record SHA-256 hashes, and never
silently combine records from different source revisions.
