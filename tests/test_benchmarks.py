from pathlib import Path

from benchmarks.compare_runs import summarize
from benchmarks.run_benchmark import source_fingerprint


def _record(*, route_evaluated: bool, warnings: list[str]) -> dict:
    return {
        "answer": "done",
        "domains": ["general"],
        "expected_points": [],
        "latency_seconds": 1.0,
        "model_calls": 1,
        "peak_vram_mb": 100,
        "route": None,
        "route_evaluated": route_evaluated,
        "warnings": warnings,
    }


def test_summary_does_not_attribute_routing_to_base_model() -> None:
    report = summarize([_record(route_evaluated=False, warnings=[])])
    assert report["route_recall"] is None
    assert report["route_exact_rate"] is None


def test_summary_counts_only_vram_budget_warnings() -> None:
    syntax_warning = _record(
        route_evaluated=False,
        warnings=["Final answer contains Python syntax errors."],
    )
    vram_warning = _record(
        route_evaluated=False,
        warnings=["Observed VRAM 200 MB exceeded the 150 MB budget."],
    )
    report = summarize([syntax_warning, vram_warning])
    assert report["budget_violations"] == 1


def test_source_fingerprint_ignores_bytecode_cache(tmp_path: Path) -> None:
    (tmp_path / "src" / "package" / "__pycache__").mkdir(parents=True)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "src" / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = tmp_path / "src" / "package" / "__pycache__" / "module.pyc"
    cache.write_bytes(b"first cache")

    first = source_fingerprint(tmp_path)
    cache.write_bytes(b"different cache")

    assert source_fingerprint(tmp_path) == first
