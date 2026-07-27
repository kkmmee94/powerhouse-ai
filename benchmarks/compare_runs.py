from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def route_recall(record: dict[str, Any]) -> float:
    expected = set(record["domains"])
    predicted = {item["name"] for item in record["route"]["domains"]}
    return len(expected & predicted) / len(expected)


def route_exact(record: dict[str, Any]) -> bool:
    expected = set(record["domains"])
    predicted = {item["name"] for item in record["route"]["domains"]}
    return expected == predicted


def python_syntax_ok(record: dict[str, Any]) -> bool:
    blocks = re.findall(
        r"```(?:python|py)[ \t]*\r?\n(.*?)```",
        record["answer"],
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not blocks:
        return False
    try:
        for block in blocks:
            ast.parse(block)
    except SyntaxError:
        return False
    return True


def requests_executable_code(record: dict[str, Any]) -> bool:
    requirements = " ".join(str(item).lower() for item in record.get("expected_points", []))
    return any(
        marker in requirements
        for marker in ("valid python", "valid code", "implementation", "code tests", "tests")
    )


def summarize(
    records: list[dict[str, Any]],
    *,
    route_evaluated: bool | None = None,
) -> dict[str, Any]:
    latencies = [float(item["latency_seconds"]) for item in records]
    vram = [int(item["peak_vram_mb"]) for item in records if item["peak_vram_mb"] is not None]
    programming = [item for item in records if requests_executable_code(item)]
    routed = [
        item
        for item in records
        if route_evaluated is not False
        and item.get("route_evaluated", True)
        and isinstance(item.get("route"), dict)
    ]
    report = {
        "cases": len(records),
        "run_metadata": records[0].get("run_metadata") if records else None,
        "route_recall": (
            round(statistics.fmean(route_recall(item) for item in routed), 4)
            if routed
            else None
        ),
        "route_exact_rate": (
            round(statistics.fmean(route_exact(item) for item in routed), 4)
            if routed
            else None
        ),
        "pipeline_completion_rate": round(
            statistics.fmean(bool(item.get("answer", "").strip()) for item in records),
            4,
        ),
        "mean_latency_seconds": round(statistics.fmean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "mean_model_calls": round(
            statistics.fmean(int(item["model_calls"]) for item in records), 2
        ),
        "peak_vram_mb": max(vram) if vram else None,
        "budget_violations": sum(
            any(
                "vram" in str(warning).lower() and "exceeded" in str(warning).lower()
                for warning in item.get("warnings", [])
            )
            for item in records
        ),
        "programming_python_syntax_rate": (
            round(statistics.fmean(python_syntax_ok(item) for item in programming), 4)
            if programming
            else None
        ),
    }
    graded = [item["grade"] for item in records if "grade" in item]
    if graded:
        mean_score = statistics.fmean(float(item["total"]) for item in graded)
        report.update(
            {
                "mean_judge_score": round(mean_score, 3),
                "judge_score_percent": round(mean_score / 8 * 100, 2),
                "unsupported_claims": sum(
                    int(item.get("unsupported_claims", 0)) for item in graded
                ),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    runs = {
        str(path): summarize(
            load(path),
            route_evaluated=(
                False
                if path.name.lower() in {"base.jsonl", "base.scored.jsonl"}
                else None
            ),
        )
        for path in args.runs
    }
    report: dict[str, Any] = {"runs": runs}
    if len(args.runs) > 1:
        baseline_name = str(args.runs[0])
        baseline = runs[baseline_name]
        comparisons = {}
        for path in args.runs[1:]:
            candidate_name = str(path)
            candidate = runs[candidate_name]
            comparison: dict[str, Any] = {
                "latency_multiplier": round(
                    candidate["mean_latency_seconds"] / baseline["mean_latency_seconds"],
                    3,
                ),
                "route_recall_point_change": (
                    round(
                        (candidate["route_recall"] - baseline["route_recall"]) * 100,
                        2,
                    )
                    if candidate["route_recall"] is not None
                    and baseline["route_recall"] is not None
                    else None
                ),
                "peak_vram_change_mb": (
                    candidate["peak_vram_mb"] - baseline["peak_vram_mb"]
                    if candidate["peak_vram_mb"] is not None
                    and baseline["peak_vram_mb"] is not None
                    else None
                ),
            }
            if "mean_judge_score" in baseline and "mean_judge_score" in candidate:
                comparison["judge_score_relative_change_percent"] = round(
                    (candidate["mean_judge_score"] / baseline["mean_judge_score"] - 1) * 100,
                    2,
                )
                base_claims = baseline["unsupported_claims"]
                candidate_claims = candidate["unsupported_claims"]
                comparison["unsupported_claims_relative_change_percent"] = (
                    round((candidate_claims / base_claims - 1) * 100, 2)
                    if base_claims
                    else None
                )
            comparisons[f"{candidate_name} versus {baseline_name}"] = comparison
        report["comparisons"] = comparisons
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
