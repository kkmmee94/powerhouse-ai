from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from powerhouse_ai.storage import MemoryStore

CASES = [
    ("The simulation time step is 0.25 hours.", "simulation time step hours"),
    ("The chosen growth model is logistic with carrying capacity.", "chosen growth model"),
    ("The local API port is 8123.", "local API port"),
    ("The database uses SQLite in WAL mode.", "database SQLite WAL"),
    ("The experimental organism is E. coli K-12.", "experimental organism coli K-12"),
    ("The primary output unit is cells per millilitre.", "primary output unit cells"),
    ("The calibration file is named calibration-v3.csv.", "calibration filename v3"),
    ("The project colour is cobalt blue.", "project colour cobalt"),
    ("The retry limit is five attempts.", "retry limit attempts"),
    ("The report audience is first-year biology students.", "report audience biology students"),
    ("The selected optimizer is AdamW.", "selected optimizer AdamW"),
    ("The validation split is fifteen percent.", "validation split percent"),
    ("The deployment target is a private Windows workstation.", "deployment target Windows"),
    ("The sample label prefix is EXP-27.", "sample label prefix"),
    ("The approved temperature is 37 degrees Celsius.", "approved temperature Celsius"),
    ("The control group receives no added glucose.", "control group glucose"),
    ("The chart must use a logarithmic y-axis.", "chart logarithmic axis"),
    ("The repository branch is feature/local-runtime.", "repository branch local runtime"),
    ("The minimum confidence threshold is 0.82.", "minimum confidence threshold"),
    ("The final export format is Parquet.", "final export format Parquet"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/memory.json"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="powerhouse-memory-benchmark-") as directory:
        store = MemoryStore(Path(directory) / "memory.db")
        project = store.create_project("Memory benchmark")
        other = store.create_project("Isolation control")
        for index in range(40):
            store.add_memory(
                project.id,
                f"Distractor note {index}: routine status information with no target decision.",
                importance=0.2,
            )
        for target, _ in CASES:
            store.add_memory(project.id, target, importance=0.9)
        store.add_memory(
            other.id,
            "The simulation time step is 99 days and this belongs to another project.",
            importance=1.0,
        )

        hits = 0
        details = []
        for target, query in CASES:
            retrieved = store.search_memory(project.id, query, limit=8)
            found = target in retrieved
            hits += found
            details.append({"query": query, "found": found, "retrieved": retrieved})

        isolation_results = store.search_memory(project.id, "time step")
        isolated = all("99 days" not in item for item in isolation_results)
        result = {
            "cases": len(CASES),
            "top_8_hits": hits,
            "top_8_accuracy": round(hits / len(CASES), 4),
            "project_isolation": isolated,
            "details": details,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
