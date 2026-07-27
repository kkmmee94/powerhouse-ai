from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from powerhouse_ai.config import get_settings
from powerhouse_ai.gpu import PeakVRAMTracker
from powerhouse_ai.llm import LLMClient
from powerhouse_ai.orchestrator import Orchestrator
from powerhouse_ai.router import Router
from powerhouse_ai.schemas import DocumentAdd
from powerhouse_ai.storage import MemoryStore


def source_fingerprint(project_root: Path) -> str:
    digest = hashlib.sha256()
    source_files = (
        path
        for path in (project_root / "src").rglob("*")
        if path.is_file() and path.suffix in {".py", ".txt"}
    )
    candidates = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        *source_files,
        *(project_root / "benchmarks").glob("*.py"),
        *(project_root / "benchmarks").glob("*.jsonl"),
    ]
    for path in sorted((item for item in candidates if item.is_file()), key=lambda item: str(item)):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/questions.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=["base", "fast", "balanced", "power", "adaptive"],
        default="adaptive",
    )
    args = parser.parse_args()
    output = args.output or Path(f"benchmarks/results/{args.mode}.jsonl")

    project_root = Path(__file__).resolve().parents[1]
    temporary_directory = tempfile.TemporaryDirectory(prefix="powerhouse-end-to-end-benchmark-")
    settings = get_settings().model_copy(
        update={"database_path": Path(temporary_directory.name) / "benchmark.db"}
    )
    store = MemoryStore(settings.database_path)
    llm = LLMClient(settings)
    orchestrator = Orchestrator(settings, store, llm)
    run_id = int(time.time())
    run_metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": git_commit(project_root),
        "source_sha256": source_fingerprint(project_root),
        "model_name": settings.model_name,
        "model_base_url": settings.model_base_url,
        "context_tokens": settings.context_tokens,
        "max_output_tokens": settings.max_output_tokens,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.questions.open(encoding="utf-8") as source,
        output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            case = json.loads(line)
            case_started = time.perf_counter()
            calls_before = llm.call_count
            try:
                if args.mode == "base":
                    with PeakVRAMTracker() as tracker:
                        answer = llm.chat(
                            [{"role": "user", "content": case["question"]}],
                            temperature=0.15,
                        ).content
                    record = {
                        **case,
                        "run_metadata": run_metadata,
                        "mode": args.mode,
                        "answer": answer,
                        "route": None,
                        "route_evaluated": False,
                        "latency_seconds": round(time.perf_counter() - case_started, 3),
                        "model_calls": llm.call_count - calls_before,
                        "peak_vram_mb": tracker.peak_mb,
                        "warnings": [],
                    }
                else:
                    project = store.create_project(f"Benchmark {run_id} {case['id']}")
                    store.add_memory(
                        project.id,
                        (
                            "Saved project decisions: use hours as the time unit, use a logistic "
                            "growth model, use Python 3.12 and SQLite. Unfinished work: add the "
                            "visualisation."
                        ),
                        importance=1.0,
                    )
                    for memory in case.get("seed_memories", []):
                        store.add_memory(project.id, memory, importance=1.0)
                    for document in case.get("seed_documents", []):
                        store.add_document(
                            DocumentAdd(
                                project_id=project.id,
                                domain=document["domain"],
                                title=document["title"],
                                content=document["content"],
                                source=document.get("source", "benchmark"),
                                trust_level=document.get("trust_level", "high"),
                            )
                        )
                    result = orchestrator.process(
                        project.id,
                        case["question"],
                        mode=args.mode,
                    )
                    record = {
                        **case,
                        "run_metadata": run_metadata,
                        "mode": args.mode,
                        "answer": result.answer,
                        "route": result.route.model_dump(mode="json"),
                        "route_evaluated": True,
                        "latency_seconds": result.latency_seconds,
                        "model_calls": result.model_calls,
                        "peak_vram_mb": result.peak_vram_mb,
                        "warnings": result.warnings,
                    }
            except Exception as exc:
                route = None
                if args.mode != "base":
                    route = Router().heuristic(case["question"]).model_dump(mode="json")
                record = {
                    **case,
                    "run_metadata": run_metadata,
                    "mode": args.mode,
                    "answer": "",
                    "route": route,
                    "route_evaluated": args.mode != "base",
                    "latency_seconds": round(time.perf_counter() - case_started, 3),
                    "model_calls": llm.call_count - calls_before,
                    "peak_vram_mb": None,
                    "warnings": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            status = f", ERROR: {record['error']}" if record.get("error") else ""
            print(
                f"{case['id']}: {record['latency_seconds']}s, "
                f"{record['model_calls']} calls, {record['peak_vram_mb']} MB{status}"
            )
    llm.close()
    temporary_directory.cleanup()


if __name__ == "__main__":
    main()
