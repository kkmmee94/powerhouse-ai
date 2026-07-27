from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from powerhouse_ai.config import ExecutionMode, get_settings
from powerhouse_ai.gpu import get_gpu_status
from powerhouse_ai.llm import LLMClient
from powerhouse_ai.orchestrator import Orchestrator
from powerhouse_ai.schemas import ChatResponse, DocumentAdd, Domain
from powerhouse_ai.storage import MemoryStore

app = typer.Typer(help="Powerhouse AI local runtime")
console = Console()


def _components() -> tuple[MemoryStore, LLMClient, Orchestrator]:
    settings = get_settings()
    store = MemoryStore(settings.database_path)
    llm = LLMClient(settings)
    return store, llm, Orchestrator(settings, store, llm)


def _print_evidence(result: ChatResponse) -> None:
    if not result.evidence:
        return
    console.print("\nSources:", style="dim")
    for reference in result.evidence:
        console.print(
            f"[{reference.label}] {reference.title} — {reference.source} "
            f"(trust={reference.trust_level})",
            style="dim",
            markup=False,
        )


@app.command()
def serve(port: int = 8000, reload: bool = False) -> None:
    """Start the local-only HTTP API."""
    uvicorn.run(
        "powerhouse_ai.api:create_app",
        host="127.0.0.1",
        port=port,
        reload=reload,
        factory=True,
    )


@app.command()
def status() -> None:
    """Show GPU, model-server, and database status."""
    settings = get_settings()
    gpu = get_gpu_status()
    llm = LLMClient(settings)
    table = Table(title="Powerhouse AI")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Model", settings.model_name)
    table.add_row("Model endpoint", settings.model_base_url)
    table.add_row("Model online", str(llm.health()))
    table.add_row("Database", str(settings.database_path))
    table.add_row("Mode", settings.mode)
    table.add_row("GPU", gpu.name or "unavailable")
    table.add_row("VRAM", f"{gpu.used_mb or 0} / {gpu.total_mb or 0} MB")
    table.add_row("VRAM budget", f"{settings.vram_limit_mb} MB")
    python_tool_status = "ENABLED (trusted code only)" if settings.enable_python_tool else "off"
    table.add_row("Python tool", python_tool_status)
    console.print(table)
    llm.close()


@app.command("create-project")
def create_project(name: str, description: str = "") -> None:
    """Create a persistent project and print its ID."""
    settings = get_settings()
    store = MemoryStore(settings.database_path)
    project = store.create_project(name, description)
    console.print_json(project.model_dump_json())


@app.command()
def projects() -> None:
    """List persistent projects."""
    settings = get_settings()
    store = MemoryStore(settings.database_path)
    console.print_json(json.dumps([item.model_dump(mode="json") for item in store.list_projects()]))


@app.command()
def ingest(
    path: Path,
    domain: Domain,
    project_id: str | None = None,
    trust_level: str = "medium",
) -> None:
    """Index local text/code files for project-aware retrieval."""
    settings = get_settings()
    store = MemoryStore(settings.database_path)
    if not path.exists():
        raise typer.BadParameter("Path does not exist.", param_hint="path")
    if project_id and store.get_project(project_id) is None:
        raise typer.BadParameter("Project was not found.", param_hint="project_id")
    supported = {".txt", ".md", ".rst", ".py", ".json", ".csv", ".yaml", ".yml"}
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    created = 0
    accepted = 0
    for file in files:
        if file.suffix.lower() not in supported:
            continue
        content = file.read_text(encoding="utf-8", errors="replace")
        ids = store.add_document(
            DocumentAdd(
                project_id=project_id,
                domain=domain,
                title=file.name,
                content=content,
                source=str(file.resolve()),
                trust_level=trust_level,
            )
        )
        accepted += 1
        created += len(ids)
    console.print(f"Indexed {accepted} files into {created} searchable chunks.")


@app.command()
def chat(
    project_id: str,
    message: str,
    mode: ExecutionMode | None = None,
) -> None:
    """Send one request through the complete local orchestration pipeline."""
    store, llm, orchestrator = _components()
    try:
        result = orchestrator.process(project_id, message, mode=mode)
        console.print(result.answer)
        _print_evidence(result)
        console.print(
            f"\n[dim]{result.model_calls} model calls, "
            f"{result.latency_seconds}s, peak VRAM {result.peak_vram_mb} MB[/dim]"
        )
    finally:
        llm.close()


@app.command()
def ask(message: str, mode: ExecutionMode | None = None) -> None:
    """Ask immediately, creating and reusing a default persistent project."""
    store, llm, orchestrator = _components()
    try:
        project = next(
            (item for item in store.list_projects() if item.name == "Default Workspace"),
            None,
        )
        if project is None:
            project = store.create_project(
                "Default Workspace",
                "Automatically created persistent Powerhouse AI workspace.",
            )
        result = orchestrator.process(project.id, message, mode=mode)
        console.print(result.answer)
        _print_evidence(result)
        console.print(
            f"\n[dim]project {project.id}, {result.model_calls} model calls, "
            f"{result.latency_seconds}s, peak VRAM {result.peak_vram_mb} MB[/dim]"
        )
    finally:
        llm.close()


if __name__ == "__main__":
    app()
