from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query

from powerhouse_ai import __version__
from powerhouse_ai.config import Settings, get_settings
from powerhouse_ai.errors import ModelUnavailableError, PowerhouseError
from powerhouse_ai.gpu import get_gpu_status
from powerhouse_ai.llm import LLMClient
from powerhouse_ai.orchestrator import Orchestrator
from powerhouse_ai.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentAdd,
    Project,
    ProjectCreate,
    RuntimeStatus,
)
from powerhouse_ai.storage import MemoryStore


def create_app(
    settings: Settings | None = None,
    *,
    store: MemoryStore | None = None,
    llm: LLMClient | None = None,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    active_store = store or MemoryStore(active_settings.database_path)
    active_llm = llm or LLMClient(active_settings)
    active_orchestrator = orchestrator or Orchestrator(
        active_settings,
        active_store,
        active_llm,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        active_llm.close()

    app = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.store = active_store
    app.state.llm = active_llm
    app.state.orchestrator = active_orchestrator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status", response_model=RuntimeStatus)
    def status() -> RuntimeStatus:
        gpu = get_gpu_status()
        within = None if gpu.used_mb is None else gpu.used_mb <= active_settings.vram_limit_mb
        return RuntimeStatus(
            app=active_settings.app_name,
            version=__version__,
            model_name=active_settings.model_name,
            model_online=active_llm.health(),
            mode=active_settings.mode,
            database_path=str(active_settings.database_path),
            gpu=gpu,
            vram_limit_mb=active_settings.vram_limit_mb,
            vram_within_budget=within,
            python_tool_enabled=active_settings.enable_python_tool,
        )

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if not active_llm.health():
            raise HTTPException(status_code=503, detail="Local model server is unavailable.")
        return {"status": "ready"}

    @app.post("/projects", response_model=Project, status_code=201)
    def create_project(request: ProjectCreate) -> Project:
        return active_store.create_project(request.name, request.description)

    @app.get("/projects", response_model=list[Project])
    def list_projects() -> list[Project]:
        return active_store.list_projects()

    @app.get("/projects/{project_id}", response_model=Project)
    def get_project(project_id: str) -> Project:
        project = active_store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return project

    @app.get("/projects/{project_id}/memory")
    def get_memory(
        project_id: str,
        query: str,
        limit: Annotated[int, Query(ge=1, le=20)] = 8,
    ) -> dict[str, Any]:
        if active_store.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return {
            "project_id": project_id,
            "query": query,
            "items": active_store.search_memory(project_id, query, limit=limit),
        }

    @app.get("/projects/{project_id}/state")
    def get_project_state(project_id: str) -> dict[str, Any]:
        if active_store.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        return active_store.project_state(project_id)

    @app.post("/documents", status_code=201)
    def add_document(document: DocumentAdd) -> dict[str, Any]:
        if document.project_id and active_store.get_project(document.project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        ids = active_store.add_document(document)
        return {"chunks_created": len(ids), "ids": ids}

    @app.get("/modules")
    def modules() -> dict[str, Any]:
        return {
            "domains": sorted(domain.value for domain in active_orchestrator._module_prompts),
            "tools": [
                definition["function"]["name"]
                for definition in active_orchestrator.tools.definitions()
            ],
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            return active_orchestrator.process(
                request.project_id,
                request.message,
                mode=request.mode,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PowerhouseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
