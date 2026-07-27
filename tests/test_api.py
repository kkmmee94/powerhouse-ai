from pathlib import Path

from fastapi.testclient import TestClient

import powerhouse_ai.api
from powerhouse_ai.api import create_app
from powerhouse_ai.config import Settings
from powerhouse_ai.schemas import (
    ChatResponse,
    Domain,
    DomainScore,
    PlanStep,
    RouteDecision,
    TaskPlan,
)
from powerhouse_ai.storage import MemoryStore


class FakeLLM:
    call_count = 0

    def health(self) -> bool:
        return True

    def close(self) -> None:
        pass


class FakeOrchestrator:
    _module_prompts = {Domain.GENERAL: "general"}

    class Tools:
        @staticmethod
        def definitions():
            return []

    tools = Tools()

    @staticmethod
    def process(project_id: str, message: str, mode: str | None = None) -> ChatResponse:
        route = RouteDecision(
            goal=message,
            domains=[DomainScore(name=Domain.GENERAL, importance=1.0)],
        )
        plan = TaskPlan(
            goal=message,
            steps=[PlanStep(id=1, domain=Domain.GENERAL, goal=message)],
        )
        return ChatResponse(
            project_id=project_id,
            answer="local answer",
            route=route,
            plan=plan,
            latency_seconds=0.01,
            model_calls=1,
        )


def test_project_and_chat_api(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")
    store = MemoryStore(settings.database_path)
    app = create_app(
        settings,
        store=store,
        llm=FakeLLM(),  # type: ignore[arg-type]
        orchestrator=FakeOrchestrator(),  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        status = client.get("/status")
        assert status.status_code == 200
        assert status.json()["python_tool_enabled"] is False

        project_response = client.post("/projects", json={"name": "Test"})
        assert project_response.status_code == 201
        project_id = project_response.json()["id"]
        assert client.get("/projects").json()[0]["id"] == project_id
        assert client.get(f"/projects/{project_id}").json()["name"] == "Test"
        assert client.get("/projects/missing").status_code == 404

        document = client.post(
            "/documents",
            json={
                "project_id": project_id,
                "domain": "general",
                "title": "Release note",
                "content": "The release codename is cobalt.",
                "source": "test",
                "trust_level": "high",
            },
        )
        assert document.status_code == 201
        assert document.json()["chunks_created"] == 1
        assert client.post(
            "/documents",
            json={
                "project_id": "missing",
                "domain": "general",
                "title": "Missing",
                "content": "Not stored.",
            },
        ).status_code == 404

        modules = client.get("/modules")
        assert modules.status_code == 200
        assert modules.json()["domains"] == ["general"]

        response = client.post(
            "/chat",
            json={"project_id": project_id, "message": "Hello"},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "local answer"

        invalid_mode = client.post(
            "/chat",
            json={"project_id": project_id, "message": "Hello", "mode": "turbo"},
        )
        assert invalid_mode.status_code == 422

        invalid_limit = client.get(
            f"/projects/{project_id}/memory",
            params={"query": "anything", "limit": -1},
        )
        assert invalid_limit.status_code == 422

        state = client.get(f"/projects/{project_id}/state")
        assert state.status_code == 200
        assert state.json()["latest_goal"] is None

        missing = client.get("/projects/missing/state")
        assert missing.status_code == 404


def test_api_module_uses_an_explicit_factory() -> None:
    assert not hasattr(powerhouse_ai.api, "app")
