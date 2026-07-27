from datetime import UTC, datetime

import powerhouse_ai.cli
from powerhouse_ai.schemas import (
    ChatResponse,
    Domain,
    DomainScore,
    PlanStep,
    Project,
    RouteDecision,
    TaskPlan,
)


def test_cli_uses_configured_mode_when_flag_is_omitted(monkeypatch) -> None:
    project = Project(
        id="project-id",
        name="Default Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    route = RouteDecision(
        goal="hello",
        domains=[DomainScore(name=Domain.GENERAL, importance=1.0)],
    )
    plan = TaskPlan(
        goal="hello",
        steps=[PlanStep(id=1, domain=Domain.GENERAL, goal="hello")],
    )

    class Store:
        @staticmethod
        def list_projects() -> list[Project]:
            return [project]

    class LLM:
        @staticmethod
        def close() -> None:
            pass

    class Runtime:
        modes: list[object] = []

        @classmethod
        def process(cls, project_id: str, message: str, mode=None) -> ChatResponse:
            cls.modes.append(mode)
            return ChatResponse(
                project_id=project_id,
                answer=message,
                route=route,
                plan=plan,
                latency_seconds=0.01,
                model_calls=1,
            )

    monkeypatch.setattr(
        powerhouse_ai.cli,
        "_components",
        lambda: (Store(), LLM(), Runtime()),
    )

    powerhouse_ai.cli.ask("hello")
    powerhouse_ai.cli.chat(project.id, "hello")

    assert Runtime.modes == [None, None]


def test_serve_is_loopback_only(monkeypatch) -> None:
    observed = {}

    def fake_run(application, **kwargs) -> None:
        observed["application"] = application
        observed.update(kwargs)

    monkeypatch.setattr(powerhouse_ai.cli.uvicorn, "run", fake_run)

    powerhouse_ai.cli.serve(port=8123, reload=True)

    assert observed == {
        "application": "powerhouse_ai.api:create_app",
        "host": "127.0.0.1",
        "port": 8123,
        "reload": True,
        "factory": True,
    }
