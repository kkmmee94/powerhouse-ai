from pathlib import Path

import pytest

from powerhouse_ai.config import Settings
from powerhouse_ai.errors import ModelResponseError
from powerhouse_ai.llm import ModelReply
from powerhouse_ai.orchestrator import Orchestrator
from powerhouse_ai.schemas import (
    DocumentAdd,
    Domain,
    DomainScore,
    RouteDecision,
    ToolName,
)
from powerhouse_ai.storage import MemoryStore
from powerhouse_ai.tools import ToolRegistry


class FixedRouter:
    @staticmethod
    def route(
        user_prompt: str,
        *,
        force_llm: bool = False,
        allow_llm: bool = True,
    ) -> RouteDecision:
        return RouteDecision(
            goal=user_prompt,
            domains=[
                DomainScore(name=Domain.BIOLOGY, importance=0.5),
                DomainScore(name=Domain.MATHEMATICS, importance=0.3),
                DomainScore(name=Domain.PROGRAMMING, importance=0.2),
            ],
            tools=[ToolName.CALCULATOR],
            complexity=0.7,
            requires_multiple_steps=True,
            confidence=1,
            source="test",
        )


class FakeLLM:
    def __init__(self) -> None:
        self.call_count = 0
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs) -> ModelReply:
        self.call_count += 1
        self.prompts.append(str(messages[-1]["content"]))
        return ModelReply(content=f"completed call {self.call_count}", tool_calls=[])


def test_multi_stage_orchestration_and_memory(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "orchestrator.db", mode="balanced")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Integrated")
    biology_evidence_id = store.add_document(
        DocumentAdd(
            project_id=project.id,
            domain=Domain.BIOLOGY,
            title="Growth protocol",
            content="The bacterial growth simulator uses a biologically constrained population.",
            source="local://biology",
            trust_level="high",
        )
    )[0]
    mathematics_evidence_id = store.add_document(
        DocumentAdd(
            project_id=project.id,
            domain=Domain.MATHEMATICS,
            title="Growth equation",
            content="The mathematical bacterial growth simulator uses a logistic equation.",
            source="local://mathematics",
            trust_level="high",
        )
    )[0]
    llm = FakeLLM()
    orchestrator = Orchestrator(
        settings,
        store,
        llm,  # type: ignore[arg-type]
        router=FixedRouter(),  # type: ignore[arg-type]
    )
    response = orchestrator.process(
        project.id,
        "Build a mathematical bacterial growth simulator.",
        mode="balanced",
    )
    assert response.model_calls == 4
    assert response.answer == "completed call 4"
    assert response.evidence == []
    assert [step.domain for step in response.plan.steps] == [
        Domain.BIOLOGY,
        Domain.MATHEMATICS,
        Domain.PROGRAMMING,
    ]
    saved = store.search_memory(project.id, "bacterial growth simulator")
    assert saved
    state = store.project_state(project.id)
    assert state["latest_goal"]["goal"] == "Build a mathematical bacterial growth simulator."
    assert len(state["steps"]) == 3
    assert state["steps"][0]["evidence_ids"] == [biology_evidence_id]
    assert state["steps"][1]["evidence_ids"] == [mathematics_evidence_id]


class CitingLLM(FakeLLM):
    def chat(self, messages, **kwargs) -> ModelReply:
        self.call_count += 1
        return ModelReply(content="The evidence supports this answer [E1].", tool_calls=[])


def test_response_exposes_only_evidence_cited_by_the_final_answer(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "cited-evidence.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Cited evidence")
    store.add_document(
        DocumentAdd(
            project_id=project.id,
            domain=Domain.BIOLOGY,
            title="ATP synthase",
            content="ATP synthase uses a proton gradient.",
            source="local://biology",
            trust_level="high",
        )
    )
    llm = CitingLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    response = orchestrator.process(
        project.id,
        "Explain how the proton gradient drives ATP synthase.",
        mode="fast",
    )

    assert [(item.label, item.source) for item in response.evidence] == [
        ("E1", "local://biology")
    ]


def test_fast_mode_preserves_full_route_but_executes_one_stage(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "fast-route.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Fast route")
    llm = FakeLLM()
    orchestrator = Orchestrator(
        settings,
        store,
        llm,  # type: ignore[arg-type]
        router=FixedRouter(),  # type: ignore[arg-type]
    )

    response = orchestrator.process(project.id, "Build a growth simulator.", mode="fast")

    assert [item.name for item in response.route.domains] == [
        Domain.BIOLOGY,
        Domain.MATHEMATICS,
        Domain.PROGRAMMING,
    ]
    assert [step.domain for step in response.plan.steps] == [Domain.BIOLOGY]
    assert response.model_calls == 1


def test_fast_mode_prefers_an_explicit_code_deliverable(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "fast-code.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Fast code")
    orchestrator = Orchestrator(
        settings,
        store,
        FakeLLM(),  # type: ignore[arg-type]
        router=FixedRouter(),  # type: ignore[arg-type]
    )

    response = orchestrator.process(
        project.id,
        "Write tested Python code for a bacterial growth simulator.",
        mode="fast",
    )

    assert [step.domain for step in response.plan.steps] == [Domain.PROGRAMMING]


class ReportRouter:
    @staticmethod
    def route(
        user_prompt: str,
        *,
        force_llm: bool = False,
        allow_llm: bool = True,
    ) -> RouteDecision:
        return RouteDecision(
            goal=user_prompt,
            domains=[
                DomainScore(name=Domain.STATISTICS, importance=1.0),
                DomainScore(name=Domain.RESEARCH, importance=0.8),
                DomainScore(name=Domain.WRITING, importance=0.7),
            ],
            complexity=0.8,
            requires_multiple_steps=True,
            confidence=1,
            source="test",
        )


def test_fast_mode_prefers_an_explicit_report_deliverable(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "fast-report.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Fast report")
    orchestrator = Orchestrator(
        settings,
        store,
        FakeLLM(),  # type: ignore[arg-type]
        router=ReportRouter(),  # type: ignore[arg-type]
    )

    response = orchestrator.process(
        project.id,
        "Write a cautious report about the observational results.",
        mode="fast",
    )

    assert [step.domain for step in response.plan.steps] == [Domain.WRITING]


class HighComplexityRouter:
    @staticmethod
    def route(
        user_prompt: str,
        *,
        force_llm: bool = False,
        allow_llm: bool = True,
    ) -> RouteDecision:
        return RouteDecision(
            goal=user_prompt,
            domains=[DomainScore(name=Domain.GENERAL, importance=1.0)],
            complexity=0.9,
            confidence=1,
            source="test",
        )


def test_fast_mode_does_not_add_a_verification_stage(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "fast-complex.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Fast complex")
    llm = FakeLLM()
    orchestrator = Orchestrator(
        settings,
        store,
        llm,  # type: ignore[arg-type]
        router=HighComplexityRouter(),  # type: ignore[arg-type]
    )

    response = orchestrator.process(project.id, "A complex request.", mode="fast")

    assert response.answer == "completed call 1"
    assert response.model_calls == 1


class ToolCallingLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, messages, **kwargs) -> ModelReply:
        self.call_count += 1
        if self.call_count == 1:
            return ModelReply(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"17 * 19"}',
                        },
                    }
                ],
            )
        assert messages[-1]["role"] == "tool"
        assert "323" in messages[-1]["content"]
        return ModelReply(content="The verified result is 323.", tool_calls=[])


def test_tool_call_round_trip(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "tools.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Calculator")
    llm = ToolCallingLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]
    response = orchestrator.process(project.id, "Calculate 17 times 19.", mode="fast")
    assert response.answer == "The verified result is 323."
    assert response.model_calls == 2


class DisallowedToolCallingLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, messages, **kwargs) -> ModelReply:
        self.call_count += 1
        if self.call_count == 1:
            return ModelReply(
                content="",
                tool_calls=[
                    {
                        "id": "disallowed-call",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code":"print(40 + 2)"}',
                        },
                    }
                ],
            )
        assert "not allowed" in messages[-1]["content"]
        return ModelReply(content="The requested tool was rejected.", tool_calls=[])


def test_tool_loop_rejects_a_model_call_outside_the_stage_allowlist(
    tmp_path: Path,
) -> None:
    settings = Settings(
        database_path=tmp_path / "tool-allowlist.db",
        enable_python_tool=True,
    )
    store = MemoryStore(settings.database_path)
    llm = DisallowedToolCallingLLM()
    orchestrator = Orchestrator(
        settings,
        store,
        llm,  # type: ignore[arg-type]
        tools=ToolRegistry(enable_python=True),
    )

    answer, results = orchestrator._tool_loop(
        [{"role": "user", "content": "Calculate a value."}],
        [ToolName.CALCULATOR],
        enable_thinking=False,
    )

    assert answer == "The requested tool was rejected."
    assert results == [
        {
            "name": "python",
            "ok": False,
            "error": "Tool is not allowed for this stage: python",
        }
    ]


def test_unambiguous_calculation_returns_authoritative_tool_result(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "direct-calculator.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Direct calculator")
    llm = FakeLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    response = orchestrator.process(
        project.id,
        "Calculate sqrt(144) + 3^2 using the calculator tool.",
        mode="fast",
    )

    assert response.answer == (
        "Calculator tool result: **21**.\n\n"
        "Verification: `sqrt(144) + 3^2 = 21`."
    )
    assert response.model_calls == 0


def test_partial_parenthesized_calculation_is_not_precomputed(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "parenthesized-calculator.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Parenthesized calculator")
    llm = FakeLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    response = orchestrator.process(
        project.id,
        "Calculate (2+3)*4 using the calculator tool.",
        mode="fast",
    )

    assert response.answer == "completed call 1"
    assert response.model_calls == 1
    assert llm.prompts
    assert '"result": 5' not in llm.prompts[0]


def test_unambiguous_conversion_returns_authoritative_tool_result(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "direct-conversion.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Direct conversion")
    llm = FakeLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    response = orchestrator.process(
        project.id,
        "Convert 2.5 kg to g using the unit conversion tool and state the exact result.",
        mode="fast",
    )

    assert response.answer == (
        "Unit conversion tool result: **2500 g**.\n\n"
        "Verification: `2.5 kg = 2500 g`."
    )
    assert response.model_calls == 0


def test_calculation_shortcut_does_not_discard_an_additional_request(
    tmp_path: Path,
) -> None:
    settings = Settings(database_path=tmp_path / "multi-part.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Multi-part")
    llm = FakeLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    response = orchestrator.process(
        project.id,
        "Calculate 2+2 and tell me the capital of France.",
        mode="fast",
    )

    assert response.answer == "completed call 1"
    assert response.model_calls == 1


def test_orchestrator_rejects_unknown_mode(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "mode.db")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Modes")
    orchestrator = Orchestrator(settings, store, FakeLLM())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        orchestrator.process(project.id, "Hello", mode="turbo")


def test_orchestrator_loads_packaged_modules_outside_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(database_path=tmp_path / "resources.db")
    store = MemoryStore(settings.database_path)
    orchestrator = Orchestrator(settings, store, FakeLLM())  # type: ignore[arg-type]
    assert "senior software engineer" in orchestrator._module_prompts[Domain.PROGRAMMING]


class FailingLLM:
    call_count = 0

    @staticmethod
    def chat(*args, **kwargs) -> ModelReply:
        raise ModelResponseError("bad model response")


def test_failed_request_does_not_persist_an_incomplete_turn(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "failed-turn.db")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Failure")
    orchestrator = Orchestrator(settings, store, FailingLLM())  # type: ignore[arg-type]

    with pytest.raises(ModelResponseError, match="bad model response"):
        orchestrator.process(project.id, "Hello")

    assert store.recent_messages(project.id) == []


def test_unrelated_recent_memory_is_not_injected(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "memory-filter.db", mode="fast")
    store = MemoryStore(settings.database_path)
    project = store.create_project("Filtered memory")
    store.add_memory(project.id, "The project colour is cobalt blue.")
    llm = FakeLLM()
    orchestrator = Orchestrator(settings, store, llm)  # type: ignore[arg-type]

    orchestrator.process(project.id, "Write a Python context manager.", mode="fast")

    assert llm.prompts
    assert all("cobalt blue" not in prompt for prompt in llm.prompts)
