from pathlib import Path

import pytest

from powerhouse_ai.router import Router
from powerhouse_ai.schemas import Domain, ToolName


def test_router_detects_multi_domain_scientific_code() -> None:
    route = Router().heuristic(
        "Write Python code using a differential equation to model bacterial population growth."
    )
    domains = {item.name for item in route.domains}
    assert Domain.BIOLOGY in domains
    assert Domain.MATHEMATICS in domains
    assert Domain.PROGRAMMING in domains
    assert route.requires_multiple_steps is True
    assert ToolName.CALCULATOR in route.tools
    assert ToolName.PYTHON in route.tools


def test_router_has_general_fallback() -> None:
    route = Router().heuristic("Hello there")
    assert route.domains[0].name is Domain.GENERAL


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Refactor this endpoint to eliminate redundant ORM round trips.", {Domain.PROGRAMMING}),
        (
            "Quantify sampling error and variance across the treatment groups.",
            {Domain.STATISTICS},
        ),
        ("Describe population dynamics in a changing ecosystem.", {Domain.BIOLOGY}),
        ("Formulate an ODE and include dimensional units.", {Domain.MATHEMATICS}),
        ("Produce a concise report for a non-technical audience.", {Domain.WRITING}),
        (
            "Assess the reliability of peer-reviewed findings and cite the evidence.",
            {Domain.RESEARCH},
        ),
    ],
)
def test_router_handles_paraphrases_outside_the_benchmark_wording(
    prompt: str,
    expected: set[Domain],
) -> None:
    routed = {item.name for item in Router().heuristic(prompt).domains}
    assert expected <= routed


class FailingRouteLLM:
    @staticmethod
    def chat_json(*args, **kwargs):
        raise AssertionError("A confident deterministic route should not spend a model call.")


def test_confident_complex_route_stays_deterministic() -> None:
    route = Router(FailingRouteLLM()).route(  # type: ignore[arg-type]
        "Create tested Python code to simulate logistic bacterial growth."
    )
    domains = {item.name for item in route.domains}
    assert domains == {Domain.BIOLOGY, Domain.MATHEMATICS, Domain.PROGRAMMING}
    assert route.source == "heuristic"


def test_router_loads_packaged_prompt_outside_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    router = Router()
    assert "routing controller" in router.prompt
