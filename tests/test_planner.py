from powerhouse_ai.planner import Planner
from powerhouse_ai.schemas import Domain, DomainScore, RouteDecision, ToolName


def test_planner_orders_science_before_math_before_code() -> None:
    route = RouteDecision(
        goal="Build a bacterial growth simulator",
        domains=[
            DomainScore(name=Domain.PROGRAMMING, importance=0.4),
            DomainScore(name=Domain.MATHEMATICS, importance=0.3),
            DomainScore(name=Domain.BIOLOGY, importance=0.3),
        ],
        tools=[ToolName.CALCULATOR, ToolName.PYTHON, ToolName.CODE_TESTS],
        complexity=0.8,
        requires_multiple_steps=True,
    )
    plan = Planner().create(route)
    assert [step.domain for step in plan.steps] == [
        Domain.BIOLOGY,
        Domain.MATHEMATICS,
        Domain.PROGRAMMING,
    ]
    assert plan.steps[2].dependencies == [2]
