from __future__ import annotations

from powerhouse_ai.schemas import Domain, PlanStep, RouteDecision, TaskPlan, ToolName

ORDER = {
    Domain.RESEARCH: 10,
    Domain.BIOLOGY: 20,
    Domain.CHEMISTRY: 20,
    Domain.STATISTICS: 30,
    Domain.MATHEMATICS: 30,
    Domain.PROGRAMMING: 40,
    Domain.WRITING: 50,
    Domain.GENERAL: 50,
}


DOMAIN_GOALS = {
    Domain.RESEARCH: "Find and assess the evidence required for the goal.",
    Domain.BIOLOGY: "Establish biologically valid facts, mechanisms, and constraints.",
    Domain.CHEMISTRY: "Establish chemically valid facts, conditions, and constraints.",
    Domain.STATISTICS: "Select and apply a statistically valid analysis.",
    Domain.MATHEMATICS: "Formulate and check the mathematical reasoning.",
    Domain.PROGRAMMING: "Implement and test the required software.",
    Domain.WRITING: "Present the result clearly for the intended audience.",
    Domain.GENERAL: "Solve the user's goal accurately and directly.",
}


class Planner:
    """Deterministic planning avoids spending an extra model call on an already clear route."""

    def create(self, route: RouteDecision) -> TaskPlan:
        domains = sorted(
            (score.name for score in route.domains),
            key=lambda domain: (ORDER[domain], domain.value),
        )
        if not domains:
            domains = [Domain.GENERAL]
        if not route.requires_multiple_steps:
            domain = domains[0]
            return TaskPlan(
                goal=route.goal,
                steps=[
                    PlanStep(
                        id=1,
                        domain=domain,
                        goal=route.goal,
                        tools=self._tools_for(domain, route.tools),
                    )
                ],
            )

        steps: list[PlanStep] = []
        for index, domain in enumerate(domains, start=1):
            dependencies = [index - 1] if index > 1 else []
            steps.append(
                PlanStep(
                    id=index,
                    domain=domain,
                    goal=f"{DOMAIN_GOALS[domain]} Overall objective: {route.goal}",
                    dependencies=dependencies,
                    tools=self._tools_for(domain, route.tools),
                )
            )
        return TaskPlan(goal=route.goal, steps=steps)

    @staticmethod
    def _tools_for(domain: Domain, requested: list[ToolName]) -> list[ToolName]:
        relevant: dict[Domain, set[ToolName]] = {
            Domain.PROGRAMMING: {
                ToolName.PYTHON,
                ToolName.CODE_TESTS,
                ToolName.FILE_SEARCH,
                ToolName.DOCUMENT_RETRIEVAL,
            },
            Domain.MATHEMATICS: {
                ToolName.CALCULATOR,
                ToolName.UNIT_CONVERSION,
                ToolName.PYTHON,
            },
            Domain.STATISTICS: {ToolName.CALCULATOR, ToolName.PYTHON},
            Domain.CHEMISTRY: {
                ToolName.CALCULATOR,
                ToolName.UNIT_CONVERSION,
                ToolName.DOCUMENT_RETRIEVAL,
            },
            Domain.BIOLOGY: {ToolName.DOCUMENT_RETRIEVAL},
            Domain.RESEARCH: {
                ToolName.DOCUMENT_RETRIEVAL,
                ToolName.FILE_SEARCH,
            },
        }
        allowed = relevant.get(domain, set())
        return [tool for tool in requested if tool in allowed]
