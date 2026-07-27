from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from powerhouse_ai.config import ExecutionMode


class Domain(StrEnum):
    GENERAL = "general"
    PROGRAMMING = "programming"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    MATHEMATICS = "mathematics"
    STATISTICS = "statistics"
    WRITING = "writing"
    RESEARCH = "research"


class ToolName(StrEnum):
    CALCULATOR = "calculator"
    PYTHON = "python"
    CODE_TESTS = "code_tests"
    FILE_SEARCH = "file_search"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    UNIT_CONVERSION = "unit_conversion"


class DomainScore(BaseModel):
    name: Domain
    importance: float = Field(ge=0.0, le=1.0)


class RouteDecision(BaseModel):
    goal: str = Field(min_length=1, max_length=500)
    domains: list[DomainScore] = Field(default_factory=list)
    tools: list[ToolName] = Field(default_factory=list)
    complexity: float = Field(default=0.3, ge=0.0, le=1.0)
    requires_multiple_steps: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "heuristic"


class PlanStep(BaseModel):
    id: int = Field(ge=1)
    domain: Domain
    goal: str
    dependencies: list[int] = Field(default_factory=list)
    tools: list[ToolName] = Field(default_factory=list)


class TaskPlan(BaseModel):
    goal: str
    steps: list[PlanStep]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class Project(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    project_id: str
    message: str = Field(min_length=1, max_length=100_000)
    mode: ExecutionMode | None = None


class StepResult(BaseModel):
    step: PlanStep
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    label: str = Field(pattern=r"^E[1-9]\d*$")
    id: str
    title: str
    domain: Domain
    source: str
    trust_level: str


class ChatResponse(BaseModel):
    project_id: str
    answer: str
    route: RouteDecision
    plan: TaskPlan
    latency_seconds: float
    model_calls: int
    peak_vram_mb: int | None = None
    warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentAdd(BaseModel):
    project_id: str | None = None
    domain: Domain
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=5_000_000)
    source: str = Field(default="user", max_length=1000)
    trust_level: str = Field(default="medium", pattern="^(low|medium|high)$")


class Evidence(BaseModel):
    id: str
    title: str
    content: str
    domain: Domain
    source: str
    trust_level: str
    score: float = 0.0


class GPUStatus(BaseModel):
    available: bool
    name: str | None = None
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    utilization_percent: int | None = None
    temperature_c: int | None = None
    error: str | None = None


class RuntimeStatus(BaseModel):
    app: str
    version: str
    model_name: str
    model_online: bool
    mode: str
    database_path: str
    gpu: GPUStatus
    vram_limit_mb: int
    vram_within_budget: bool | None
    python_tool_enabled: bool
