from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    POWER = "power"
    ADAPTIVE = "adaptive"


class Settings(BaseSettings):
    """Runtime configuration, overridable through POWERHOUSE_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="POWERHOUSE_",
        extra="ignore",
    )

    app_name: str = "Powerhouse AI"
    model_base_url: str = "http://127.0.0.1:8080/v1"
    model_name: str = "local-model"
    model_api_key: str = "local"
    database_path: Path = Path("data/powerhouse.db")
    mode: ExecutionMode = ExecutionMode.ADAPTIVE
    context_tokens: int = Field(default=8192, ge=2048, le=262144)
    max_output_tokens: int = Field(default=4096, ge=128, le=32768)
    vram_limit_mb: int = Field(default=15872, ge=4096)
    request_timeout_seconds: float = Field(default=600.0, ge=10.0)
    model_enable_thinking: bool = False
    enable_python_tool: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
