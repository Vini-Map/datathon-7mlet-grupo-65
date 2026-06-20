"""Central, environment-driven configuration.

All tunables come from environment variables (optionally loaded from a local
``.env`` — never committed). See ``.env.example`` for the documented set.
Keeping configuration in one typed place supports reproducibility: the master
seed and tracking URI are read from here everywhere.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (src/aep/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AEP_",
        extra="ignore",
    )

    # Reproducibility
    random_seed: int = 42

    # LLM provider: "mock" (offline, default) or "anthropic"
    llm_provider: str = "mock"
    anthropic_model: str = "claude-opus-4-8"

    # MLflow
    mlflow_experiment: str = "aep-bandits"

    # Service
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Data
    data_dir: Path = Field(default=REPO_ROOT / "data")


def get_settings() -> Settings:
    """Return a freshly-loaded :class:`Settings` instance."""
    return Settings()
