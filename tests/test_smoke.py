"""Stage 0 smoke tests: the package imports, version is exposed, settings load
with documented defaults, and the CLI is wired up. These guard the foundation
so later stages build on a known-good base."""

from __future__ import annotations

from typer.testing import CliRunner

import aep
from aep.cli import app
from aep.config import Settings, get_settings

runner = CliRunner()


def test_version_is_exposed() -> None:
    assert aep.__version__ == "0.1.0"


def test_settings_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    # Documented defaults from .env.example.
    assert settings.random_seed == 42
    assert settings.llm_provider in {"mock", "anthropic"}


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_help_lists_stage_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "train", "eval", "serve", "demo"):
        assert command in result.stdout
