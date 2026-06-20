<#
.SYNOPSIS
    PowerShell wrapper that mirrors the Makefile targets, since `make` is not
    available on a default Windows install. Usage:  ./make.ps1 <target>
    Run  ./make.ps1 help  to list targets.
#>
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

function Invoke-Step($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $cmd" }
}

switch ($Target.ToLower()) {
    "help" {
        Write-Host "Available targets:" -ForegroundColor Green
        @(
            "  setup       Create the venv and install core + dev dependencies",
            "  setup-all   Install ALL dependency groups (eda, ml, service, rag, llm)",
            "  data        (Stage 1-2) Process Kaggle base + generate synthetic enrichment",
            "  train       (Stage 3) Train/simulate bandit policies and log to MLflow",
            "  eval        (Stage 4) Run reproducible offline evaluation vs golden set",
            "  serve       (Stage 5) Start the FastAPI decision service",
            "  demo        (Stage 5) Run the end-to-end pipeline locally",
            "  test        Run the test suite",
            "  lint        Lint with ruff",
            "  format      Format with black + ruff import sort",
            "  clean       Remove caches and build artifacts"
        ) | ForEach-Object { Write-Host $_ }
    }
    "setup"     { Invoke-Step "uv sync" }
    "setup-all" { Invoke-Step "uv sync --all-groups" }
    "data"      { Invoke-Step "uv run aep data --help" }
    "train"     { Invoke-Step "uv run aep train --help" }
    "eval"      { Invoke-Step "uv run aep eval --help" }
    "serve"     { Invoke-Step "uv run aep serve --help" }
    "demo"      { Invoke-Step "uv run aep demo --help" }
    "test"      { Invoke-Step "uv run pytest" }
    "lint"      { Invoke-Step "uv run ruff check src tests" }
    "format"    {
        Invoke-Step "uv run black src tests"
        Invoke-Step "uv run ruff check --fix src tests"
    }
    "clean" {
        @(".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist", "htmlcov") |
            ForEach-Object { if (Test-Path $_) { Remove-Item -Recurse -Force $_ } }
    }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host "Run  ./make.ps1 help  to list targets."
        exit 1
    }
}
