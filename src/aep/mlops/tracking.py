"""Thin MLflow helper (Stage 3 / Stage 7).

Centralizes tracking-URI and experiment setup so every component logs to the
same place. Defaults to local file-based tracking (``./mlruns``); override with
the ``MLFLOW_TRACKING_URI`` environment variable for a remote server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from aep.config import get_settings


def configure() -> str:
    """Point MLflow at the configured tracking URI + experiment. Returns the URI."""
    # Keep the simple local file store working without extra setup (recent MLflow
    # requires an explicit opt-in for the file backend).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(get_settings().mlflow_experiment)
    return uri


@contextmanager
def start_run(run_name: str, nested: bool = False) -> Iterator[mlflow.ActiveRun]:
    """Context manager wrapping ``mlflow.start_run`` after configuration."""
    configure()
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run
