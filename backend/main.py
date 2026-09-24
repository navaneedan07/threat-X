"""Threat-X API entrypoint.

Run locally:  uvicorn backend.main:app --reload
Docs:         http://localhost:8000/docs

Only /health exists at this stage. The threat endpoints listed in the README
(GET /api/v1/threats, ...) must not be added until they can serve real pipeline
output -- see the README's "Responsible Use" section.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import __version__


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log the configuration the service came up with, once wiring exists."""
    yield


app = FastAPI(
    title="Threat-X -- Extreme Weather Threat Intelligence API",
    description=(
        "Machine-readable threat information for tracked extreme-weather anomalies. "
        "This is a research/decision-support prototype, not an official warning service."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, object]:
    """Liveness probe.

    `pipeline_stages` reports which stages are actually wired up. Do not flip a
    stage to true until it can serve real output.
    """
    return {
        "status": "ok",
        "version": __version__,
        "data_root": os.getenv("DATA_ROOT", "./data"),
        "pipeline_stages": {
            "preprocessing": False,
            "detection": False,
            "tracking": False,
            "precursors": False,
            "transition": False,
            "downscaling": False,
            "validation": False,
        },
    }
