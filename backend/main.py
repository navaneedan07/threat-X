"""Threat-X API entrypoint.

Run locally:
    uvicorn backend.main:app --reload

Interactive docs:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)

All threat endpoints are prefixed with /api/v1.
The /health endpoint is at the root for standard container liveness probes.

Pipeline stages are reported as False until the corresponding upstream module
has been integrated and has produced validated output.
Do NOT flip a stage to True without real output to back it up.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import alerts, footprint, precursors, threats, trajectory, transition
from backend.schemas.threat import HealthResponse, PipelineStages
from src import __version__

# ---------------------------------------------------------------------------
# Pipeline stage wiring flags.
# Flip to True only when the upstream module has been integrated and produces
# real validated output. This is the authoritative source for /health.
# ---------------------------------------------------------------------------
_PIPELINE_STAGES = PipelineStages(
    data_ingestion=False,           # Aravinth
    anomaly_detection=False,        # Pushpa
    threat_tracking=False,          # Sachin
    precursor_analysis=True,        # Hariharan — src/precursors/ implemented
    # Navaneedan — src/models/transition/ runs, fits and scores real events, but
    # two things are still missing: transition_service._load_from_pipeline() does
    # not read the report yet, and the target is a severity proxy. Before flipping
    # this, check TransitionReport.can_publish() — it withholds any horizon that
    # does not beat the base-rate forecast, and it does not replace the proxy
    # caveat in docs/transition.md §2.
    transition_intelligence=False,
    # Aravinth / Navaneedan — src/downscaling/baseline.py is implemented and tested,
    # but no coarse/fine pair of real fields exists yet, so there is no downscaled
    # field to serve.
    downscaling=False,
    # Navaneedan — src/validation/evaluation.py is implemented and is *supposed* to
    # be undecided: every gate threshold stays null until real metric distributions
    # are measured, so no PASS/DEGRADE verdict can be issued yet. There is also no
    # backend service for it.
    validation_gate=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    # Future: load model weights, open DB connections, warm caches here.
    yield


app = FastAPI(
    title="Threat-X — Extreme Weather Threat Intelligence API",
    description=(
        "Machine-readable threat information for tracked extreme-weather anomalies. "
        "This is a research/decision-support prototype — NOT an official meteorological "
        "warning service. All outputs must be independently verified before operational use."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — open during development. Restrict origins in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(threats.router)
app.include_router(trajectory.router)
app.include_router(precursors.router)
app.include_router(transition.router)
app.include_router(footprint.router)
app.include_router(alerts.router)


# ---------------------------------------------------------------------------
# Health endpoint (root — not under /api/v1 for container probe compatibility)
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness verification and pipeline capability discovery",
    description=(
        "Reports which pipeline stages are wired and capable of serving real data. "
        "Enables degraded-mode operation when some stages are unavailable."
    ),
)
def health() -> HealthResponse:
    """Liveness probe.

    `pipeline_stages` is the authoritative source of truth for which upstream
    modules are producing real validated output. Degrade gracefully — all
    endpoints remain accessible even when stages report False, returning
    fixture-backed responses where necessary.
    """
    active = sum(
        1
        for v in _PIPELINE_STAGES.model_dump().values()
        if v
    )
    total = len(_PIPELINE_STAGES.model_dump())

    if active == total:
        status = "healthy"
    elif active == 0:
        status = "degraded"
    else:
        status = "degraded"  # partial wiring = degraded until all stages live

    return HealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc),
        version=__version__,
        pipeline_stages=_PIPELINE_STAGES,
    )
