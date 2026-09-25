"""Router: active threat objects collection and single threat detail.

Endpoints:
  GET /api/v1/threats                  — list all tracked threats
  GET /api/v1/threats/{threat_id}      — full detail for one threat
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.threat import ThreatListResponse, ThreatObject, ThreatSummaryItem
from backend.services import threat_service

router = APIRouter(prefix="/api/v1/threats", tags=["threats"])

_VALID_SEVERITY = {"moderate", "severe", "extreme"}


@router.get(
    "",
    response_model=ThreatListResponse,
    summary="List active threat objects",
    description=(
        "Enumerate all actively tracked extreme weather threat objects. "
        "Optionally filter by severity band and limit result count."
    ),
)
def list_threats(
    severity: Annotated[
        Optional[str],
        Query(description="Filter by severity band: moderate | severe | extreme"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=500, description="Maximum number of results (default 50)"),
    ] = 50,
) -> ThreatListResponse:
    if severity is not None and severity not in _VALID_SEVERITY:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity '{severity}'. Must be one of: {sorted(_VALID_SEVERITY)}",
        )

    raw_threats = threat_service.get_all_threats(severity=severity, limit=limit)

    items = [
        ThreatSummaryItem(
            threat_id=t["threat_id"],
            event_type=t["event_type"],
            severity=t["severity"],
            confidence=t["confidence"],
            location=t["location"],
            movement=t["movement"],
            transition=t["transition"],
            provenance={
                "detector_version": t.get("provenance", {}).get("detector", "unknown"),
                "tracker_version": t.get("provenance", {}).get("pipeline_run_id", "unknown"),
                "forecast_reference_time": t.get("provenance", {}).get(
                    "forecast_reference_time"
                ),
            },
        )
        for t in raw_threats
    ]

    return ThreatListResponse(
        count=len(items),
        timestamp=datetime.now(timezone.utc),
        threats=items,
    )


@router.get(
    "/{threat_id}",
    response_model=ThreatObject,
    summary="Get single threat detail",
    description=(
        "Fetch complete atomic state for a specific threat, including full precursor, "
        "transition, and validation flags."
    ),
    responses={
        404: {"description": "Threat not found"},
    },
)
def get_threat(threat_id: str) -> ThreatObject:
    threat = threat_service.get_threat_by_id(threat_id)
    if threat is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threat {threat_id} not found",
        )
    return ThreatObject(**threat)
