"""Router: atmospheric precursor time-series.

Endpoint:
  GET /api/v1/threats/{threat_id}/precursors
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.threat import PrecursorSeriesResponse
from backend.services import precursor_service

router = APIRouter(prefix="/api/v1/threats", tags=["precursors"])


@router.get(
    "/{threat_id}/precursors",
    response_model=PrecursorSeriesResponse,
    summary="Get atmospheric precursor time-series",
    description=(
        "Retrieve the temporal sequence of physical diagnostic features driving "
        "threat evolution, supporting explainability visuals."
    ),
    responses={
        404: {"description": "Threat or precursor data not found"},
    },
)
def get_precursors(
    threat_id: str,
    window_hours: Annotated[
        int,
        Query(ge=1, le=720, description="Historical precursor window to inspect (hours, default 24)"),
    ] = 24,
) -> PrecursorSeriesResponse:
    data = precursor_service.get_precursors(threat_id, window_hours=window_hours)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threat {threat_id} not found",
        )
    return PrecursorSeriesResponse(**data)
