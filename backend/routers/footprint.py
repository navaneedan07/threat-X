"""Router: threat spatial footprint (GeoJSON).

Endpoint:
  GET /api/v1/threats/{threat_id}/footprint
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.threat import FootprintResponse
from backend.services import footprint_service

router = APIRouter(prefix="/api/v1/threats", tags=["footprint"])


@router.get(
    "/{threat_id}/footprint",
    response_model=FootprintResponse,
    summary="Get threat spatial footprint (GeoJSON)",
    description=(
        "Supplies spatial polygon geometries representing both the coarse candidate "
        "region and the 5 km downscaled boundary for GIS rendering. "
        "Coordinates follow RFC 7946: [longitude, latitude]."
    ),
    responses={
        404: {"description": "Threat or footprint not found"},
    },
)
def get_footprint(
    threat_id: str,
    downscaled: Annotated[
        bool,
        Query(
            description="If true, return the 5 km downscaled boundary when available"
        ),
    ] = False,
) -> FootprintResponse:
    data = footprint_service.get_footprint(threat_id, downscaled=downscaled)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threat {threat_id} not found",
        )
    return FootprintResponse(**data)
