"""Router: threat trajectory history and forecast path.

Endpoint:
  GET /api/v1/threats/{threat_id}/trajectory
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.threat import TrajectoryResponse
from backend.services import trajectory_service

router = APIRouter(prefix="/api/v1/threats", tags=["trajectory"])


@router.get(
    "/{threat_id}/trajectory",
    response_model=TrajectoryResponse,
    summary="Get trajectory history and forecast path",
    description=(
        "Provide historical centroids (T0...Tn) and forward-projected path "
        "coordinates for GIS mapping."
    ),
    responses={
        404: {"description": "Threat or trajectory not found"},
    },
)
def get_trajectory(threat_id: str) -> TrajectoryResponse:
    trajectory = trajectory_service.get_trajectory(threat_id)
    if trajectory is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threat {threat_id} not found",
        )
    return TrajectoryResponse(**trajectory)
