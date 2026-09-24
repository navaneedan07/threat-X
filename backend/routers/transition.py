"""Router: threat transition risk and escalation horizon.

Endpoint:
  GET /api/v1/threats/{threat_id}/transition

Degraded-mode contract:
  If Navaneedan's transition model has not run or is uncalibrated, this
  endpoint returns HTTP 200 with probability = null and calibrated = false.
  It does NOT return an error status.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.threat import TransitionResponse
from backend.services import transition_service

router = APIRouter(prefix="/api/v1/threats", tags=["transition"])


@router.get(
    "/{threat_id}/transition",
    response_model=TransitionResponse,
    summary="Get threat transition risk",
    description=(
        "Returns the calibrated probability and expected temporal window for a "
        "threat shifting into higher severity states. Fields are null if the "
        "transition model has not yet been calibrated."
    ),
    responses={
        404: {"description": "Threat not found"},
    },
)
def get_transition(threat_id: str) -> TransitionResponse:
    data = transition_service.get_transition(threat_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threat {threat_id} not found",
        )
    return TransitionResponse(**data)
