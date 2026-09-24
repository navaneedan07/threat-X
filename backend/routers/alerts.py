"""Router: actionable operational alerts.

Endpoint:
  GET /api/v1/alerts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.threat import AlertsResponse
from backend.services import alert_service

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

_VALID_SEVERITY = {"moderate", "severe", "extreme"}


@router.get(
    "",
    response_model=AlertsResponse,
    summary="Get actionable operational alerts",
    description=(
        "High-level summary of active operational warnings designed for disaster "
        "authorities, emergency response teams, and notification feeds."
    ),
)
def get_alerts(
    min_severity: Annotated[
        Optional[str],
        Query(description="Minimum severity floor: moderate | severe | extreme"),
    ] = "moderate",
) -> AlertsResponse:
    if min_severity is not None and min_severity not in _VALID_SEVERITY:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid min_severity '{min_severity}'. Must be one of: {sorted(_VALID_SEVERITY)}",
        )

    alerts = alert_service.get_alerts(min_severity=min_severity)
    return AlertsResponse(
        generated_at=datetime.now(timezone.utc),
        active_alerts_count=len(alerts),
        alerts=alerts,
    )
