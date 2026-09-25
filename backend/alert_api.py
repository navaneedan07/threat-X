"""
FastAPI alerting service.

USAGE:
  uvicorn api:app --reload --port 8000
  Then visit http://localhost:8000/docs
"""

import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

app = FastAPI(
    title="Extreme Weather Anomaly Alert API",
    description="SIH26078 proof-of-concept -- spatio-temporal anomaly tracking + alerting",
    version="0.1.0",
)


class Alert(BaseModel):
    event: str
    timestamp: str
    lat: float
    lon: float
    severity: str
    peak_zscore: float
    radius_km: float


def severity_from_zscore(z: float) -> str:
    if z >= 4.0:
        return "severe"
    elif z >= 3.0:
        return "moderate"
    else:
        return "low"


def load_anomalies(event: str):
    path = os.path.join(OUT_DIR, f"anomalies_{event}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No anomaly data for event '{event}'. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "service": "Extreme Weather Anomaly Alert API",
        "endpoints": ["/alerts/{event}", "/alerts/{event}/latest", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/alerts/{event}", response_model=list[Alert])
def get_alerts(event: str, min_severity: Optional[str] = None):
    data = load_anomalies(event)
    alerts = []
    for frame in data["frames"]:
        for box in frame["boxes"]:
            severity = severity_from_zscore(box["peak_zscore"])
            if min_severity and severity != min_severity:
                continue
            alerts.append(Alert(
                event=event,
                timestamp=frame["timestamp"],
                lat=box["centroid_lat"],
                lon=box["centroid_lon"],
                severity=severity,
                peak_zscore=box["peak_zscore"],
                radius_km=5.0,
            ))
    return alerts


@app.get("/alerts/{event}/latest", response_model=Alert)
def get_latest_alert(event: str):
    alerts = get_alerts(event)
    if not alerts:
        raise HTTPException(status_code=404, detail="No active alerts")
    return max(alerts, key=lambda a: a.peak_zscore)


@app.get("/alerts/{event}/trajectory")
def get_trajectory(event: str):
    data = load_anomalies(event)
    return data["trajectory"]