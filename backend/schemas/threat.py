"""Pydantic schemas for the Threat-X REST API.

This file is the single source of truth for the JSON contract between the
backend and the GIS dashboard (frontend / Sachin).  Any field additions or
removals here constitute a breaking change and must be communicated to all
team members before merging.

Field inventory is aligned with:
  - configs/tracking.yaml  (threat_object_fields)
  - API_design.md          (response schema definitions)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-schemas — reused across multiple top-level response models
# ---------------------------------------------------------------------------


class LocationSchema(BaseModel):
    """Geographic location descriptor for a threat centroid."""

    centroid: tuple[float, float] = Field(
        ..., description="[latitude, longitude] coordinates of the threat centre of mass"
    )
    footprint_radius_km: float = Field(
        ..., ge=0.0, description="Equivalent circular radius bounding the anomaly area (km)"
    )


class MovementSchema(BaseModel):
    """Translational motion of the threat object."""

    direction_deg: float = Field(
        ..., ge=0.0, le=360.0, description="Bearing in degrees clockwise from true North"
    )
    speed_kmh: float = Field(..., ge=0.0, description="Movement velocity in km/h")


class EvolutionSchema(BaseModel):
    """Intensity-evolution metrics for the threat."""

    intensity_anomaly_sigma: float = Field(
        ..., description="Climatological Z-score (standard deviations above local mean)"
    )
    growth_rate_km2_h: float = Field(
        ..., description="Rate of footprint expansion in km2/h"
    )
    persistence_hours: float = Field(
        ..., ge=0.0, description="Hours elapsed since threat first instantiation"
    )


class PrecursorsSchema(BaseModel):
    """Scalar atmospheric precursor features over the threat centroid.

    All fields are Optional because upstream src/precursors/ may not have run.
    Null semantics: a null value means the module has not produced output,
    it does NOT mean the physical value is zero.
    """

    moisture_flux_convergence_g_kg_s: Optional[float] = None
    pressure_tendency_3h_hpa: Optional[float] = None
    vorticity_850_s1: Optional[float] = None
    theta_e_gradient_k_100km: Optional[float] = None


class TransitionSchema(BaseModel):
    """Escalation target embedded at the threat level (summary view).

    All probability fields are strictly null until Navaneedan's transition
    model (src/transition/) has been validated and calibrated.
    """

    target_severity: Optional[str] = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    window_hours: Optional[float] = Field(None, ge=0.0)


class ValidationSchema(BaseModel):
    """Quality-gate verdict from src/validation/ (Navaneedan)."""

    gate_verdict: str = Field(..., description="PASS, DEGRADE, or SUPPRESS")
    extreme_preservation_ratio: Optional[float] = None


class ProvenanceSchema(BaseModel):
    """Operational metadata tracing origin models and forecast runtime."""

    pipeline_run_id: str
    detector: str
    precursor_engine: str
    forecast_reference_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Top-level threat object (used by GET /api/v1/threats/{threat_id})
# ---------------------------------------------------------------------------


class ThreatObject(BaseModel):
    """Complete atomic state of a single tracked threat object.

    Regex: threat_id must match THR-YYYY-NNNN (e.g. THR-2026-0001).
    """

    threat_id: str = Field(
        ...,
        pattern=r"^THR-\d{4}-\d{4}$",
        description="Globally unique persistent threat identifier",
    )
    event_type: str = Field(
        ..., description="Categorical event type: extreme_rainfall | heatwave | gale_wind"
    )
    timestamp: datetime = Field(..., description="Observation timestamp in UTC")
    severity: str = Field(..., description="Current severity tier: moderate | severe | extreme")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated model confidence [0,1]")
    location: LocationSchema
    movement: MovementSchema
    evolution: EvolutionSchema
    precursors: PrecursorsSchema
    transition: TransitionSchema
    validation: ValidationSchema
    provenance: ProvenanceSchema


# ---------------------------------------------------------------------------
# Collection response (GET /api/v1/threats)
# ---------------------------------------------------------------------------


class ThreatSummaryItem(BaseModel):
    """Lightweight threat entry for the collection response (list view)."""

    threat_id: str
    event_type: str
    severity: str
    confidence: float
    location: LocationSchema
    movement: MovementSchema
    transition: TransitionSchema
    provenance: dict[str, Any]


class ThreatListResponse(BaseModel):
    count: int
    timestamp: datetime
    threats: list[ThreatSummaryItem]


# ---------------------------------------------------------------------------
# Trajectory response (GET /api/v1/threats/{threat_id}/trajectory)
# ---------------------------------------------------------------------------


class HistoricalStep(BaseModel):
    timestamp: datetime
    centroid: tuple[float, float] = Field(..., description="[lat, lon]")
    severity: str


class ForecastStep(BaseModel):
    lead_time_hours: int = Field(..., ge=0)
    timestamp: datetime
    centroid: tuple[float, float] = Field(..., description="[lat, lon]")
    uncertainty_radius_km: float = Field(..., ge=0.0)


class TrajectoryResponse(BaseModel):
    threat_id: str
    historical_steps: list[HistoricalStep]
    forecast_steps: list[ForecastStep]


# ---------------------------------------------------------------------------
# Precursor time-series response (GET /api/v1/threats/{threat_id}/precursors)
# ---------------------------------------------------------------------------


class PrecursorSeriesPoint(BaseModel):
    """Single time step in the precursor diagnostic series."""

    timestamp: datetime
    pressure_tendency_3h_hpa: Optional[float] = None
    moisture_flux_convergence_g_kg_s: Optional[float] = None
    vorticity_850_s1: Optional[float] = None
    theta_e_gradient_k_100km: Optional[float] = None


class PrecursorSeriesResponse(BaseModel):
    threat_id: str
    variables_analyzed: list[str]
    series: list[PrecursorSeriesPoint]
    explanatory_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Transition response (GET /api/v1/threats/{threat_id}/transition)
# ---------------------------------------------------------------------------


class TransitionWindowHours(BaseModel):
    earliest: Optional[int] = None
    most_likely: Optional[int] = None
    latest: Optional[int] = None


class TransitionEvaluation(BaseModel):
    target_state: Optional[str] = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    calibrated: bool = False
    brier_score_baseline: Optional[float] = None
    expected_window_hours: Optional[TransitionWindowHours] = None


class TransitionProvenanceSchema(BaseModel):
    model_id: Optional[str] = None
    training_run: Optional[str] = None


class TransitionResponse(BaseModel):
    threat_id: str
    current_state: str
    transition_evaluation: TransitionEvaluation
    lifecycle_phase: str
    provenance: TransitionProvenanceSchema


# ---------------------------------------------------------------------------
# Footprint / GeoJSON response (GET /api/v1/threats/{threat_id}/footprint)
# ---------------------------------------------------------------------------


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: list[Any]


class GeoJSONProperties(BaseModel):
    threat_id: str
    resolution_km: float
    peak_intensity_value: Optional[float] = None
    intensity_unit: Optional[str] = None
    extreme_preserved: Optional[bool] = None


class FootprintResponse(BaseModel):
    """RFC 7946 GeoJSON Feature. Coordinates: [longitude, latitude]."""

    type: str = "Feature"
    id: str
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties


# ---------------------------------------------------------------------------
# Alerts response (GET /api/v1/alerts)
# ---------------------------------------------------------------------------


class AlertProvenance(BaseModel):
    pipeline_version: Optional[str] = None
    validation_gate: Optional[str] = None


class AlertItem(BaseModel):
    alert_id: str
    threat_id: str
    headline: str
    severity: str
    urgency: str = Field(..., description="immediate | expected | future")
    target_area_description: str
    centroid: tuple[float, float] = Field(..., description="[lat, lon]")
    primary_driver: str
    transition_risk: Optional[str] = None
    validation_status: str
    provenance: AlertProvenance


class AlertsResponse(BaseModel):
    generated_at: datetime
    active_alerts_count: int
    alerts: list[AlertItem]


# ---------------------------------------------------------------------------
# Health response (GET /health)
# ---------------------------------------------------------------------------


class PipelineStages(BaseModel):
    data_ingestion: bool = False
    anomaly_detection: bool = False
    threat_tracking: bool = False
    precursor_analysis: bool = False
    transition_intelligence: bool = False
    downscaling: bool = False
    validation_gate: bool = False


class HealthResponse(BaseModel):
    status: str = Field(..., description="healthy | degraded | unhealthy")
    timestamp: datetime
    version: str
    pipeline_stages: PipelineStages
