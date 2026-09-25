"""Shared Threat Object contract — the schema every layer serializes.

Owner: Navaneedan (with Sachin at M3). Consumers: tracking, GNN, TTIE,
validation, backend, dashboard.

Why this module exists
----------------------
Four components serialize the Threat Object, so inventing a second definition
anywhere is a breaking change. This module is the **Python-side** contract used
by the pipeline (`src/`). `backend/schemas/threat.py` is the **HTTP-side**
mirror; `tests/test_contracts.py` asserts the two carry the same field names, so
neither can drift silently.

Dependency note: this module imports **stdlib only** so every stage can use it
without pulling in xarray/torch/FastAPI. That is deliberate — the interpolation
baseline and the lifecycle state machine must be testable on a bare Python.

Null semantics (repo-wide rule)
-------------------------------
A field is ``None`` when the producing module has **not run**, never as a
stand-in for a guessed value. ``null`` means "not computed yet". A consumer
renders "—". Never substitute 0, 0.0 or a placeholder number.

Integrity rules encoded here
----------------------------
1. ``TransitionEstimate.probability`` may only be set alongside a named
   ``model_id``. A probability with no model behind it is rejected, which is the
   contract-level version of "no hard-coded numbers".
2. ``ValidationResult.gate_verdict`` must be one of PASS / DEGRADE / SUPPRESS,
   the vocabulary in ``configs/validation.yaml``.
3. Timestamps must be timezone-aware and are normalised to UTC on parse.

What is intentionally NOT here
------------------------------
Gate thresholds, association thresholds, severity band cut-offs and
``coarse_resolution_deg`` / ``fine_resolution_deg`` stay ``null`` in
``configs/``. They are tunable parameters and must not be baked into a schema.
This file holds **structure** (field names, types, units, allowed labels) — not
thresholds. The lifecycle states and severity tiers below are declared in
``configs/tracking.yaml``; :func:`contract_drift` reports if they ever diverge.

Units (never ship a bare number whose unit is only in a docstring)
------------------------------------------------------------------
distances km · speeds km/h · bearing degrees clockwise from true north ·
pressure tendency hPa/3h · wind divergence s^-1 · precipitation mm/3h ·
temperature anomaly K · times ISO 8601 UTC · centroids ``[latitude, longitude]`` ·
GeoJSON footprint coordinates ``[longitude, latitude]`` (RFC 7946).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "ContractError",
    "EventType",
    "Footprint",
    "ForecastPoint",
    "GateVerdict",
    "LifecycleEvent",
    "LifecycleEventRecord",
    "LifecyclePhase",
    "Location",
    "Movement",
    "Evolution",
    "PrecursorSample",
    "PrecursorSeries",
    "Provenance",
    "Severity",
    "ThreatHistory",
    "ThreatObject",
    "Trajectory",
    "TrajectoryPoint",
    "TransitionEstimate",
    "TransitionWindow",
    "Urgency",
    "ValidationResult",
    "format_timestamp",
    "parse_timestamp",
    "load_threat_objects",
    "load_threat_history",
    "write_threat_objects",
    "contract_drift",
]


# ---------------------------------------------------------------------------
# Constants mirroring configs/
# ---------------------------------------------------------------------------

THREAT_ID_PATTERN = re.compile(r"^THR-\d{4}-\d{4}$")
"""Persistent threat identifier, e.g. ``THR-2026-0001`` (matches the API schema)."""

THREAT_OBJECT_FIELDS = (
    "threat_id",
    "event_type",
    "timestamp",
    "severity",
    "confidence",
    "location",
    "movement",
    "evolution",
    "precursors",
    "transition",
    "validation",
    "provenance",
    "ensemble_agreement",
    "footprint",
    "lifecycle_phase",
)
"""Canonical top-level Threat Object fields, in serialisation order.

Kept identical to ``configs/tracking.yaml -> threat_object_fields``;
:func:`contract_drift` enforces that in both directions. These are the *nested*
names actually serialised (``event_type``, not ``type``; ``location``, not
``centroid``) because the API schema and every fixture already use these.
"""

LIFECYCLE_STATES = ("FORMATION", "INTENSIFICATION", "EXPANSION", "PEAK", "DECAY")
"""``configs/tracking.yaml -> lifecycle.states``. These are the canonical states."""

LIFECYCLE_EVENTS = ("MERGER", "SPLIT", "WEAKENING", "RELOCATION")
"""``configs/tracking.yaml -> lifecycle.events``."""

SEVERITY_TIERS = ("moderate", "severe", "extreme")
"""``configs/tracking.yaml -> severity_bands`` declares moderate/severe; the API
schema and fixtures also use ``extreme`` as a target tier, so it is accepted.
The *band values* stay null in config — only the labels are fixed here.
"""

_PRECURSOR_NUMERIC_FIELDS = (
    "pressure_tendency_3h_hpa",
    "wind_divergence_s1",
    "wind_speed_ms",
    "precipitation_rate_mm3h",
    "t2m_anomaly_k",
    "moisture_flux_convergence_g_kg_s",
    "vorticity_850_s1",
    "theta_e_gradient_k_100km",
)
"""Numeric predecessor fields, in the order the pipeline emits them."""

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class ContractError(ValueError):
    """A Threat Object violated the shared contract.

    Raised at the producer (pipeline) rather than the consumer (API), so a
    malformed object is caught where it is created and not served as valid JSON.
    """


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Categorical threat type."""

    EXTREME_RAINFALL = "extreme_rainfall"
    HEATWAVE = "heatwave"
    GALE_WIND = "gale_wind"
    CYCLONE = "cyclone"
    """Emitted by ``src/precursors/pipeline.py`` for Cyclone Amphan
    (``THR-2020-0001``). The API schema listed only the first three; the real
    producer needs this fourth label, so it is part of the contract."""


class Severity(str, Enum):
    """Severity tier. Graduations between tiers come from ``severity_bands``."""

    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class LifecyclePhase(str, Enum):
    """Deterministic lifecycle state (``src/transition/lifecycle.py``)."""

    FORMATION = "FORMATION"
    INTENSIFICATION = "INTENSIFICATION"
    EXPANSION = "EXPANSION"
    PEAK = "PEAK"
    DECAY = "DECAY"


class LifecycleEvent(str, Enum):
    """Discrete lifecycle events layered on top of the ordered phases."""

    MERGER = "MERGER"
    SPLIT = "SPLIT"
    WEAKENING = "WEAKENING"
    RELOCATION = "RELOCATION"


class GateVerdict(str, Enum):
    """Validation gate outcome (``configs/validation.yaml``)."""

    PASS = "PASS"
    DEGRADE = "DEGRADE"
    SUPPRESS = "SUPPRESS"


class Urgency(str, Enum):
    """Alert urgency, per the API alert schema."""

    IMMEDIATE = "immediate"
    EXPECTED = "expected"
    FUTURE = "future"


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_enum(enum_cls: type[Enum], value: Any, path: str) -> Any:
    """Coerce ``value`` to ``enum_cls``, accepting a member, value or name.

    Matching is case-insensitive so the lowercase ``lifecycle_phase`` in
    ``data/samples/transitions.json`` parses to ``LifecyclePhase.INTENSIFICATION``.
    """
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected string, got {type(value).__name__}")
    key = value.strip().casefold()
    for member in enum_cls:
        if key == member.value.casefold() or key == member.name.casefold():
            return member
    allowed = ", ".join(member.value for member in enum_cls)
    raise ContractError(f"{path}: {value!r} is not one of: {allowed}")


def _check_id(value: Any, path: str = "threat_id") -> str:
    """Validate a persistent threat identifier."""
    if not isinstance(value, str) or not THREAT_ID_PATTERN.match(value):
        raise ContractError(
            f"{path}: expected a threat id matching THR-YYYY-NNNN, got {value!r}"
        )
    return value


def _check_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Validate an optional finite number, optionally bounded."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path}: expected a number or null, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{path}: expected a finite number, got {value!r}")
    if minimum is not None and number < minimum:
        raise ContractError(f"{path}: {number} is below the minimum {minimum}")
    if maximum is not None and number > maximum:
        raise ContractError(f"{path}: {number} is above the maximum {maximum}")
    return number


def _check_probability(value: Any, path: str) -> float | None:
    """Validate an optional probability in [0, 1]."""
    return _check_number(value, path, minimum=0.0, maximum=1.0)


def _check_centroid(lat: Any, lon: Any, path: str) -> tuple[float, float]:
    """Validate a ``[latitude, longitude]`` pair."""
    return (
        _check_number(lat, f"{path}.latitude", minimum=-90.0, maximum=90.0),
        _check_number(lon, f"{path}.longitude", minimum=-180.0, maximum=180.0),
    )


def parse_timestamp(value: Any, path: str = "timestamp") -> datetime:
    """Parse an ISO 8601 timestamp and normalise it to UTC.

    A naive (timezone-less) timestamp is rejected: the repo standard is ISO 8601
    UTC everywhere, and a silently-assumed local timezone is a real bug source.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ContractError(f"{path}: {value!r} is not ISO 8601 ({exc})") from exc
    else:
        raise ContractError(
            f"{path}: expected an ISO 8601 string or datetime, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        raise ContractError(
            f"{path}: naive timestamp {value!r}; timestamps must be timezone-aware UTC"
        )
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Format a datetime as ISO 8601 UTC with a ``Z`` suffix."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_str(value: Any, path: str) -> str | None:
    """Validate an optional string."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected a string or null, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Threat Object sub-blocks
# ---------------------------------------------------------------------------


@dataclass
class Location:
    """Geographic location of a threat centroid."""

    centroid_lat: float
    centroid_lon: float
    footprint_radius_km: float

    def __post_init__(self) -> None:
        self.centroid_lat, self.centroid_lon = _check_centroid(
            self.centroid_lat, self.centroid_lon, "location.centroid"
        )
        self.footprint_radius_km = _check_number(
            self.footprint_radius_km, "location.footprint_radius_km", minimum=0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid": [self.centroid_lat, self.centroid_lon],
            "footprint_radius_km": self.footprint_radius_km,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Location:
        centroid = data.get("centroid") or [None, None]
        if len(centroid) != 2:
            raise ContractError("location.centroid: expected [latitude, longitude]")
        return cls(
            centroid_lat=centroid[0],
            centroid_lon=centroid[1],
            footprint_radius_km=data.get("footprint_radius_km"),
        )


@dataclass
class Movement:
    """Translational motion. Bearing is degrees clockwise from true north."""

    direction_deg: float | None = None
    speed_kmh: float | None = None

    def __post_init__(self) -> None:
        self.direction_deg = _check_number(
            self.direction_deg, "movement.direction_deg", minimum=0.0, maximum=360.0
        )
        self.speed_kmh = _check_number(
            self.speed_kmh, "movement.speed_kmh", minimum=0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {"direction_deg": self.direction_deg, "speed_kmh": self.speed_kmh}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Movement:
        return cls(
            direction_deg=data.get("direction_deg"), speed_kmh=data.get("speed_kmh")
        )


@dataclass
class Evolution:
    """Intensity-evolution metrics.

    ``growth_rate_km2_h`` is signed on purpose: a shrinking footprint is a real
    observation, not a contract violation.
    """

    intensity_anomaly_sigma: float | None = None
    growth_rate_km2_h: float | None = None
    persistence_hours: float | None = None

    def __post_init__(self) -> None:
        self.intensity_anomaly_sigma = _check_number(
            self.intensity_anomaly_sigma, "evolution.intensity_anomaly_sigma"
        )
        self.growth_rate_km2_h = _check_number(
            self.growth_rate_km2_h, "evolution.growth_rate_km2_h"
        )
        self.persistence_hours = _check_number(
            self.persistence_hours, "evolution.persistence_hours", minimum=0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intensity_anomaly_sigma": self.intensity_anomaly_sigma,
            "growth_rate_km2_h": self.growth_rate_km2_h,
            "persistence_hours": self.persistence_hours,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evolution:
        return cls(
            intensity_anomaly_sigma=data.get("intensity_anomaly_sigma"),
            growth_rate_km2_h=data.get("growth_rate_km2_h"),
            persistence_hours=data.get("persistence_hours"),
        )


@dataclass
class PrecursorSample:
    """Atmospheric precursor scalars for one time step, over the threat centroid.

    The first five numeric fields are computed today by
    ``src/precursors/engine.py`` from ERA5 surface fields. The last three need
    pressure-level data the confirmed dataset does not contain, so they are
    ``None`` — accurately unavailable, not a computation failure.

    ``timestamp`` is set on samples inside a :class:`PrecursorSeries` and stays
    ``None`` when the sample is embedded on a Threat Object, where the threat's
    own ``timestamp`` already carries the time.
    """

    timestamp: datetime | None = None
    pressure_tendency_3h_hpa: float | None = None
    wind_divergence_s1: float | None = None
    wind_speed_ms: float | None = None
    precipitation_rate_mm3h: float | None = None
    t2m_anomaly_k: float | None = None
    moisture_flux_convergence_g_kg_s: float | None = None
    vorticity_850_s1: float | None = None
    theta_e_gradient_k_100km: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp is not None:
            self.timestamp = parse_timestamp(self.timestamp, "precursors.timestamp")
        for name in _PRECURSOR_NUMERIC_FIELDS:
            setattr(self, name, _check_number(getattr(self, name), f"precursors.{name}"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": format_timestamp(self.timestamp) if self.timestamp else None
        }
        for name in _PRECURSOR_NUMERIC_FIELDS:
            payload[name] = getattr(self, name)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrecursorSample:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PrecursorSeries:
    """Full precursor time series for one threat.

    Shape matches the real output of ``python -m src.precursors.pipeline``
    (``data/processed/precursors.json``) and the API precursor response.
    """

    threat_id: str
    series: list[PrecursorSample] = field(default_factory=list)
    variables_analyzed: list[str] = field(default_factory=list)
    unavailable_variables: list[str] = field(default_factory=list)
    explanatory_summary: str | None = None

    def __post_init__(self) -> None:
        self.threat_id = _check_id(self.threat_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "variables_analyzed": list(self.variables_analyzed),
            "unavailable_variables": list(self.unavailable_variables),
            "series": [sample.to_dict() for sample in self.series],
            "explanatory_summary": self.explanatory_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrecursorSeries:
        raw_series = data.get("series") or []
        parsed = [
            PrecursorSample.from_dict(dict(entry)) for entry in raw_series
        ]
        return cls(
            threat_id=data.get("threat_id"),
            series=parsed,
            variables_analyzed=data.get("variables_analyzed") or [],
            unavailable_variables=data.get("unavailable_variables") or [],
            explanatory_summary=data.get("explanatory_summary"),
        )


@dataclass
class TransitionWindow:
    """Expected transition window, in hours from the current time step."""

    earliest: int | None = None
    most_likely: int | None = None
    latest: int | None = None

    def __post_init__(self) -> None:
        for name in ("earliest", "most_likely", "latest"):
            value = getattr(self, name)
            if value is None:
                continue
            checked = _check_number(value, f"transition.window.{name}", minimum=0.0)
            setattr(self, name, int(checked))
        bounds = [self.earliest, self.most_likely, self.latest]
        present = [b for b in bounds if b is not None]
        if present != sorted(present):
            raise ContractError(
                "transition.window: earliest <= most_likely <= latest is required"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "earliest": self.earliest,
            "most_likely": self.most_likely,
            "latest": self.latest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionWindow:
        return cls(
            earliest=data.get("earliest"),
            most_likely=data.get("most_likely"),
            latest=data.get("latest"),
        )


@dataclass
class TransitionEstimate:
    """Threat Transition Intelligence output (``src/transition/``).

    Integrity rule enforced in ``__post_init__``: a non-null ``probability``
    requires a non-null ``model_id``. This is how "never publish a hard-coded
    number" is made structural rather than a promise in a README.

    ``calibrated`` is separate from ``probability`` on purpose — an uncalibrated
    logistic-regression probability is a legitimate output; claiming it is
    calibrated when it has not been checked is not.
    """

    target_severity: Severity | None = None
    probability: float | None = None
    horizon_hours: int | None = None
    window: TransitionWindow | None = None
    drivers: list[str] = field(default_factory=list)
    confidence: float | None = None
    calibrated: bool = False
    brier_score_baseline: float | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        self.target_severity = _coerce_enum(
            Severity, self.target_severity, "transition.target_severity"
        )
        self.probability = _check_probability(
            self.probability, "transition.probability"
        )
        if self.horizon_hours is not None:
            self.horizon_hours = int(
                _check_number(self.horizon_hours, "transition.horizon_hours", minimum=0.0)
            )
        self.confidence = _check_probability(self.confidence, "transition.confidence")
        self.brier_score_baseline = _check_number(
            self.brier_score_baseline, "transition.brier_score_baseline", minimum=0.0
        )
        self.model_id = _optional_str(self.model_id, "transition.model_id")
        if self.probability is not None and not self.model_id:
            raise ContractError(
                "transition.probability is set but transition.model_id is null: "
                "a probability may only be published by a named, tested model"
            )

    @property
    def is_publishable(self) -> bool:
        """True when a probability exists and names the model that produced it."""
        return self.probability is not None and bool(self.model_id)

    @property
    def window_hours(self) -> float | None:
        """Most-likely transition lead time in hours, or None."""
        return float(self.window.most_likely) if self.window and self.window.most_likely else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for both existing consumers.

        ``window_hours`` is the flat key used by the threat-level summary
        (``TransitionSchema``); ``expected_window_hours`` is the structured key
        used by the transition endpoint (``TransitionEvaluation``). Both are
        emitted so the two API shapes stay satisfied by one contract.
        """
        return {
            "target_severity": self.target_severity.value
            if self.target_severity
            else None,
            "probability": self.probability,
            "horizon_hours": self.horizon_hours,
            "window_hours": self.window_hours,
            "expected_window_hours": self.window.to_dict() if self.window else None,
            "drivers": list(self.drivers),
            "confidence": self.confidence,
            "calibrated": self.calibrated,
            "brier_score_baseline": self.brier_score_baseline,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionEstimate:
        window = data.get("expected_window_hours") or data.get("window")
        if window is None and data.get("window_hours") is not None:
            # Tolerate the flat form on read so older records stay loadable.
            window = {"most_likely": data["window_hours"]}
        return cls(
            # ``target_state`` is the key used by the transition endpoint and by
            # data/samples/transitions.json; ``target_severity`` is the key used
            # by the threat-level summary. Both are accepted on read.
            target_severity=data.get("target_severity", data.get("target_state")),
            probability=data.get("probability"),
            horizon_hours=data.get("horizon_hours"),
            window=TransitionWindow.from_dict(window) if window else None,
            drivers=list(data.get("drivers") or []),
            confidence=data.get("confidence"),
            calibrated=bool(data.get("calibrated", False)),
            brier_score_baseline=data.get("brier_score_baseline"),
            model_id=data.get("model_id"),
        )


@dataclass
class ValidationResult:
    """Quality-gate verdict from ``src/validation/``.

    ``gate_verdict`` is deliberately **not** defaulted: a verdict must be a
    decision from the metric table, so it has to be stated explicitly. Thresholds
    live in ``configs/validation.yaml`` and are set only once measured.
    """

    gate_verdict: GateVerdict
    extreme_preservation_ratio: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        self.gate_verdict = _coerce_enum(
            GateVerdict, self.gate_verdict, "validation.gate_verdict"
        )
        if self.gate_verdict is None:
            raise ContractError("validation.gate_verdict: a verdict is required")
        self.extreme_preservation_ratio = _check_number(
            self.extreme_preservation_ratio,
            "validation.extreme_preservation_ratio",
            minimum=0.0,
        )
        if self.metrics:
            self.metrics = {
                name: _check_number(value, f"validation.metrics.{name}")
                for name, value in self.metrics.items()
            }
        self.reason = _optional_str(self.reason, "validation.reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_verdict": self.gate_verdict.value,
            "extreme_preservation_ratio": self.extreme_preservation_ratio,
            "metrics": dict(self.metrics),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        return cls(
            gate_verdict=data.get("gate_verdict"),
            extreme_preservation_ratio=data.get("extreme_preservation_ratio"),
            metrics=dict(data.get("metrics") or {}),
            reason=data.get("reason"),
        )


@dataclass
class Provenance:
    """Reproducibility metadata. A number without its run and config is not reproducible."""

    pipeline_run_id: str | None = None
    detector: str | None = None
    precursor_engine: str | None = None
    forecast_reference_time: datetime | None = None

    def __post_init__(self) -> None:
        self.pipeline_run_id = _optional_str(self.pipeline_run_id, "provenance.pipeline_run_id")
        self.detector = _optional_str(self.detector, "provenance.detector")
        self.precursor_engine = _optional_str(
            self.precursor_engine, "provenance.precursor_engine"
        )
        if self.forecast_reference_time is not None:
            self.forecast_reference_time = parse_timestamp(
                self.forecast_reference_time, "provenance.forecast_reference_time"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "detector": self.detector,
            "precursor_engine": self.precursor_engine,
            "forecast_reference_time": format_timestamp(self.forecast_reference_time)
            if self.forecast_reference_time
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            pipeline_run_id=data.get("pipeline_run_id"),
            detector=data.get("detector"),
            precursor_engine=data.get("precursor_engine"),
            forecast_reference_time=data.get("forecast_reference_time"),
        )


@dataclass
class Footprint:
    """Threat footprint as an RFC 7946 GeoJSON Feature.

    Coordinates are ``[longitude, latitude]`` — the opposite order to a centroid
    ``[latitude, longitude]``. Getting this backwards is the classic GIS bug, so
    :meth:`__post_init__` validates every ring vertex and requires closed rings.
    """

    resolution_km: float
    coordinates: list[list[list[float]]] | None = None
    geometry_type: str = "Polygon"
    threat_id: str | None = None
    peak_intensity_value: float | None = None
    intensity_unit: str | None = None
    extreme_preserved: bool | None = None

    def __post_init__(self) -> None:
        self.resolution_km = _check_number(
            self.resolution_km, "footprint.resolution_km", minimum=0.0
        )
        self.peak_intensity_value = _check_number(
            self.peak_intensity_value, "footprint.peak_intensity_value"
        )
        self.intensity_unit = _optional_str(self.intensity_unit, "footprint.intensity_unit")
        if self.threat_id is not None:
            self.threat_id = _check_id(self.threat_id)

        if self.coordinates is not None:
            for ring_index, ring in enumerate(self.coordinates):
                if len(ring) < 4:
                    raise ContractError(
                        f"footprint.coordinates[{ring_index}]: a ring needs >= 4 positions"
                    )
                for vertex_index, vertex in enumerate(ring):
                    if len(vertex) != 2:
                        raise ContractError(
                            f"footprint.coordinates[{ring_index}][{vertex_index}]: "
                            "expected [longitude, latitude]"
                        )
                    _check_number(
                        vertex[0],
                        f"footprint.coordinates[{ring_index}][{vertex_index}].longitude",
                        minimum=-180.0,
                        maximum=180.0,
                    )
                    _check_number(
                        vertex[1],
                        f"footprint.coordinates[{ring_index}][{vertex_index}].latitude",
                        minimum=-90.0,
                        maximum=90.0,
                    )
                if ring[0] != ring[-1]:
                    raise ContractError(
                        f"footprint.coordinates[{ring_index}]: ring is not closed "
                        "(first and last position must match)"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a GeoJSON Feature, matching ``data/samples/footprints.json``."""
        return {
            "type": "Feature",
            "id": self.threat_id,
            "geometry": {"type": self.geometry_type, "coordinates": self.coordinates},
            "properties": {
                "threat_id": self.threat_id,
                "resolution_km": self.resolution_km,
                "peak_intensity_value": self.peak_intensity_value,
                "intensity_unit": self.intensity_unit,
                "extreme_preserved": self.extreme_preserved,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Footprint:
        geometry = data.get("geometry") or {}
        properties = data.get("properties") or {}
        return cls(
            resolution_km=properties.get("resolution_km"),
            coordinates=geometry.get("coordinates"),
            geometry_type=geometry.get("type", "Polygon"),
            threat_id=properties.get("threat_id", data.get("id")),
            peak_intensity_value=properties.get("peak_intensity_value"),
            intensity_unit=properties.get("intensity_unit"),
            extreme_preserved=properties.get("extreme_preserved"),
        )


# ---------------------------------------------------------------------------
# The Threat Object
# ---------------------------------------------------------------------------


@dataclass
class ThreatObject:
    """Complete atomic state of one tracked threat at one time step.

    Only ``threat_id``, ``event_type``, ``timestamp`` and ``severity`` are
    required. Every other block defaults to ``None`` so a stage can publish what
    it has and leave the rest genuinely unset rather than filled with zeros.
    """

    threat_id: str
    event_type: EventType
    timestamp: datetime
    severity: Severity
    confidence: float | None = None
    location: Location | None = None
    movement: Movement | None = None
    evolution: Evolution | None = None
    precursors: PrecursorSample | None = None
    transition: TransitionEstimate | None = None
    validation: ValidationResult | None = None
    provenance: Provenance | None = None
    ensemble_agreement: float | None = None
    footprint: Footprint | None = None
    lifecycle_phase: LifecyclePhase | None = None
    """Assigned per time step by ``src/transition/lifecycle.py``. Not part of
    ``configs/tracking.yaml -> threat_object_fields``, but the API transition
    response and ``data/samples/transitions.json`` both carry it."""

    def __post_init__(self) -> None:
        self.threat_id = _check_id(self.threat_id)
        self.event_type = _coerce_enum(EventType, self.event_type, "event_type")
        if self.event_type is None:
            raise ContractError("event_type: required")
        self.timestamp = parse_timestamp(self.timestamp, "timestamp")
        self.severity = _coerce_enum(Severity, self.severity, "severity")
        if self.severity is None:
            raise ContractError("severity: required")
        self.confidence = _check_probability(self.confidence, "confidence")
        self.ensemble_agreement = _check_probability(
            self.ensemble_agreement, "ensemble_agreement"
        )
        self.lifecycle_phase = _coerce_enum(
            LifecyclePhase, self.lifecycle_phase, "lifecycle_phase"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the interchange shape served by ``backend/schemas/threat.py``."""
        return {
            "threat_id": self.threat_id,
            "event_type": self.event_type.value,
            "timestamp": format_timestamp(self.timestamp),
            "severity": self.severity.value,
            "confidence": self.confidence,
            "location": self.location.to_dict() if self.location else None,
            "movement": self.movement.to_dict() if self.movement else None,
            "evolution": self.evolution.to_dict() if self.evolution else None,
            "precursors": self.precursors.to_dict() if self.precursors else None,
            "transition": self.transition.to_dict() if self.transition else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "ensemble_agreement": self.ensemble_agreement,
            "footprint": self.footprint.to_dict() if self.footprint else None,
            "lifecycle_phase": self.lifecycle_phase.value if self.lifecycle_phase else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThreatObject:
        """Parse an interchange dict. Unknown keys are ignored for forward compatibility."""
        return cls(
            threat_id=data.get("threat_id"),
            event_type=data.get("event_type"),
            timestamp=data.get("timestamp"),
            severity=data.get("severity"),
            confidence=data.get("confidence"),
            location=Location.from_dict(data["location"]) if data.get("location") else None,
            movement=Movement.from_dict(data["movement"]) if data.get("movement") else None,
            evolution=Evolution.from_dict(data["evolution"]) if data.get("evolution") else None,
            precursors=PrecursorSample.from_dict(data["precursors"])
            if data.get("precursors")
            else None,
            transition=TransitionEstimate.from_dict(data["transition"])
            if data.get("transition")
            else None,
            validation=ValidationResult.from_dict(data["validation"])
            if data.get("validation")
            else None,
            provenance=Provenance.from_dict(data["provenance"])
            if data.get("provenance")
            else None,
            ensemble_agreement=data.get("ensemble_agreement"),
            footprint=Footprint.from_dict(data["footprint"]) if data.get("footprint") else None,
            lifecycle_phase=data.get("lifecycle_phase"),
        )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPoint:
    """One observed (historical) position on a threat track."""

    timestamp: datetime
    centroid_lat: float
    centroid_lon: float
    severity: Severity | None = None

    def __post_init__(self) -> None:
        self.timestamp = parse_timestamp(self.timestamp)
        self.centroid_lat, self.centroid_lon = _check_centroid(
            self.centroid_lat, self.centroid_lon, "trajectory.centroid"
        )
        self.severity = _coerce_enum(Severity, self.severity, "trajectory.severity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_timestamp(self.timestamp),
            "centroid": [self.centroid_lat, self.centroid_lon],
            "severity": self.severity.value if self.severity else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryPoint:
        centroid = data.get("centroid") or [None, None]
        if len(centroid) != 2:
            raise ContractError("trajectory.centroid: expected [latitude, longitude]")
        return cls(
            timestamp=data.get("timestamp"),
            centroid_lat=centroid[0],
            centroid_lon=centroid[1],
            severity=data.get("severity"),
        )


@dataclass
class ForecastPoint:
    """A projected position with an explicit uncertainty radius."""

    lead_time_hours: int
    timestamp: datetime
    centroid_lat: float
    centroid_lon: float
    uncertainty_radius_km: float

    def __post_init__(self) -> None:
        self.lead_time_hours = int(
            _check_number(self.lead_time_hours, "forecast.lead_time_hours", minimum=0.0)
        )
        self.timestamp = parse_timestamp(self.timestamp, "forecast.timestamp")
        self.centroid_lat, self.centroid_lon = _check_centroid(
            self.centroid_lat, self.centroid_lon, "forecast.centroid"
        )
        self.uncertainty_radius_km = _check_number(
            self.uncertainty_radius_km, "forecast.uncertainty_radius_km", minimum=0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead_time_hours": self.lead_time_hours,
            "timestamp": format_timestamp(self.timestamp),
            "centroid": [self.centroid_lat, self.centroid_lon],
            "uncertainty_radius_km": self.uncertainty_radius_km,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastPoint:
        centroid = data.get("centroid") or [None, None]
        if len(centroid) != 2:
            raise ContractError("forecast.centroid: expected [latitude, longitude]")
        return cls(
            lead_time_hours=data.get("lead_time_hours"),
            timestamp=data.get("timestamp"),
            centroid_lat=centroid[0],
            centroid_lon=centroid[1],
            uncertainty_radius_km=data.get("uncertainty_radius_km"),
        )


@dataclass
class Trajectory:
    """Observed and projected path for one threat (``src/tracking/``)."""

    threat_id: str
    historical_steps: list[TrajectoryPoint] = field(default_factory=list)
    forecast_steps: list[ForecastPoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.threat_id = _check_id(self.threat_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "historical_steps": [step.to_dict() for step in self.historical_steps],
            "forecast_steps": [step.to_dict() for step in self.forecast_steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            threat_id=data.get("threat_id"),
            historical_steps=[
                TrajectoryPoint.from_dict(step)
                for step in (data.get("historical_steps") or [])
            ],
            forecast_steps=[
                ForecastPoint.from_dict(step) for step in (data.get("forecast_steps") or [])
            ],
        )


# ---------------------------------------------------------------------------
# History (input to the lifecycle state machine)
# ---------------------------------------------------------------------------


@dataclass
class LifecycleEventRecord:
    """A discrete lifecycle event observed at a point in time."""

    timestamp: datetime
    event: LifecycleEvent

    def __post_init__(self) -> None:
        self.timestamp = parse_timestamp(self.timestamp)
        self.event = _coerce_enum(LifecycleEvent, self.event, "lifecycle_event.event")
        if self.event is None:
            raise ContractError("lifecycle_event.event: required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_timestamp(self.timestamp),
            "event": self.event.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleEventRecord:
        return cls(timestamp=data.get("timestamp"), event=data.get("event"))


@dataclass
class ThreatHistory:
    """Ordered snapshots of one threat, plus any discrete lifecycle events.

    This is the input to the deterministic lifecycle state machine and,
    later, to the learned transition model.
    """

    threat_id: str
    snapshots: list[ThreatObject] = field(default_factory=list)
    events: list[LifecycleEventRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.threat_id = _check_id(self.threat_id)
        for index, snapshot in enumerate(self.snapshots):
            if snapshot.threat_id != self.threat_id:
                raise ContractError(
                    f"snapshots[{index}].threat_id: {snapshot.threat_id!r} does not "
                    f"match history threat_id {self.threat_id!r}"
                )

    def ordered(self) -> list[ThreatObject]:
        """Snapshots sorted by timestamp, rejecting duplicates at one time step."""
        ordered = sorted(self.snapshots, key=lambda snapshot: snapshot.timestamp)
        seen: set[datetime] = set()
        for snapshot in ordered:
            if snapshot.timestamp in seen:
                raise ContractError(
                    f"threat {self.threat_id}: duplicate snapshot at "
                    f"{format_timestamp(snapshot.timestamp)}; one state per time step"
                )
            seen.add(snapshot.timestamp)
        return ordered

    @property
    def first(self) -> ThreatObject | None:
        ordered = self.ordered()
        return ordered[0] if ordered else None

    @property
    def latest(self) -> ThreatObject | None:
        ordered = self.ordered()
        return ordered[-1] if ordered else None

    @property
    def duration_hours(self) -> float | None:
        """Observed lifetime in hours, or None if the track has fewer than two steps."""
        ordered = self.ordered()
        if len(ordered) < 2:
            return None
        delta = ordered[-1].timestamp - ordered[0].timestamp
        return delta.total_seconds() / 3600.0

    def severities(self) -> list[Severity]:
        return [snapshot.severity for snapshot in self.ordered()]


# ---------------------------------------------------------------------------
# File I/O — the handoff format for data/processed/threats.json
# ---------------------------------------------------------------------------

_META_NOTE = (
    "Serialised from src/shared/contracts.py. Null means the producing module "
    "has not run — not zero, not an estimate."
)


def load_threat_objects(path: str | Path) -> list[ThreatObject]:
    """Load threat objects from ``{"threats": [...]}`` or a bare JSON list.

    This is the reader for ``data/processed/threats.json``, the integration point
    ``backend/services/threat_service.py`` already looks for.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("threats") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ContractError(f"{path}: expected a 'threats' list, got {type(entries).__name__}")
    return [ThreatObject.from_dict(entry) for entry in entries]


def load_threat_history(path: str | Path) -> list[ThreatHistory]:
    """Group threat objects by ``threat_id`` into per-threat histories."""
    grouped: dict[str, ThreatHistory] = {}
    for threat in load_threat_objects(path):
        history = grouped.setdefault(threat.threat_id, ThreatHistory(threat_id=threat.threat_id))
        history.snapshots.append(threat)
    return list(grouped.values())


def write_threat_objects(objects: list[ThreatObject], path: str | Path) -> Path:
    """Write threat objects with provenance metadata, creating parent directories.

    ``_meta`` records that this is pipeline output, so a reader can always tell
    pipeline output from a hand-written demo fixture.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generated_at": format_timestamp(datetime.now(timezone.utc)),
            "source": "src.shared.contracts.write_threat_objects",
            "note": _META_NOTE,
        },
        "threats": [threat.to_dict() for threat in objects],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def contract_drift(config_path: str | Path | None = None) -> list[str]:
    """Compare the declared contract against ``configs/tracking.yaml``.

    Returns a list of human-readable divergences (empty when in sync). Only the
    *structure* is compared — states, events and field names. Threshold values are
    intentionally ``null`` in config and are not part of this check.

    Field names are compared in **both** directions: a config name the contract
    does not carry is drift, and a contract name the config does not declare is
    drift too. One-directional checking would let a newly added field go
    undeclared, which is how the flat ``type``/``centroid`` vocabulary survived
    here in the first place.
    """
    path = Path(config_path) if config_path else _CONFIGS_DIR / "tracking.yaml"
    try:
        import yaml
    except ImportError:
        return [f"PyYAML unavailable — cannot read {path}"]

    if not path.exists():
        return [f"{path} not found"]

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    drift: list[str] = []

    lifecycle = config.get("lifecycle") or {}
    declared_states = tuple(lifecycle.get("states") or ())
    if declared_states and declared_states != LIFECYCLE_STATES:
        drift.append(
            f"lifecycle.states: config={declared_states} contract={LIFECYCLE_STATES}"
        )
    declared_events = tuple(lifecycle.get("events") or ())
    if declared_events and declared_events != LIFECYCLE_EVENTS:
        drift.append(
            f"lifecycle.events: config={declared_events} contract={LIFECYCLE_EVENTS}"
        )

    declared_tiers = tuple((config.get("severity_bands") or {}).keys())
    if declared_tiers and set(declared_tiers) - set(SEVERITY_TIERS):
        drift.append(
            f"severity_bands: config declares {declared_tiers}, "
            f"contract knows {SEVERITY_TIERS}"
        )

    declared_fields = tuple(config.get("threat_object_fields") or ())
    if declared_fields:
        unknown = set(declared_fields) - set(THREAT_OBJECT_FIELDS)
        if unknown:
            drift.append(
                f"threat_object_fields: config declares {sorted(unknown)}, "
                "which ThreatObject does not carry"
            )
        undeclared = set(THREAT_OBJECT_FIELDS) - set(declared_fields)
        if undeclared:
            drift.append(
                f"threat_object_fields: ThreatObject carries {sorted(undeclared)}, "
                "which configs/tracking.yaml does not declare"
            )
    return drift
