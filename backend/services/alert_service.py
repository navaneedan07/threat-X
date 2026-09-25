"""Alert service adapter.

Responsibilities:
  - Load operational alert objects from the integrated alert engine.
  - Fall back to data/samples/alerts.json fixtures.
  - Apply min_severity filtering.

The alert engine is the final aggregation of all validated pipeline stages.
When the full pipeline is wired, this service should produce alerts derived
from real threat, precursor, transition, and validation outputs rather than
from the fixture file.

Integration point: Replace _load_from_pipeline() once the alert engine
produces persistent output in data/processed/alerts.json.
"""

from __future__ import annotations

from typing import Optional

from . import fixtures

_SEVERITY_ORDER = {"moderate": 0, "severe": 1, "extreme": 2}


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> list[dict] | None:
    """Attempt to load real alert engine output.

    # TODO: INTEGRATE (Alert engine — final pipeline stage)
    # When the integrated alert engine writes alerts, load them here.
    # Return None to fall back to fixtures.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "alerts.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["alerts"]
    """
    return None  # alert engine not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_alerts(min_severity: Optional[str] = "moderate") -> list[dict]:
    """Return active operational alerts filtered by minimum severity floor.

    Parameters
    ----------
    min_severity : str
        Lowest severity to include (moderate | severe | extreme).
        Defaults to "moderate" (include all).
    """
    alerts = _load_from_pipeline() or fixtures.load_alerts()

    floor = _SEVERITY_ORDER.get(min_severity or "moderate", 0)
    return [
        a for a in alerts if _SEVERITY_ORDER.get(a.get("severity", "moderate"), 0) >= floor
    ]
