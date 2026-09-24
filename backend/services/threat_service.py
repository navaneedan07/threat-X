"""Threat service adapter.

Responsibilities:
  - Load threat objects from the tracking module output (src/tracking/).
  - Fall back to data/samples/threats.json fixtures when real output is absent.
  - Apply severity filtering and pagination as required by the API contract.

Integration point for Sachin (src/tracking/):
  When the tracking module writes its output, replace _load_from_pipeline()
  with real file/DB reads. The rest of the service (filtering, response
  building) does not need to change.
"""

from __future__ import annotations

from typing import Optional

from . import fixtures


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> list[dict] | None:
    """Attempt to load real tracking output.

    # TODO: INTEGRATE (Sachin - src/tracking/)
    # When tracking produces a persistent state file (e.g. data/processed/threats.json),
    # load it here and return the list of threat dicts. Return None to fall back
    # to fixtures.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "threats.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["threats"]
    """
    return None  # pipeline not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_all_threats(
    severity: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Return threat objects, optionally filtered by severity and capped at limit.

    Falls back to fixtures if the pipeline has not produced output.
    """
    threats = _load_from_pipeline() or fixtures.load_threats()

    if severity is not None:
        threats = [t for t in threats if t.get("severity") == severity]

    return threats[:limit]


def get_threat_by_id(threat_id: str) -> dict | None:
    """Return a single threat object by ID, or None if not found."""
    threats = _load_from_pipeline() or fixtures.load_threats()
    for threat in threats:
        if threat.get("threat_id") == threat_id:
            return threat
    return None
