"""Transition service adapter.

Responsibilities:
  - Load transition intelligence output from src/transition/ (Navaneedan).
  - Fall back to data/samples/transitions.json fixtures (which have null
    probabilities) when the model has not yet been calibrated.

Degraded-mode contract (from API spec):
  If the transition model has not produced validated, calibrated output,
  this service returns an object where probability = null and calibrated = false.
  The API returns HTTP 200 with null fields — it does NOT return an error.

Integration point for Navaneedan (src/transition/):
  When the transition model is trained and calibrated, replace
  _load_from_pipeline() with real output reads. The null semantics are
  then automatically resolved by actual probability values.
"""

from __future__ import annotations

from . import fixtures


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> dict | None:
    """Attempt to load real transition model output.

    # TODO: INTEGRATE (Navaneedan - src/transition/)
    # When the TTIE (Threat Transition Intelligence Engine) writes calibrated
    # predictions, load them here keyed by threat_id. Return None to fall back.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "transitions.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["transitions"]
    """
    return None  # transition model not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_transition(threat_id: str) -> dict | None:
    """Return transition intelligence for a threat.

    Returns None if no transition data exists for the given threat_id.
    Returns an object with null probability fields if model is uncalibrated
    (degraded mode — HTTP 200 with null values, as per spec).
    """
    all_transitions = _load_from_pipeline() or fixtures.load_transitions()
    return all_transitions.get(threat_id)
