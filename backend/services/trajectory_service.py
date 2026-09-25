"""Trajectory service adapter.

Responsibilities:
  - Load historical track points from src/tracking/ (Sachin).
  - Load forecast path vectors from src/models/gnn/ (Varnika).
  - Fall back to data/samples/trajectories.json fixtures when real output is absent.

Integration points:
  - Sachin (src/tracking/): historical_steps come from the threat tracker output.
  - Varnika (src/models/gnn/): forecast_steps come from the GNN path predictor.
    When Varnika's model produces path vectors, replace the forecast section of
    _load_from_pipeline() with real reads.
"""

from __future__ import annotations

from . import fixtures


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> dict | None:
    """Attempt to load real trajectory output.

    # TODO: INTEGRATE (Sachin - src/tracking/, Varnika - src/models/gnn/)
    # When tracking / GNN produce a trajectory file, load it here keyed by
    # threat_id and return. Return None to fall back to fixtures.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "trajectories.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["trajectories"]
    """
    return None  # pipeline not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_trajectory(threat_id: str) -> dict | None:
    """Return trajectory data (historical + forecast) for a threat.

    Returns None if no trajectory exists for the given threat_id.
    """
    all_trajectories = _load_from_pipeline() or fixtures.load_trajectories()
    return all_trajectories.get(threat_id)
