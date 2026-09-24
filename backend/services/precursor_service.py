"""Precursor service adapter.

Responsibilities:
  - Load atmospheric precursor time-series from src/precursors/ (Hariharan's module).
  - Fall back to data/samples/precursors.json fixtures when real output is absent.
  - Apply window_hours filtering to restrict the time series returned.

Integration point for Hariharan (src/precursors/):
  When the precursor engine writes its output, replace _load_from_pipeline()
  with real file/DB reads and update the window filtering if needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import fixtures


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> dict | None:
    """Attempt to load real precursor output.

    # TODO: INTEGRATE (Hariharan - src/precursors/)
    # When the precursor engine writes output (e.g. data/processed/precursors.json),
    # load it here keyed by threat_id and return. Return None to fall back.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "precursors.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["precursors"]
    """
    return None  # pipeline not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_precursors(threat_id: str, window_hours: int = 24) -> dict | None:
    """Return the precursor series for a threat, filtered to the last window_hours.

    Returns None if no precursor data exists for the given threat_id.
    """
    all_precursors = _load_from_pipeline() or fixtures.load_precursors()
    entry = all_precursors.get(threat_id)
    if entry is None:
        return None

    # Apply window_hours filter: keep series points within [now - window_hours, now].
    # For fixtures the 'now' reference is the last point's timestamp; for live data
    # it would be datetime.now(timezone.utc).
    series = entry.get("series", [])
    if series:
        # Use the latest timestamp in the series as the reference horizon.
        try:
            ref_ts = max(
                datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
                for p in series
            )
            cutoff = ref_ts.timestamp() - window_hours * 3600
            series = [
                p
                for p in series
                if datetime.fromisoformat(
                    p["timestamp"].replace("Z", "+00:00")
                ).timestamp()
                >= cutoff
            ]
        except (KeyError, ValueError):
            pass  # leave series unfiltered if timestamps are malformed

    return {**entry, "series": series}
