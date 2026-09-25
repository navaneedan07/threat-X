"""Precursor service adapter.

Responsibilities:
  - Load atmospheric precursor time-series from src/precursors/ pipeline output.
  - Fall back to data/samples/precursors.json fixtures when real output is absent.
  - Apply window_hours filtering to restrict the time series returned.

Integration status (2026-09-25):
  IMPLEMENTED — src/precursors/ is wired. Real computed outputs are written
  to data/processed/precursors.json by running:
      python -m src.precursors.pipeline

  The service loads real output when available, and falls back to fixtures
  for legacy threat IDs (e.g. THR-2026-0001) that are not in the real output.

Data source note:
  Real outputs come from ERA5 surface-level data (u10, v10, t2m, msl, tp).
  Pressure-level variables (850 hPa vorticity, moisture flux convergence,
  bulk shear, theta-e gradient) are None because they are not in the dataset.
  These nulls are accurate representations — not fabrication failures.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from . import fixtures

logger = logging.getLogger(__name__)

# Path to the processed output from src/precursors/pipeline.py
_DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
_PROCESSED_PRECURSORS = _DATA_ROOT / "processed" / "precursors.json"


def _load_from_pipeline() -> dict | None:
    """Load real precursor output from data/processed/precursors.json.

    Returns dict keyed by threat_id, or None if not available.
    """
    if not _PROCESSED_PRECURSORS.exists():
        return None
    try:
        with _PROCESSED_PRECURSORS.open("r", encoding="utf-8") as f:
            data = json.load(f)
        precursors = data.get("precursors", {})
        if precursors:
            logger.debug(
                "Loaded real precursor output: %d threats from %s",
                len(precursors), _PROCESSED_PRECURSORS,
            )
            return precursors
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning("Could not read processed precursors: %s", exc)
    return None


def get_precursors(threat_id: str, window_hours: int = 24) -> dict | None:
    """Return the precursor series for a threat, filtered to the last window_hours.

    Returns None if no precursor data exists for the given threat_id.
    Checks real pipeline output first, then falls back to demo fixtures.
    """
    # Check real computed output first
    real = _load_from_pipeline()
    if real is not None and threat_id in real:
        entry = real[threat_id]
    else:
        # Fall back to fixtures (covers THR-2026-0001 demo ID)
        all_fixtures = fixtures.load_precursors()
        entry = all_fixtures.get(threat_id)

    if entry is None:
        return None

    # Apply window_hours filter over the series
    series = entry.get("series", [])
    if series:
        try:
            ref_ts = max(
                datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
                for p in series
            )
            cutoff = ref_ts.timestamp() - window_hours * 3600
            series = [
                p for p in series
                if datetime.fromisoformat(
                    p["timestamp"].replace("Z", "+00:00")
                ).timestamp() >= cutoff
            ]
        except (KeyError, ValueError) as exc:
            logger.warning("Timestamp filtering failed: %s", exc)

    return {**entry, "series": series}
