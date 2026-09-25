"""Atmospheric precursor feature engineering and analysis.

This package computes physically meaningful atmospheric precursor signals
from ERA5 surface-level analysis fields (the available dataset variables).

Available dataset variables (confirmed by dataset inspection 2026-09-25):
  Instantaneous:
    u10   10 m U wind component   (m/s)
    v10   10 m V wind component   (m/s)
    t2m   2 m temperature         (K)
    msl   Mean sea level pressure (Pa)
  Accumulated (per 3-hour step):
    tp    Total precipitation     (m, convert to mm)

Computable precursor signals from these surface fields:
  1. MSLP Tendency (3-hour)   — derived from msl
  2. 10 m Wind Convergence    — derived from u10, v10 (finite-difference divergence)
  3. 10 m Wind Speed          — derived from u10, v10
  4. Precipitation Rate       — derived from tp
  5. 2 m Temperature Anomaly  — derived from t2m relative to dataset mean

NOT computable without pressure-level data (acknowledged absence):
  - Moisture flux convergence at 850 hPa (requires q850, u850, v850)
  - Vorticity at 850 hPa (requires u850, v850)
  - Deep-layer bulk wind shear (requires multi-level wind)
  - Theta-e gradient (requires T, q at multiple levels)
  These are logged as unavailable rather than fabricated.

Public API:
  compute_precursors(ds, centroid_lat, centroid_lon, ...) -> PrecursorResult
  compute_precursor_series(ds, centroid_lat, centroid_lon, ...) -> list[dict]
  generate_explanatory_summary(series) -> str
"""

from src.precursors.engine import (
    PrecursorResult,
    compute_precursors,
    compute_precursor_series,
    generate_explanatory_summary,
)

__all__ = [
    "PrecursorResult",
    "compute_precursors",
    "compute_precursor_series",
    "generate_explanatory_summary",
]
