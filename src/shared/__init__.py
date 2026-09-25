
"""Shared helpers used across layers.

Modules:
  contracts.py  the Threat Object schema every layer serializes
  metrics.py    the metric-table container every validated stage produces
  fields.py     the GriddedField container and its xarray bridges
  config.py     configs/ loading, so no stage hard-codes a parameter
  geo.py        spherical distance and bearing
  synthetic.py  deterministic field generators for tests without data

Everything here imports only the standard library plus numpy, so any stage can
use it without pulling in xarray, torch or FastAPI.
"""

from __future__ import annotations

__all__: list[str] = []
