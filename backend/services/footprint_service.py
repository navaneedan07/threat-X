"""Footprint service adapter.

Responsibilities:
  - Load coarse candidate region polygons from src/detection/ (Pushpa).
  - Load 5 km downscaled boundary from src/downscaling/ (Aravinth / Navaneedan).
  - Fall back to data/samples/footprints.json fixtures when real output is absent.
  - Select the coarse or downscaled variant based on the ?downscaled query param.

Integration points:
  - Pushpa (src/detection/): coarse footprint polygons from anomaly clustering.
  - Aravinth / Navaneedan (src/downscaling/): 5 km refined boundary polygons.
"""

from __future__ import annotations

from . import fixtures


# ---------------------------------------------------------------------------
# Pipeline integration point
# ---------------------------------------------------------------------------


def _load_from_pipeline() -> dict | None:
    """Attempt to load real footprint output.

    # TODO: INTEGRATE (Pushpa - src/detection/, Aravinth/Navaneedan - src/downscaling/)
    # When detection/downscaling produce footprint GeoJSON files, load them here
    # keyed by threat_id with 'coarse' and 'downscaled' sub-keys. Return None
    # to fall back to fixtures.

    Example (uncomment and adapt when ready):
        processed_path = Path(os.getenv("DATA_ROOT", "./data")) / "processed" / "footprints.json"
        if processed_path.exists():
            with processed_path.open() as f:
                return json.load(f)["footprints"]
    """
    return None  # pipeline not wired yet


# ---------------------------------------------------------------------------
# Public interface used by routers
# ---------------------------------------------------------------------------


def get_footprint(threat_id: str, downscaled: bool = False) -> dict | None:
    """Return the GeoJSON Feature for a threat's spatial footprint.

    Parameters
    ----------
    threat_id : str
        Persistent threat identifier.
    downscaled : bool
        If True, return the 5 km refined boundary when available, otherwise
        fall back to the coarse boundary.

    Returns None if no footprint exists for the given threat_id.
    """
    all_footprints = _load_from_pipeline() or fixtures.load_footprints()
    entry = all_footprints.get(threat_id)
    if entry is None:
        return None

    if downscaled and "downscaled" in entry:
        return entry["downscaled"]
    return entry.get("coarse", entry)  # always fall back to coarse if downscaled absent
