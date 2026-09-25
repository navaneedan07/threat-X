"""Geospatial math shared by tracking, lifecycle and validation.

Distances are kilometres on a sphere and bearings are degrees clockwise from true
North — the same convention ``docs/api.md`` specifies for ``direction_deg``.

Spherical formulas, not ellipsoidal: at the scales being measured here (a threat
footprint of tens of km, a track of hundreds) the difference from WGS-84 is far
below the resolution of the grid the centroids came from, and a documented
approximation beats an undisclosed one. If an accuracy-critical distance is ever
needed, it belongs in a geodesic call, not an edit to these constants.

Functions accept scalars or numpy arrays, so a whole track can be passed at once.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
"""Mean earth radius (IUGG). Spherical assumption — see the module docstring."""


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km between two points (or arrays of them)."""
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Initial bearing from point 1 to point 2, in degrees clockwise from North.

    The API reports ``direction_deg`` in this convention, so a threat moving
    north-east is around 45, not 315.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def step_distances_km(centroids: list[tuple[float, float]]) -> np.ndarray:
    """Consecutive displacement between track points, in km.

    Returns an empty array for fewer than two points — an empty result is the
    honest answer for a track with no movement to measure, and the caller must
    decide what an unmeasurable displacement means.
    """
    if len(centroids) < 2:
        return np.empty(0, dtype=float)
    latitudes = np.array([point[0] for point in centroids], dtype=float)
    longitudes = np.array([point[1] for point in centroids], dtype=float)
    return haversine_km(latitudes[:-1], longitudes[:-1], latitudes[1:], longitudes[1:])


def step_bearings_deg(centroids: list[tuple[float, float]]) -> list[float] | None:
    """Consecutive bearings, or ``None`` when the track is too short."""
    if len(centroids) < 2:
        return None
    return [
        bearing_deg(*centroids[index], *centroids[index + 1])
        for index in range(len(centroids) - 1)
    ]


def speed_kmh(distances_km: np.ndarray, step_hours: float) -> np.ndarray | None:
    """Speeds from step distances and a step duration, or ``None`` if undefined.

    A zero or negative ``step_hours`` makes speed meaningless, so it is refused
    rather than producing an infinite value that would silently propagate into a
    threat object.
    """
    if step_hours <= 0:
        return None
    return np.asarray(distances_km, dtype=float) / step_hours
