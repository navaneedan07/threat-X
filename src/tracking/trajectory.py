"""Adapter converting tracker tracks to repository Trajectory contracts.

Reuses shared contracts (Trajectory, TrajectoryPoint) from src.shared.contracts.
Preserves persistent threat IDs, chronological historical steps, and timestamps.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.shared.contracts import Trajectory, TrajectoryPoint
from src.shared.geo import step_distances_km
from src.tracking.tracker import ActiveTrack


def track_to_trajectory(track: ActiveTrack) -> Trajectory:
    """Convert an ActiveTrack into a Trajectory object with historical steps.

    Populates TrajectoryPoint from detection history (timestamp, centroid_lat, centroid_lon).
    Leaves severity as None per TrajectoryPoint contract (no authoritative severity available).
    Leaves forecast_steps empty (reserved for downstream forecasting models).
    """
    sorted_detections = sorted(track.detections, key=lambda d: d.timestamp)
    historical_steps = [
        TrajectoryPoint(
            timestamp=det.timestamp,
            centroid_lat=det.centroid_lat,
            centroid_lon=det.centroid_lon,
            severity=None,
        )
        for det in sorted_detections
    ]
    return Trajectory(
        threat_id=track.threat_id,
        historical_steps=historical_steps,
        forecast_steps=[],
    )


def tracks_to_trajectories(tracks: Iterable[ActiveTrack]) -> dict[str, Trajectory]:
    """Convert a collection of ActiveTracks to a dictionary of Trajectories keyed by threat_id.

    This matches the structure expected by downstream services and fixtures.
    """
    return {track.threat_id: track_to_trajectory(track) for track in tracks}


def trajectory_step_distances_km(trajectory: Trajectory) -> np.ndarray:
    """Compute consecutive step distances along historical points using shared geo helper."""
    centroids = [(pt.centroid_lat, pt.centroid_lon) for pt in trajectory.historical_steps]
    return step_distances_km(centroids)
