"""Metrics for the threat-tracking stage.

Centroid error, trajectory error, track continuity, ID consistency and duration
error, computed by matching predicted tracks against reference tracks.

How matching works
------------------
A reference track is a sequence of timestamped centroids. A predicted track is the
same. For every reference observation, the nearest predicted observation **at the
same timestamp** within ``match_radius_km`` is the match; a reference step with no
prediction inside the radius is a miss. This keeps the metrics about *tracking*
(persistence, identity, position) rather than re-scoring detection, which has its
own module.

``match_radius_km`` is required and has no default. It is a matching tolerance, and
a defaulted one would be a guessed threshold baked into every score — the same
reason ``iou`` in the downscaling metrics requires its threshold and
``configs/tracking.yaml`` keeps its association distances ``null``.

Metric definitions
------------------
| Metric | Definition |
|---|---|
| ``centroid_error_km`` | mean great-circle distance between matched steps |
| ``trajectory_error_km`` | RMS of the same distances — exposes one jump the mean hides |
| ``track_continuity`` | matched reference steps / all reference steps |
| ``id_consistency`` | share of covered reference tracks served by one predicted id |
| ``duration_error_hours`` | mean lifetime error, over tracks matched to a single predicted id |

Null discipline: every metric returns ``None`` when its denominator is zero.
``track_continuity`` with no reference steps is undefined, not ``1.0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.shared.contracts import Trajectory
from src.shared.geo import haversine_km
from src.shared.metrics import MetricTable

__all__ = [
    "TrackMatching",
    "centroid_error_km",
    "duration_error_hours",
    "evaluate_tracking",
    "id_consistency",
    "match_tracks",
    "track_continuity",
    "trajectory_error_km",
]


@dataclass
class TrackMatching:
    """The association between a predicted and a reference track set.

    Carries the raw distances and counts so each metric is a pure function of this
    structure and can be asserted independently.
    """

    distances_km: list[float] = field(default_factory=list)
    duration_errors_hours: list[float] = field(default_factory=list)
    reference_steps: int = 0
    covered_steps: int = 0
    reference_tracks: int = 0
    covered_tracks: int = 0
    consistent_tracks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_steps": self.reference_steps,
            "covered_steps": self.covered_steps,
            "reference_tracks": self.reference_tracks,
            "covered_tracks": self.covered_tracks,
            "consistent_tracks": self.consistent_tracks,
            "matched_pairs": len(self.distances_km),
        }


def _ordered_points(track: Trajectory):
    """A track's history sorted by timestamp."""
    return sorted(track.historical_steps, key=lambda point: point.timestamp)


def _duration_hours(timestamps: list[datetime]) -> float | None:
    if len(timestamps) < 2:
        return None
    return (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0


def match_tracks(
    predicted: Sequence[Trajectory],
    reference: Sequence[Trajectory],
    *,
    match_radius_km: float,
) -> TrackMatching:
    """Associate predicted tracks to reference tracks step by step.

    Raises ``ValueError`` for a non-positive radius: a zero tolerance would match
    only exactly coincident centroids, which is a threshold that means something
    other than what the caller intended.
    """
    if match_radius_km <= 0:
        raise ValueError("match_radius_km must be positive")

    by_time: dict[datetime, list[tuple[str, float, float]]] = {}
    predicted_span: dict[str, list[datetime]] = {}
    for track in predicted:
        points = _ordered_points(track)
        for point in points:
            by_time.setdefault(point.timestamp, []).append(
                (track.threat_id, point.centroid_lat, point.centroid_lon)
            )
        if points:
            predicted_span[track.threat_id] = [point.timestamp for point in points]

    matching = TrackMatching()
    for track in reference:
        reference_points = _ordered_points(track)
        if not reference_points:
            continue
        matching.reference_steps += len(reference_points)
        matching.reference_tracks += 1

        used_ids: set[str] = set()
        covered = False
        for point in reference_points:
            candidates = by_time.get(point.timestamp)
            if not candidates:
                continue
            distances = [
                (
                    float(
                        haversine_km(
                            point.centroid_lat, point.centroid_lon, lat, lon
                        )
                    ),
                    threat_id,
                )
                for threat_id, lat, lon in candidates
            ]
            distance, threat_id = min(distances, key=lambda item: item[0])
            if distance <= match_radius_km:
                matching.distances_km.append(distance)
                matching.covered_steps += 1
                used_ids.add(threat_id)
                covered = True

        if not covered:
            continue
        matching.covered_tracks += 1
        if len(used_ids) == 1:
            matching.consistent_tracks += 1
            matched_id = next(iter(used_ids))
            predicted_duration = _duration_hours(predicted_span.get(matched_id, []))
            reference_duration = _duration_hours(
                [point.timestamp for point in reference_points]
            )
            if predicted_duration is not None and reference_duration is not None:
                matching.duration_errors_hours.append(
                    abs(predicted_duration - reference_duration)
                )

    return matching


def centroid_error_km(matching: TrackMatching) -> float | None:
    """Mean position error over matched steps, or ``None`` if nothing matched."""
    if not matching.distances_km:
        return None
    return float(np.mean(matching.distances_km))


def trajectory_error_km(matching: TrackMatching) -> float | None:
    """RMS position error over matched steps, or ``None`` if nothing matched.

    Reported alongside the mean because a track that follows the reference for ten
    steps and then jumps 200 km scores a modest mean and a large RMS — the second
    number is the one that reveals the jump.
    """
    if not matching.distances_km:
        return None
    return float(np.sqrt(np.mean(np.square(matching.distances_km))))


def track_continuity(matching: TrackMatching) -> float | None:
    """Share of reference steps that had a prediction within the radius."""
    if matching.reference_steps == 0:
        return None
    return float(matching.covered_steps / matching.reference_steps)


def id_consistency(matching: TrackMatching) -> float | None:
    """Share of covered reference tracks served by a single predicted id.

    A reference track whose observations are split across two predicted ids is an
    identity switch. Tracks that were never covered are excluded — an unmatched
    track is a continuity failure, and counting it twice would double-penalise the
    same error. ``None`` when no reference track was covered at all.
    """
    if matching.covered_tracks == 0:
        return None
    return float(matching.consistent_tracks / matching.covered_tracks)


def duration_error_hours(matching: TrackMatching) -> float | None:
    """Mean absolute lifetime error, over tracks matched to a single predicted id.

    ``None`` when no track was long enough on both sides to have a duration, so a
    pair of single-step tracks cannot report a perfect ``0.0``.
    """
    if not matching.duration_errors_hours:
        return None
    return float(np.mean(matching.duration_errors_hours))


def evaluate_tracking(
    predicted: Sequence[Trajectory],
    reference: Sequence[Trajectory],
    *,
    match_radius_km: float,
    context: dict[str, Any] | None = None,
) -> MetricTable:
    """Full tracking metric table.

    Metric names match ``configs/validation.yaml`` → ``tracking.metrics``
    (``centroid_error_km``, ``trajectory_error_km``, ``track_continuity``,
    ``id_consistency``, ``duration_error_hours``).
    """
    matching = match_tracks(predicted, reference, match_radius_km=match_radius_km)
    table = MetricTable(
        # Mixed units across the metric set (km and hours), and each name already
        # ends in its unit, so the table carries no single unit of its own.
        units=None,
        context={
            "match_radius_km": float(match_radius_km),
            **matching.to_dict(),
            **(context or {}),
        },
    )

    table.metrics["centroid_error_km"] = centroid_error_km(matching)
    table.metrics["trajectory_error_km"] = trajectory_error_km(matching)
    table.metrics["track_continuity"] = track_continuity(matching)
    table.metrics["id_consistency"] = id_consistency(matching)
    table.metrics["duration_error_hours"] = duration_error_hours(matching)

    if matching.reference_steps == 0:
        table.notes["track_continuity"] = "the reference has no track steps"
    elif matching.covered_steps == 0:
        table.notes["track_continuity"] = (
            "no predicted observation fell within the match radius of any "
            "reference step"
        )
    if matching.covered_tracks == 0:
        table.notes["id_consistency"] = "no reference track was covered"
    if not matching.duration_errors_hours:
        table.notes["duration_error_hours"] = (
            "no covered track was long enough on both sides to have a duration"
        )
    return table
