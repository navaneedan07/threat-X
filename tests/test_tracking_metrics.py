"""Tests for the threat-tracking metrics.

Run:  pytest tests/test_tracking_metrics.py -v

Two properties matter here.

**The mean must not hide a single large excursion.** ``trajectory_error_km`` is the
RMS of the same per-step distances as ``centroid_error_km``; the track below that
is fine for most steps and then jumps scores a small mean and a much larger RMS.

**An identity switch must be visible.** A reference track covered by two different
predicted threat ids is a split, and ``id_consistency`` must fall below 1.0 rather
than silently reporting a clean track.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.shared.contracts import Trajectory, TrajectoryPoint, format_timestamp
from src.validation import tracking_metrics as T

REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_TIME = datetime(2026, 9, 23, tzinfo=timezone.utc)


def _track(threat_id: str, points: list[tuple[int, float, float]]) -> Trajectory:
    """Build a track from ``(hour, lat, lon)`` tuples on a fixed day."""
    return Trajectory(
        threat_id=threat_id,
        historical_steps=[
            TrajectoryPoint(
                timestamp=format_timestamp(_BASE_TIME + timedelta(hours=hour)),
                centroid_lat=lat,
                centroid_lon=lon,
            )
            for hour, lat, lon in points
        ],
    )


_REFERENCE_POINTS = [
    (0, 10.0, 80.0),
    (6, 10.2, 80.3),
    (12, 10.4, 80.6),
    (18, 10.6, 80.9),
    (24, 10.8, 81.2),
]


class TestMatching:
    def test_radius_is_required_and_must_be_positive(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        with pytest.raises(ValueError, match="positive"):
            T.match_tracks(reference, reference, match_radius_km=0.0)

    def test_identical_tracks_match_perfectly(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [_track("THR-2026-0002", _REFERENCE_POINTS)]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        assert T.centroid_error_km(matching) == pytest.approx(0.0)
        assert T.trajectory_error_km(matching) == pytest.approx(0.0)
        assert T.track_continuity(matching) == pytest.approx(1.0)
        assert T.id_consistency(matching) == pytest.approx(1.0)
        assert T.duration_error_hours(matching) == pytest.approx(0.0)

    def test_a_constant_offset_shows_up_as_centroid_error(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        shifted = [(hour, lat + 0.1, lon) for hour, lat, lon in _REFERENCE_POINTS]
        predicted = [_track("THR-2026-0002", shifted)]
        matching = T.match_tracks(predicted, reference, match_radius_km=30.0)
        # 0.1 degrees of latitude is ~11.1 km.
        assert T.centroid_error_km(matching) == pytest.approx(11.1, abs=0.5)

    def test_missing_steps_reduce_continuity(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [_track("THR-2026-0002", _REFERENCE_POINTS[:4])]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        assert T.track_continuity(matching) == pytest.approx(4 / 5)
        assert T.centroid_error_km(matching) == pytest.approx(0.0)

    def test_a_far_step_outside_the_radius_is_a_miss_not_a_match(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [_track("THR-2026-0002", [(0, 40.0, 80.0), *_REFERENCE_POINTS[1:]])]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        assert T.track_continuity(matching) == pytest.approx(4 / 5)

    def test_a_single_large_excursion_is_visible_in_the_rms(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        jumped = list(_REFERENCE_POINTS)
        jumped[2] = (12, 12.0, 80.6)  # ~220 km north of the reference at t=12
        predicted = [_track("THR-2026-0002", jumped)]
        matching = T.match_tracks(predicted, reference, match_radius_km=500.0)
        mean = T.centroid_error_km(matching)
        rms = T.trajectory_error_km(matching)
        assert mean is not None and rms is not None
        assert rms > mean * 1.5, "the RMS must expose the excursion the mean hides"

    def test_split_identity_is_not_consistent(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [
            _track("THR-2026-0002", _REFERENCE_POINTS[:3]),
            _track("THR-2026-0003", _REFERENCE_POINTS[3:]),
        ]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        assert T.track_continuity(matching) == pytest.approx(1.0)
        assert T.id_consistency(matching) == pytest.approx(0.0)
        # A split track has no single id to measure a duration against.
        assert T.duration_error_hours(matching) is None

    def test_duration_error_is_measured_on_the_matched_id(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [_track("THR-2026-0002", _REFERENCE_POINTS[:4])]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        # predicted spans 18 h, reference 24 h.
        assert T.duration_error_hours(matching) == pytest.approx(6.0)

    def test_uncovered_reference_track_has_undefined_identity(self):
        reference = [_track("THR-2026-0001", _REFERENCE_POINTS)]
        predicted = [_track("THR-2026-0002", [(0, 40.0, 80.0)])]
        matching = T.match_tracks(predicted, reference, match_radius_km=20.0)
        assert T.id_consistency(matching) is None
        assert T.centroid_error_km(matching) is None
        assert T.track_continuity(matching) == pytest.approx(0.0)


class TestTrackingTable:
    def test_table_emits_every_config_metric(self):
        import yaml

        declared = yaml.safe_load(
            (REPO_ROOT / "configs" / "validation.yaml").read_text(encoding="utf-8")
        )["tracking"]["metrics"]
        table = T.evaluate_tracking(
            [_track("THR-2026-0002", _REFERENCE_POINTS)],
            [_track("THR-2026-0001", _REFERENCE_POINTS)],
            match_radius_km=20.0,
        )
        missing = [name for name in declared if name not in table.metrics]
        assert not missing, f"config declares metrics the table does not produce: {missing}"

    def test_radius_is_recorded_in_context(self):
        table = T.evaluate_tracking(
            [_track("THR-2026-0002", _REFERENCE_POINTS)],
            [_track("THR-2026-0001", _REFERENCE_POINTS)],
            match_radius_km=15.0,
        )
        assert table.context["match_radius_km"] == 15.0
        assert table.context["reference_steps"] == 5

    def test_empty_reference_has_no_continuity(self):
        table = T.evaluate_tracking([], [], match_radius_km=20.0)
        assert table["track_continuity"] is None
        assert table["centroid_error_km"] is None
        assert "no track steps" in table.notes["track_continuity"]

    def test_units_are_not_attached_to_the_km_and_hour_mix(self):
        table = T.evaluate_tracking(
            [_track("THR-2026-0002", _REFERENCE_POINTS)],
            [_track("THR-2026-0001", _REFERENCE_POINTS)],
            match_radius_km=20.0,
        )
        assert table.units is None
