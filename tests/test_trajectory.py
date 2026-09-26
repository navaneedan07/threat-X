"""Permanent tests for the tracking-to-trajectory adapter.

Verifies:
1. A single-detection track converts correctly to Trajectory.
2. A multi-detection track preserves chronological historical steps.
3. The persistent threat_id is preserved.
4. Latitude, longitude, and timestamp values are preserved with high fidelity.
5. Optional severity remains None when no authoritative severity exists.
6. No ThreatObject or ThreatHistory is created.
7. Existing geographic helpers (step_distances_km, haversine_km) are reused.
8. Empty/insufficient history follows the contract without fabricated data.
9. Multiple tracks map to {threat_id: Trajectory} matching trajectory service schema.
10. Round-trip serialization with Trajectory.to_dict() and from_dict().
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.shared.contracts import THREAT_ID_PATTERN, Trajectory, TrajectoryPoint
from src.shared.geo import haversine_km
from src.tracking.tracker import ActiveTrack, Detection, ThreatTracker
from src.tracking.trajectory import (
    track_to_trajectory,
    tracks_to_trajectories,
    trajectory_step_distances_km,
)


def _dt(hour: int, day: int = 1) -> datetime:
    return datetime(2026, 9, day, hour, 0, 0, tzinfo=timezone.utc)


def _det(lat: float, lon: float, hour: int, zscore: float = 3.0, day: int = 1) -> Detection:
    return Detection(
        centroid_lat=lat,
        centroid_lon=lon,
        timestamp=_dt(hour, day),
        peak_zscore=zscore,
        lat_min=lat - 0.2,
        lat_max=lat + 0.2,
        lon_min=lon - 0.2,
        lon_max=lon + 0.2,
        cell_count=20,
        label_id=1,
    )


class TestSingleDetectionConversion:
    def test_single_detection_converts_correctly(self):
        det = _det(13.0827, 80.2707, 10)
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[det])

        traj = track_to_trajectory(track)

        assert isinstance(traj, Trajectory)
        assert traj.threat_id == "THR-2026-0001"
        assert len(traj.historical_steps) == 1
        assert traj.forecast_steps == []

        point = traj.historical_steps[0]
        assert isinstance(point, TrajectoryPoint)
        assert point.centroid_lat == pytest.approx(13.0827)
        assert point.centroid_lon == pytest.approx(80.2707)
        assert point.timestamp == _dt(10)
        assert point.severity is None


class TestMultiDetectionChronology:
    def test_multi_detection_preserves_chronological_steps(self):
        # Provide detections in out-of-order sequence to verify strict chronological sorting
        det0 = _det(12.0, 79.0, 10)
        det1 = _det(12.5, 79.5, 12)
        det2 = _det(13.0, 80.0, 14)
        det3 = _det(13.5, 80.5, 16)

        track = ActiveTrack(
            threat_id="THR-2026-0002",
            detections=[det2, det0, det3, det1],
        )

        traj = track_to_trajectory(track)

        assert len(traj.historical_steps) == 4
        expected_times = [_dt(10), _dt(12), _dt(14), _dt(16)]
        actual_times = [p.timestamp for p in traj.historical_steps]
        assert actual_times == expected_times

        expected_coords = [(12.0, 79.0), (12.5, 79.5), (13.0, 80.0), (13.5, 80.5)]
        actual_coords = [(p.centroid_lat, p.centroid_lon) for p in traj.historical_steps]
        assert actual_coords == pytest.approx(expected_coords)


class TestThreatIdPreservation:
    def test_persistent_threat_id_preserved_and_valid(self):
        track = ActiveTrack(threat_id="THR-2026-0042", detections=[_det(10.0, 70.0, 1)])
        traj = track_to_trajectory(track)

        assert traj.threat_id == "THR-2026-0042"
        assert THREAT_ID_PATTERN.match(traj.threat_id)


class TestCoordinateAndTimestampFidelity:
    def test_lat_lon_and_timestamp_exact_fidelity(self):
        det = _det(14.123456, 75.987654, 8)
        track = ActiveTrack(threat_id="THR-2026-0005", detections=[det])
        traj = track_to_trajectory(track)

        pt = traj.historical_steps[0]
        assert pt.centroid_lat == 14.123456
        assert pt.centroid_lon == 75.987654
        assert pt.timestamp.tzinfo is not None
        assert pt.timestamp == datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)


class TestOptionalSeverityHandling:
    def test_severity_remains_none_without_authoritative_source(self):
        det = _det(12.0, 80.0, 2, zscore=4.5)  # High z-score but no authoritative severity
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[det])
        traj = track_to_trajectory(track)

        point = traj.historical_steps[0]
        assert point.severity is None

        # Verify serialized dict leaves severity as None, not invented default
        d = traj.to_dict()
        assert d["historical_steps"][0]["severity"] is None


class TestContractIsolation:
    def test_no_threat_object_or_threat_history_created(self):
        det = _det(12.0, 80.0, 0)
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[det])
        result = track_to_trajectory(track)

        # Output must be Trajectory with TrajectoryPoint, never ThreatObject/ThreatHistory
        assert isinstance(result, Trajectory)
        assert not hasattr(result, "event_type")
        assert not hasattr(result, "lifecycle_phase")
        assert not hasattr(result, "snapshots")
        for step in result.historical_steps:
            assert isinstance(step, TrajectoryPoint)


class TestGeographicHelperReuse:
    def test_existing_geo_helpers_reused_for_trajectory_distances(self):
        p0 = _det(12.0, 80.0, 0)
        p1 = _det(12.5, 80.5, 1)
        p2 = _det(13.0, 81.0, 2)
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[p0, p1, p2])
        traj = track_to_trajectory(track)

        distances = trajectory_step_distances_km(traj)
        assert len(distances) == 2

        # Verify values match direct haversine_km calculation from src.shared.geo
        d01 = float(haversine_km(12.0, 80.0, 12.5, 80.5))
        d12 = float(haversine_km(12.5, 80.5, 13.0, 81.0))
        assert distances[0] == pytest.approx(d01)
        assert distances[1] == pytest.approx(d12)


class TestEmptyTrackHandling:
    def test_empty_track_history_produces_empty_historical_steps_without_fabrication(self):
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[])
        traj = track_to_trajectory(track)

        assert isinstance(traj, Trajectory)
        assert traj.threat_id == "THR-2026-0001"
        assert traj.historical_steps == []
        assert traj.forecast_steps == []

        distances = trajectory_step_distances_km(traj)
        assert isinstance(distances, np.ndarray)
        assert len(distances) == 0


class TestMultipleTracksMapping:
    def test_tracks_to_trajectories_maps_by_threat_id(self):
        t1 = ActiveTrack(threat_id="THR-2026-0001", detections=[_det(10.0, 70.0, 0)])
        t2 = ActiveTrack(threat_id="THR-2026-0002", detections=[_det(15.0, 75.0, 0)])

        mapping = tracks_to_trajectories([t1, t2])
        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {"THR-2026-0001", "THR-2026-0002"}
        assert mapping["THR-2026-0001"].threat_id == "THR-2026-0001"
        assert mapping["THR-2026-0002"].threat_id == "THR-2026-0002"
        assert len(mapping["THR-2026-0001"].historical_steps) == 1
        assert len(mapping["THR-2026-0002"].historical_steps) == 1


class TestTrackerEndToEndTrajectoryIntegration:
    def test_tracker_to_trajectory_pipeline(self):
        tracker = ThreatTracker(default_year=2026)
        tracker.update([_det(13.0, 80.0, 0)], _dt(0))
        tracker.update([_det(13.2, 80.2, 1)], _dt(1))
        tracker.update([_det(13.4, 80.4, 2)], _dt(2))

        assert len(tracker.tracks) == 1
        traj = track_to_trajectory(tracker.tracks[0])

        assert traj.threat_id == "THR-2026-0001"
        assert len(traj.historical_steps) == 3

        # Test full round-trip serialization via Trajectory contract
        traj_dict = traj.to_dict()
        assert traj_dict["threat_id"] == "THR-2026-0001"
        assert len(traj_dict["historical_steps"]) == 3
        assert traj_dict["forecast_steps"] == []

        restored = Trajectory.from_dict(traj_dict)
        assert restored.threat_id == traj.threat_id
        assert len(restored.historical_steps) == 3
        assert restored.historical_steps[0].centroid_lat == pytest.approx(13.0)
        assert restored.historical_steps[0].centroid_lon == pytest.approx(80.0)
