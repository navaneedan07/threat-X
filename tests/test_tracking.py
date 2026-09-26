"""Permanent regression tests for the core tracking layer.

Verifies:
1. Detection.from_box() mapping from detector output.
2. Single detection track initialization and THR-YYYY-NNNN format.
3. Persistent association across consecutive frames.
4. Multiple detections association and one-to-one matching.
5. New unmatched detections receive distinct sequential IDs.
6. Empty frame / gap handling (null max_gap_steps vs configured coasting).
7. Null threshold semantics (no arbitrary cutoffs invented).
8. Association cost is exactly haversine_km with no motion penalty.

Run:  python -m pytest tests/test_tracking.py -q
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.shared.contracts import THREAT_ID_PATTERN
from src.shared.geo import haversine_km
from src.tracking.tracker import ActiveTrack, Detection, ThreatTracker, TrackingConfig


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
        cell_count=15,
        label_id=1,
    )


class TestDetectionFromBox:
    def test_from_box_reads_all_fields_correctly(self):
        box = {
            "centroid_lat": 13.0827,
            "centroid_lon": 80.2707,
            "lat_min": 12.8,
            "lat_max": 13.3,
            "lon_min": 80.0,
            "lon_max": 80.5,
            "peak_zscore": 3.42,
            "cell_count": 28,
            "label_id": 2,
        }
        timestamp = "2026-09-24T16:00:00Z"
        det = Detection.from_box(box, timestamp)

        assert det.centroid_lat == pytest.approx(13.0827)
        assert det.centroid_lon == pytest.approx(80.2707)
        assert det.lat_min == pytest.approx(12.8)
        assert det.lat_max == pytest.approx(13.3)
        assert det.lon_min == pytest.approx(80.0)
        assert det.lon_max == pytest.approx(80.5)
        assert det.bbox == (12.8, 13.3, 80.0, 80.5)
        assert det.peak_zscore == pytest.approx(3.42)
        assert det.cell_count == 28
        assert det.label_id == 2
        assert det.timestamp == datetime(2026, 9, 24, 16, 0, 0, tzinfo=timezone.utc)


class TestSingleDetectionInitialization:
    def test_first_detection_creates_single_track_with_persistent_id(self):
        tracker = ThreatTracker(default_year=2026)
        det = _det(12.0, 80.0, 0)
        active = tracker.update([det], _dt(0))

        assert len(active) == 1
        assert len(tracker.tracks) == 1
        track = tracker.tracks[0]

        assert THREAT_ID_PATTERN.match(track.threat_id)
        assert track.threat_id == "THR-2026-0001"
        assert track.status == "active"
        assert track.consecutive_misses == 0
        assert len(track.detections) == 1
        assert track.timestamps == [_dt(0)]
        assert track.centroids == [(12.0, 80.0)]
        assert track.latest_centroid == (12.0, 80.0)
        assert track.latest_detection == det


class TestPersistentAssociation:
    def test_same_threat_across_frames_keeps_id_and_grows_history(self):
        tracker = ThreatTracker(default_year=2026)
        det0 = _det(12.0, 80.0, 0)
        det1 = _det(12.2, 80.1, 1)
        det2 = _det(12.4, 80.3, 2)

        tracker.update([det0], _dt(0))
        tracker.update([det1], _dt(1))
        tracker.update([det2], _dt(2))

        assert len(tracker.tracks) == 1
        track = tracker.tracks[0]

        assert track.threat_id == "THR-2026-0001"
        assert track.status == "active"
        assert len(track.detections) == 3
        assert track.timestamps == [_dt(0), _dt(1), _dt(2)]
        assert track.centroids == [(12.0, 80.0), (12.2, 80.1), (12.4, 80.3)]
        assert track.latest_centroid == (12.4, 80.3)
        assert track.consecutive_misses == 0


class TestMultipleDetections:
    def test_two_detections_in_one_frame_create_distinct_ids(self):
        tracker = ThreatTracker(default_year=2026)
        det_a0 = _det(10.0, 70.0, 0)
        det_b0 = _det(20.0, 85.0, 0)

        active = tracker.update([det_a0, det_b0], _dt(0))
        assert len(active) == 2
        assert {t.threat_id for t in active} == {"THR-2026-0001", "THR-2026-0002"}

    def test_two_detections_remain_associated_one_to_one_in_next_frame(self):
        tracker = ThreatTracker(default_year=2026)
        det_a0 = _det(10.0, 70.0, 0)
        det_b0 = _det(20.0, 85.0, 0)
        tracker.update([det_a0, det_b0], _dt(0))

        # Consecutive frame: both shift slightly
        det_a1 = _det(10.1, 70.1, 1)
        det_b1 = _det(20.1, 85.1, 1)
        active = tracker.update([det_a1, det_b1], _dt(1))

        assert len(active) == 2
        track_a = next(t for t in tracker.tracks if t.threat_id == "THR-2026-0001")
        track_b = next(t for t in tracker.tracks if t.threat_id == "THR-2026-0002")

        assert len(track_a.detections) == 2
        assert track_a.centroids == [(10.0, 70.0), (10.1, 70.1)]
        assert len(track_b.detections) == 2
        assert track_b.centroids == [(20.0, 85.0), (20.1, 85.1)]


class TestNewUnmatchedDetection:
    def test_unmatched_new_detection_receives_sequential_id(self):
        tracker = ThreatTracker(default_year=2026)
        tracker.update([_det(10.0, 70.0, 0)], _dt(0))
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].threat_id == "THR-2026-0001"

        # Frame 1: track 1 continues, and a new anomaly emerges
        tracker.update([_det(10.1, 70.1, 1), _det(15.0, 75.0, 1)], _dt(1))
        assert len(tracker.tracks) == 2

        track_1 = next(t for t in tracker.tracks if t.threat_id == "THR-2026-0001")
        track_2 = next(t for t in tracker.tracks if t.threat_id == "THR-2026-0002")

        assert len(track_1.detections) == 2
        assert len(track_2.detections) == 1
        assert track_2.centroids == [(15.0, 75.0)]


class TestEmptyFrameAndGapBehavior:
    def test_null_max_gap_steps_terminates_unmatched_track_immediately(self):
        config = TrackingConfig(max_gap_steps=None)
        tracker = ThreatTracker(config=config, default_year=2026)

        tracker.update([_det(12.0, 80.0, 0)], _dt(0))
        assert tracker.tracks[0].status == "active"

        # Empty frame -> terminates immediately per null semantics (no coasting)
        tracker.update([], _dt(1))
        assert tracker.tracks[0].status == "terminated"
        assert tracker.tracks[0].consecutive_misses == 1

        # Reappearance at frame 2 creates a NEW track, never revives terminated
        tracker.update([_det(12.1, 80.1, 2)], _dt(2))
        assert len(tracker.tracks) == 2
        assert tracker.tracks[0].threat_id == "THR-2026-0001"
        assert tracker.tracks[0].status == "terminated"
        assert tracker.tracks[1].threat_id == "THR-2026-0002"
        assert tracker.tracks[1].status == "active"

    def test_configured_gap_coasts_and_recovers_same_id(self):
        config = TrackingConfig(max_gap_steps=2)
        tracker = ThreatTracker(config=config, default_year=2026)

        tracker.update([_det(12.0, 80.0, 0)], _dt(0))

        # 1 missed frame -> coasting
        tracker.update([], _dt(1))
        assert tracker.tracks[0].status == "coasting"
        assert tracker.tracks[0].consecutive_misses == 1

        # Re-detection on next frame -> re-associated to same ID, status active
        tracker.update([_det(12.2, 80.2, 2)], _dt(2))
        assert len(tracker.tracks) == 1
        track = tracker.tracks[0]
        assert track.threat_id == "THR-2026-0001"
        assert track.status == "active"
        assert track.consecutive_misses == 0
        assert len(track.detections) == 2

    def test_exceeding_configured_max_gap_steps_terminates_track(self):
        config = TrackingConfig(max_gap_steps=1)
        tracker = ThreatTracker(config=config, default_year=2026)

        tracker.update([_det(12.0, 80.0, 0)], _dt(0))

        # 1 miss -> coasting
        tracker.update([], _dt(1))
        assert tracker.tracks[0].status == "coasting"

        # 2 misses > max_gap_steps (1) -> terminated
        tracker.update([], _dt(2))
        assert tracker.tracks[0].status == "terminated"
        assert tracker.tracks[0].consecutive_misses == 2


class TestNullThresholdSemantics:
    def test_shipped_config_loads_null_thresholds_without_fallbacks(self):
        config = TrackingConfig.from_config()
        assert config.max_centroid_distance_km is None
        assert config.min_iou is None
        assert config.intensity_tolerance is None
        assert config.max_gap_steps is None
        assert config.require_motion_consistency is True

        unresolved = set(config.unresolved())
        assert unresolved == {
            "association.max_centroid_distance_km",
            "association.min_iou",
            "association.intensity_tolerance",
            "association.max_gap_steps",
        }

    def test_null_distance_threshold_does_not_invent_distance_cutoff(self):
        # With max_centroid_distance_km=None, a candidate is not rejected by an arbitrary distance
        config = TrackingConfig(max_centroid_distance_km=None)
        tracker = ThreatTracker(config=config, default_year=2026)

        tracker.update([_det(10.0, 70.0, 0)], _dt(0))
        # Large jump: 500 km away
        tracker.update([_det(14.0, 72.0, 1)], _dt(1))

        assert len(tracker.tracks) == 1
        assert len(tracker.tracks[0].detections) == 2

    def test_configured_distance_threshold_enforces_cutoff(self):
        # With an explicit max_centroid_distance_km=100.0, a 500 km displacement rejects matching
        config = TrackingConfig(max_centroid_distance_km=100.0)
        tracker = ThreatTracker(config=config, default_year=2026)

        tracker.update([_det(10.0, 70.0, 0)], _dt(0))
        # 500 km away exceeds 100 km threshold -> cannot associate
        tracker.update([_det(14.0, 72.0, 1)], _dt(1))

        assert len(tracker.tracks) == 2
        assert tracker.tracks[0].threat_id == "THR-2026-0001"
        assert tracker.tracks[0].status == "terminated"  # unmatched in frame 1
        assert tracker.tracks[1].threat_id == "THR-2026-0002"  # spawned as new track


class TestAssociationCost:
    def test_cost_matrix_is_strictly_haversine_distance(self):
        tracker = ThreatTracker(default_year=2026)
        track = ActiveTrack(threat_id="THR-2026-0001", detections=[_det(10.0, 70.0, 0)])
        cand_det = _det(10.5, 70.8, 1)

        cost_matrix = tracker._cost_matrix([track], [cand_det])
        expected_dist = float(haversine_km(10.0, 70.0, 10.5, 70.8))

        assert cost_matrix.shape == (1, 1)
        assert cost_matrix[0, 0] == pytest.approx(expected_dist)

    def test_cost_does_not_apply_motion_penalty(self):
        # Track has 2 points moving northeast
        tracker = ThreatTracker(default_year=2026)
        track = ActiveTrack(
            threat_id="THR-2026-0001",
            detections=[_det(10.0, 70.0, 0), _det(10.5, 70.5, 1)],
        )
        # Candidate sharply reverses direction southwest
        cand_det = _det(10.0, 70.0, 2)

        cost_matrix = tracker._cost_matrix([track], [cand_det])
        expected_dist = float(haversine_km(10.5, 70.5, 10.0, 70.0))

        # Cost must strictly equal pure Haversine distance without invented angular penalty
        assert cost_matrix[0, 0] == pytest.approx(expected_dist)
        assert np.isfinite(cost_matrix[0, 0])


class TestProcessFramesIntegration:
    def test_process_frames_ingests_detector_output_structure(self):
        tracker = ThreatTracker(default_year=2026)
        frames = [
            {
                "time_index": 0,
                "timestamp": "2026-09-01T00:00:00Z",
                "boxes": [
                    {
                        "label_id": 1,
                        "centroid_lat": 13.0,
                        "centroid_lon": 80.0,
                        "lat_min": 12.8,
                        "lat_max": 13.2,
                        "lon_min": 79.8,
                        "lon_max": 80.2,
                        "peak_zscore": 3.4,
                        "cell_count": 20,
                    }
                ],
            },
            {
                "time_index": 1,
                "timestamp": "2026-09-01T03:00:00Z",
                "boxes": [
                    {
                        "label_id": 1,
                        "centroid_lat": 13.2,
                        "centroid_lon": 80.3,
                        "lat_min": 13.0,
                        "lat_max": 13.4,
                        "lon_min": 80.1,
                        "lon_max": 80.5,
                        "peak_zscore": 3.8,
                        "cell_count": 25,
                    }
                ],
            },
        ]
        tracks = tracker.process_frames(frames)
        assert len(tracks) == 1
        assert tracks[0].threat_id == "THR-2026-0001"
        assert len(tracks[0].detections) == 2
        assert tracks[0].centroids == [(13.0, 80.0), (13.2, 80.3)]
