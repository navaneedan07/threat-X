"""Permanent tests for the tracking pipeline orchestration layer.

Verifies:
1. Multiple detector frames flow through the tracker to trajectories.
2. Multiple boxes in a single frame are preserved as distinct tracks.
3. Trajectories are serialized in the exact JSON structure expected by backend services.
4. Output files can be loaded using the Trajectory contract (Trajectory.from_dict).
5. Empty detector results do not fabricate detections or trajectories.
6. Non-existent output directories are automatically created.
7. Persistent THR-YYYY-NNNN IDs are assigned and preserved.
8. File-based and in-memory detector results are handled transparently.
9. Merging with existing trajectory files preserves previous runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.contracts import THREAT_ID_PATTERN, Trajectory
from src.tracking.pipeline import run_tracking_pipeline


def _sample_detector_frames() -> list[dict[str, Any]]:
    return [
        {
            "time_index": 0,
            "timestamp": "2026-09-24T10:00:00Z",
            "boxes": [
                {
                    "label_id": 1,
                    "centroid_lat": 12.0,
                    "centroid_lon": 80.0,
                    "lat_min": 11.8,
                    "lat_max": 12.2,
                    "lon_min": 79.8,
                    "lon_max": 80.2,
                    "peak_zscore": 3.2,
                    "cell_count": 18,
                }
            ],
        },
        {
            "time_index": 1,
            "timestamp": "2026-09-24T12:00:00Z",
            "boxes": [
                {
                    "label_id": 1,
                    "centroid_lat": 12.3,
                    "centroid_lon": 80.3,
                    "lat_min": 12.1,
                    "lat_max": 12.5,
                    "lon_min": 80.1,
                    "lon_max": 80.5,
                    "peak_zscore": 3.6,
                    "cell_count": 22,
                }
            ],
        },
    ]


class TestPipelineOrchestration:
    def test_multiple_frames_flow_through_tracker(self, tmp_path: Path):
        frames = _sample_detector_frames()
        out_file = tmp_path / "trajectories.json"

        result = run_tracking_pipeline(frames, output_path=out_file, default_year=2026)

        assert "trajectories" in result
        assert "_meta" in result
        assert len(result["trajectories"]) == 1

        threat_id = "THR-2026-0001"
        assert threat_id in result["trajectories"]
        traj_data = result["trajectories"][threat_id]
        assert len(traj_data["historical_steps"]) == 2
        assert traj_data["historical_steps"][0]["centroid"] == [12.0, 80.0]
        assert traj_data["historical_steps"][1]["centroid"] == [12.3, 80.3]

    def test_multiple_boxes_preserved_as_distinct_tracks(self, tmp_path: Path):
        multi_box_frames = [
            {
                "time_index": 0,
                "timestamp": "2026-09-24T10:00:00Z",
                "boxes": [
                    {
                        "label_id": 1,
                        "centroid_lat": 10.0,
                        "centroid_lon": 70.0,
                        "peak_zscore": 3.0,
                        "cell_count": 15,
                    },
                    {
                        "label_id": 2,
                        "centroid_lat": 20.0,
                        "centroid_lon": 85.0,
                        "peak_zscore": 4.0,
                        "cell_count": 25,
                    },
                ],
            },
            {
                "time_index": 1,
                "timestamp": "2026-09-24T12:00:00Z",
                "boxes": [
                    {
                        "label_id": 1,
                        "centroid_lat": 10.1,
                        "centroid_lon": 70.1,
                        "peak_zscore": 3.2,
                        "cell_count": 16,
                    },
                    {
                        "label_id": 2,
                        "centroid_lat": 20.1,
                        "centroid_lon": 85.1,
                        "peak_zscore": 4.1,
                        "cell_count": 26,
                    },
                ],
            },
        ]
        out_file = tmp_path / "trajectories.json"
        result = run_tracking_pipeline(multi_box_frames, output_path=out_file, default_year=2026)

        assert len(result["trajectories"]) == 2
        assert "THR-2026-0001" in result["trajectories"]
        assert "THR-2026-0002" in result["trajectories"]

        t1 = result["trajectories"]["THR-2026-0001"]
        t2 = result["trajectories"]["THR-2026-0002"]
        assert len(t1["historical_steps"]) == 2
        assert len(t2["historical_steps"]) == 2

    def test_output_json_structure_matches_trajectory_service(self, tmp_path: Path):
        frames = _sample_detector_frames()
        out_file = tmp_path / "trajectories.json"
        run_tracking_pipeline(frames, output_path=out_file, default_year=2026)

        assert out_file.exists()
        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Structure matches data/samples/trajectories.json expected by backend/services/trajectory_service.py
        assert "trajectories" in data
        assert isinstance(data["trajectories"], dict)
        assert "_meta" in data
        assert "generated_at" in data["_meta"]
        assert data["_meta"]["source"] == "src.tracking.pipeline"

    def test_output_can_be_loaded_using_trajectory_contract(self, tmp_path: Path):
        frames = _sample_detector_frames()
        out_file = tmp_path / "trajectories.json"
        run_tracking_pipeline(frames, output_path=out_file, default_year=2026)

        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for threat_id, traj_dict in data["trajectories"].items():
            assert THREAT_ID_PATTERN.match(threat_id)
            # Must deserialize cleanly through Trajectory.from_dict()
            restored = Trajectory.from_dict(traj_dict)
            assert restored.threat_id == threat_id
            assert len(restored.historical_steps) == 2
            assert restored.historical_steps[0].centroid_lat == pytest.approx(12.0)
            assert restored.historical_steps[0].centroid_lon == pytest.approx(80.0)

    def test_empty_detector_results_do_not_fabricate_trajectories(self, tmp_path: Path):
        out_file = tmp_path / "trajectories.json"
        result = run_tracking_pipeline([], output_path=out_file, default_year=2026)

        assert result["trajectories"] == {}
        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["trajectories"] == {}

    def test_nested_output_directory_creation(self, tmp_path: Path):
        deep_dir = tmp_path / "deep" / "nested" / "output"
        out_file = deep_dir / "trajectories.json"
        assert not deep_dir.exists()

        run_tracking_pipeline(_sample_detector_frames(), output_path=out_file, default_year=2026)

        assert deep_dir.exists()
        assert out_file.exists()

    def test_threat_id_format_preserved(self, tmp_path: Path):
        frames = _sample_detector_frames()
        out_file = tmp_path / "trajectories.json"
        result = run_tracking_pipeline(frames, output_path=out_file, default_year=2020)

        threat_id = list(result["trajectories"].keys())[0]
        assert threat_id == "THR-2020-0001"
        assert THREAT_ID_PATTERN.match(threat_id)

    def test_json_file_input_compatibility(self, tmp_path: Path):
        # Write detector results to JSON simulating src/detection/anomaly_detection.py output
        detector_out_file = tmp_path / "detector_output.json"
        detector_payload = {
            "frames": _sample_detector_frames(),
            "trajectory": [],
        }
        with detector_out_file.open("w", encoding="utf-8") as f:
            json.dump(detector_payload, f)

        out_file = tmp_path / "processed_trajectories.json"
        result = run_tracking_pipeline(
            detector_results=detector_out_file,
            output_path=out_file,
            default_year=2026,
        )

        assert "THR-2026-0001" in result["trajectories"]
        assert len(result["trajectories"]["THR-2026-0001"]["historical_steps"]) == 2

    def test_merges_multiple_pipeline_runs(self, tmp_path: Path):
        out_file = tmp_path / "trajectories.json"

        # Run 1: Year 2020 event
        f1 = [
            {
                "time_index": 0,
                "timestamp": "2020-05-18T00:00:00Z",
                "boxes": [{"centroid_lat": 14.0, "centroid_lon": 87.0}],
            }
        ]
        run_tracking_pipeline(f1, output_path=out_file, default_year=2020)

        # Run 2: Year 2022 event with merge_existing=True
        f2 = [
            {
                "time_index": 0,
                "timestamp": "2022-05-01T00:00:00Z",
                "boxes": [{"centroid_lat": 28.5, "centroid_lon": 77.0}],
            }
        ]
        result = run_tracking_pipeline(
            f2, output_path=out_file, default_year=2022, merge_existing=True
        )

        assert "THR-2020-0001" in result["trajectories"]
        assert "THR-2022-0001" in result["trajectories"]
