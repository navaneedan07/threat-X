"""Focused checks for the GNN-to-anomaly-to-tracking prototype bridge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from src.detection.anomaly_detection import detect_field_anomalies
from src.tracking.pipeline import run_tracking_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "weights" / "gnn" / "run_gnn_tracking_demo.py"
spec = importlib.util.spec_from_file_location("gnn_tracking_demo", DEMO_PATH)
demo = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(demo)


def test_checkpoint_inference_preserves_shape_features_and_nodes():
    prediction, history, nodes, timestamp = demo.predict_next_field()

    assert prediction.shape == (66, 5)
    assert np.isfinite(prediction).all()
    assert history.shape == (6, 66, 5)
    assert [node["node_id"] for node in nodes] == list(range(66))
    assert demo.FEATURE_ORDER == ["temperature", "pressure", "humidity", "wind_speed", "precipitation"]
    assert timestamp.endswith("Z")


def test_anomaly_boxes_convert_to_tracker_detections(tmp_path: Path):
    nodes = [{"node_id": i, "latitude": 10 + (i // 6) * 0.5,
              "longitude": 76 + i % 6} for i in range(66)]
    baseline = np.zeros((6, 66), dtype=float)
    field = np.zeros(66, dtype=float)
    field[[7, 8]] = 3.0
    frames = detect_field_anomalies(field, baseline, nodes, "2020-05-23T00:00:00Z")

    assert len(frames) == 1
    assert len(frames[0]["boxes"]) == 1
    box = frames[0]["boxes"][0]
    assert box["cell_count"] == 2
    assert box["centroid_lat"] == 10.5
    assert box["centroid_lon"] == 77.5

    tracked = run_tracking_pipeline(frames, output_path=tmp_path / "trajectories.json",
                                    default_year=2020, merge_existing=False)
    assert list(tracked["trajectories"]) == ["THR-2020-0001"]
    point = tracked["trajectories"]["THR-2020-0001"]["historical_steps"][0]
    assert point["timestamp"] == "2020-05-23T00:00:00Z"
    assert point["centroid"] == [10.5, 77.5]


def test_end_to_end_demo_writes_inspectable_output(tmp_path: Path):
    output = tmp_path / "demo.json"
    payload = demo.run_demo(output)

    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["predicted_field"]) == 66
    assert saved["_meta"]["feature_order"] == demo.FEATURE_ORDER
    assert len(payload["anomaly_frames"]) == 1
    assert Path(tmp_path / "demo_trajectories.json").exists()
    assert set(payload["threat_ids"]) == set(payload["trajectories"])
