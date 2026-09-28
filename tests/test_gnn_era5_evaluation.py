"""Tests for external ERA5 evaluation of the frozen GNN checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "weights" / "gnn" / "08_evaluate_era5_pilot.py"
SPEC = importlib.util.spec_from_file_location("gnn_era5_pilot_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


@pytest.fixture(scope="module")
def pilot():
    return evaluator.load_pilot_data()


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("era5-pilot-evaluation")
    return evaluator.evaluate(output_dir=output_dir)


def test_loads_validated_pilot_tensor_features_and_node_order(pilot):
    assert pilot.tensor.shape == (168, 66, 5)
    assert np.isfinite(pilot.tensor).all()
    assert pilot.feature_names == evaluator.FEATURE_ORDER
    assert pilot.feature_units == ("degC", "hPa", "%", "m/s", "mm/hour")
    assert [node["node_id"] for node in pilot.nodes] == list(range(66))
    assert pilot.timestamps[0] == "2020-05-16T00:00:00Z"
    assert pilot.timestamps[-1] == "2020-05-22T23:00:00Z"


def test_six_step_windows_target_the_next_hour_and_preserve_timestamps(pilot):
    windows = evaluator.evaluation_windows(len(pilot.timestamps))

    assert len(windows) == 162
    first, last = windows[0], windows[-1]
    assert first.input_indices == (0, 1, 2, 3, 4, 5)
    assert first.target_index == 6
    assert pilot.timestamps[first.target_index] == "2020-05-16T06:00:00Z"
    assert last.input_indices == (161, 162, 163, 164, 165, 166)
    assert last.target_index == 167
    assert pilot.timestamps[last.target_index] == "2020-05-22T23:00:00Z"
    for window in windows:
        assert window.input_indices[-1] == window.target_index - 1
        assert window.input_indices == tuple(range(window.target_index - 6, window.target_index))


def test_metric_values_are_physical_rmse_and_mae():
    prediction = np.array([1.0, 3.0])
    target = np.array([0.0, 1.0])

    rmse, mae = evaluator.metric_values(prediction, target)

    assert rmse == pytest.approx(np.sqrt(2.5))
    assert mae == pytest.approx(1.5)


def test_graph_baseline_reuses_repository_formula():
    baseline = evaluator.load_baseline_module()
    edge_index = np.load(evaluator.EDGE_INDEX_PATH, allow_pickle=False)
    edge_weight = np.load(evaluator.EDGE_WEIGHT_PATH, allow_pickle=False)
    adjacency = baseline.build_normalized_adjacency(edge_index, edge_weight, 66)
    last_state = np.arange(66 * 5, dtype=np.float32).reshape(66, 5) / 100.0

    actual = evaluator.graph_smoothing_prediction(last_state, adjacency, baseline.ALPHA)
    expected = baseline.ALPHA * last_state + (1 - baseline.ALPHA) * (adjacency @ last_state)

    np.testing.assert_allclose(actual, expected)


def test_outputs_have_physical_metrics_all_features_and_finite_values(evaluation):
    assert len(evaluation["metrics"]) == 5
    assert [row["feature"] for row in evaluation["metrics"]] == list(evaluator.FEATURE_ORDER)
    assert all(row["prediction_count"] == 162 * 66 for row in evaluation["metrics"])
    assert all(row["unit"] in {"degC", "hPa", "%", "m/s", "mm/hour"} for row in evaluation["metrics"])
    for row in evaluation["metrics"]:
        for key in ("gnn_rmse", "gnn_mae", "persistence_rmse", "persistence_mae",
                    "smoothing_rmse", "smoothing_mae"):
            assert np.isfinite(row[key])

    output_dir = Path(evaluation["output_dir"])
    metadata = json.loads((output_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation"] == "retrospective one-hour-ahead evaluation on ERA5 reanalysis"
    assert metadata["evaluation_window"]["timestamp_count"] == 162
    assert metadata["evaluation_window"]["node_count"] == 66
    assert metadata["evaluation_window"]["node_field_prediction_count"] == 162 * 66
    assert metadata["evaluation_window"]["scalar_feature_prediction_count"] == 162 * 66 * 5
    metrics_csv = list(csv_rows(output_dir / "metrics.csv"))
    diagnostics_csv = list(csv_rows(output_dir / "domain_diagnostics.csv"))
    assert len(metrics_csv) == len(diagnostics_csv) == 5
    for row in metrics_csv + diagnostics_csv:
        for key, value in row.items():
            if key not in {"feature", "unit"}:
                assert np.isfinite(float(value))


def csv_rows(path: Path):
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def test_domain_diagnostics_include_five_features_and_expose_scale_mismatch(evaluation):
    rows = evaluation["domain_diagnostics"]
    assert [row["feature"] for row in rows] == list(evaluator.FEATURE_ORDER)
    assert all(row["checkpoint_training_std"] > 0 for row in rows)
    assert all(np.isfinite(row["pilot_std_over_checkpoint_std"]) for row in rows)
    assert all(row["pilot_std_over_checkpoint_std"] >= 0 for row in rows)
    assert rows[1]["pilot_std_over_checkpoint_std"] < 0.1
    assert rows[4]["pilot_std_over_checkpoint_std"] < 0.1


def test_checkpoint_hash_and_modification_time_are_unchanged(evaluation):
    path = evaluator.CHECKPOINT_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = evaluation["metadata"]["checkpoint"]

    assert metadata["sha256_before"] == metadata["sha256_after"] == digest
    assert metadata["mtime_ns_before"] == metadata["mtime_ns_after"] == path.stat().st_mtime_ns
