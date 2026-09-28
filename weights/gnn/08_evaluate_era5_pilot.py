"""Evaluate the frozen GNN checkpoint retrospectively on the ERA5 pilot.

This is an external, retrospective one-hour-ahead evaluation on ERA5
reanalysis. It does not train or tune the checkpoint. Predictions use the
checkpoint's stored normalization statistics, and each rolling one-step
window uses the observed ERA5 history available immediately before its target.

Run from the repository root with::

    conda run -n threat-x python -B weights/gnn/08_evaluate_era5_pilot.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = BASE_DIR / "outputs"
PILOT_DIR = OUTPUTS_DIR / "era5_pilot"
CHECKPOINT_PATH = OUTPUTS_DIR / "st_gnn_checkpoint.pt"
NODE_TABLE_PATH = OUTPUTS_DIR / "node_table.csv"
EDGE_INDEX_PATH = OUTPUTS_DIR / "edge_index.npy"
EDGE_WEIGHT_PATH = OUTPUTS_DIR / "edge_weight.npy"
FEATURE_ORDER = ("temperature", "pressure", "humidity", "wind_speed", "precipitation")
HISTORY_STEPS = 6


@dataclass(frozen=True)
class PilotData:
    tensor: np.ndarray
    timestamps: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    nodes: tuple[dict[str, float | int], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PredictionWindow:
    input_indices: tuple[int, ...]
    target_index: int


def load_pilot_data(
    pilot_dir: str | Path = PILOT_DIR,
    node_table_path: str | Path = NODE_TABLE_PATH,
) -> PilotData:
    """Load and validate the prepared tensor, timestamps, metadata, and node order."""
    pilot_dir = Path(pilot_dir)
    tensor = np.load(pilot_dir / "tensor.npy", allow_pickle=False)
    metadata = json.loads((pilot_dir / "metadata.json").read_text(encoding="utf-8"))
    if tensor.ndim != 3 or tensor.shape[1:] != (66, 5) or not np.isfinite(tensor).all():
        raise ValueError(f"Pilot tensor must be finite with shape (T,66,5); got {tensor.shape}")

    metadata_shape = tuple(metadata.get("tensor", {}).get("shape", ()))
    if metadata_shape != tensor.shape:
        raise ValueError(f"Tensor shape {tensor.shape} does not match metadata {metadata_shape}")
    features = sorted(metadata.get("features", []), key=lambda feature: feature["index"])
    feature_names = tuple(feature["name"] for feature in features)
    feature_units = tuple(feature["units"] for feature in features)
    metadata_order = tuple(metadata.get("tensor", {}).get("feature_order", ()))
    if feature_names != FEATURE_ORDER or metadata_order != FEATURE_ORDER:
        raise ValueError("Pilot feature order does not match the checkpoint feature contract")
    if len(features) != 5 or not all(feature_units):
        raise ValueError("Pilot metadata must specify units for all five features")

    with (pilot_dir / "timestamps.csv").open(encoding="utf-8", newline="") as handle:
        timestamps = tuple(row["valid_time"] for row in csv.DictReader(handle))
    if len(timestamps) != tensor.shape[0]:
        raise ValueError("Pilot timestamp count does not match tensor length")
    parsed = tuple(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in timestamps)
    if any(right - left != timedelta(hours=1) for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Pilot timestamps must be contiguous hourly ERA5 valid times")

    nodes_raw = metadata.get("nodes", [])
    if len(nodes_raw) != 66 or [node.get("node_id") for node in nodes_raw] != list(range(66)):
        raise ValueError("Pilot metadata nodes must be in node_id order 0..65")
    nodes: tuple[dict[str, float | int], ...] = tuple(
        {
            "node_id": int(node["node_id"]),
            "latitude": float(node["latitude"]),
            "longitude": float(node["longitude"]),
        }
        for node in nodes_raw
    )
    with Path(node_table_path).open(encoding="utf-8", newline="") as handle:
        node_table = list(csv.DictReader(handle))
    table_nodes = tuple(
        (int(node["node_id"]), float(node["latitude"]), float(node["longitude"]))
        for node in node_table
    )
    metadata_nodes = tuple(
        (int(node["node_id"]), float(node["latitude"]), float(node["longitude"]))
        for node in nodes
    )
    if table_nodes != metadata_nodes:
        raise ValueError("Pilot metadata node ordering/coordinates do not match node_table.csv")

    validation = metadata.get("validation", {})
    if validation.get("node_order_matches_node_table") is not True:
        raise ValueError("Pilot metadata does not confirm node order validation")
    if validation.get("exact_coordinate_selection") is not True or validation.get("interpolation") is not False:
        raise ValueError("Pilot metadata does not confirm exact, non-interpolated node selection")
    return PilotData(tensor, timestamps, feature_names, feature_units, nodes, metadata)


def load_model_class():
    """Import the existing architecture definition without running its trainer."""
    model_path = BASE_DIR / "05_st_gnn_model.py"
    spec = importlib.util.spec_from_file_location("threat_x_frozen_st_gnn", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model definition at {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpatioTemporalGNN


def load_baseline_module():
    """Load the repository's fixed graph-smoothing implementation."""
    baseline_path = BASE_DIR / "03_baseline_predictor.py"
    spec = importlib.util.spec_from_file_location("threat_x_existing_gnn_baseline", baseline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline implementation at {baseline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_windows(timestamp_count: int, history_steps: int = HISTORY_STEPS) -> tuple[PredictionWindow, ...]:
    """Return rolling one-step windows after the initial history-only hours."""
    if timestamp_count <= history_steps:
        raise ValueError("Pilot must contain at least one target after the history window")
    return tuple(
        PredictionWindow(tuple(range(target - history_steps, target)), target)
        for target in range(history_steps, timestamp_count)
    )


def metric_values(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Calculate RMSE and MAE in the units of the supplied physical arrays."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction and target arrays must have the same non-empty shape")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target arrays must be finite")
    error = prediction - target
    return float(np.sqrt(np.mean(np.square(error)))), float(np.mean(np.abs(error)))


def graph_smoothing_prediction(last_state_normalized: np.ndarray, adjacency: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the existing fixed persistence/graph-smoothing formula."""
    smoothed = adjacency @ last_state_normalized
    return alpha * last_state_normalized + (1.0 - alpha) * smoothed


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_names") != list(FEATURE_ORDER):
        raise ValueError("Checkpoint feature order does not match the ERA5 pilot")
    if checkpoint.get("t_in") != HISTORY_STEPS or checkpoint.get("n_features") != len(FEATURE_ORDER):
        raise ValueError("Checkpoint input length or feature count does not match the evaluation design")
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32).reshape(-1)
    if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all():
        raise ValueError("Checkpoint normalization statistics are invalid")
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Checkpoint normalization standard deviations must be positive and finite")
    return checkpoint, mean, std


def evaluate(
    pilot_dir: str | Path = PILOT_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    output_dir: str | Path = OUTPUTS_DIR / "era5_pilot_evaluation",
) -> dict[str, Any]:
    """Run frozen-checkpoint evaluation and write compact CSV/JSON results."""
    checkpoint_path = Path(checkpoint_path)
    before_stat = checkpoint_path.stat()
    hash_before = _checkpoint_hash(checkpoint_path)

    pilot = load_pilot_data(pilot_dir)
    checkpoint, mean, std = _load_checkpoint(checkpoint_path)
    windows = evaluation_windows(len(pilot.timestamps))
    baseline = load_baseline_module()
    edge_index = np.load(EDGE_INDEX_PATH, allow_pickle=False)
    edge_weight = np.load(EDGE_WEIGHT_PATH, allow_pickle=False)
    adjacency = baseline.build_normalized_adjacency(edge_index, edge_weight, n_nodes=66)
    alpha = float(baseline.ALPHA)
    if edge_index.shape[0] != 2 or edge_weight.shape != (edge_index.shape[1],):
        raise ValueError("Existing graph edge artifacts have incompatible shapes")

    model = load_model_class()(
        n_features=checkpoint["n_features"],
        hidden_dim=checkpoint["hidden_dim"],
        embed_dim=checkpoint["embed_dim"],
        n_processor_steps=checkpoint["n_processor_steps"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gnn_predictions: list[np.ndarray] = []
    persistence_predictions: list[np.ndarray] = []
    smoothing_predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    for window in windows:
        input_physical = pilot.tensor[list(window.input_indices)]
        input_normalized = (input_physical - mean) / std
        with torch.no_grad():
            normalized_prediction, _ = model(
                torch.as_tensor(input_normalized, dtype=torch.float32),
                torch.as_tensor(edge_index, dtype=torch.long),
                torch.as_tensor(edge_weight, dtype=torch.float32),
            )
        prediction = normalized_prediction.cpu().numpy() * std + mean
        previous_normalized = input_normalized[-1]
        smoothed_normalized = graph_smoothing_prediction(previous_normalized, adjacency, alpha)
        smoothing_physical = smoothed_normalized * std + mean
        target_index = window.target_index
        gnn_predictions.append(prediction)
        persistence_predictions.append(pilot.tensor[target_index - 1])
        smoothing_predictions.append(smoothing_physical)
        actuals.append(pilot.tensor[target_index])

    gnn_array = np.stack(gnn_predictions)
    persistence_array = np.stack(persistence_predictions)
    smoothing_array = np.stack(smoothing_predictions)
    actual_array = np.stack(actuals)
    expected_shape = (len(windows), 66, 5)
    for name, values in (
        ("GNN prediction", gnn_array), ("persistence prediction", persistence_array),
        ("smoothing prediction", smoothing_array), ("target", actual_array),
    ):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite with shape {expected_shape}; got {values.shape}")

    metric_rows = []
    for feature_index, (feature_name, unit) in enumerate(zip(pilot.feature_names, pilot.feature_units)):
        gnn_rmse, gnn_mae = metric_values(gnn_array[..., feature_index], actual_array[..., feature_index])
        persistence_rmse, persistence_mae = metric_values(
            persistence_array[..., feature_index], actual_array[..., feature_index]
        )
        smoothing_rmse, smoothing_mae = metric_values(
            smoothing_array[..., feature_index], actual_array[..., feature_index]
        )
        metric_rows.append({
            "feature": feature_name,
            "unit": unit,
            "prediction_count": int(len(windows) * len(pilot.nodes)),
            "gnn_rmse": gnn_rmse,
            "gnn_mae": gnn_mae,
            "persistence_rmse": persistence_rmse,
            "persistence_mae": persistence_mae,
            "smoothing_rmse": smoothing_rmse,
            "smoothing_mae": smoothing_mae,
        })

    pilot_mean = pilot.tensor.mean(axis=(0, 1), dtype=np.float64)
    pilot_std = pilot.tensor.std(axis=(0, 1), dtype=np.float64)
    pilot_min = pilot.tensor.min(axis=(0, 1))
    pilot_max = pilot.tensor.max(axis=(0, 1))
    diagnostics = [
        {
            "feature": name,
            "unit": unit,
            "checkpoint_training_mean": float(mean[index]),
            "checkpoint_training_std": float(std[index]),
            "pilot_mean": float(pilot_mean[index]),
            "pilot_std": float(pilot_std[index]),
            "pilot_std_over_checkpoint_std": float(pilot_std[index] / std[index]),
            "pilot_min": float(pilot_min[index]),
            "pilot_max": float(pilot_max[index]),
        }
        for index, (name, unit) in enumerate(zip(pilot.feature_names, pilot.feature_units))
    ]
    if not np.isfinite(np.asarray([
        value for row in diagnostics for key, value in row.items()
        if key not in {"feature", "unit"}
    ], dtype=np.float64)).all():
        raise ValueError("Domain diagnostics contain NaN or infinity")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    diagnostics_path = output_dir / "domain_diagnostics.csv"
    metadata_path = output_dir / "evaluation_metadata.json"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    first_window, last_window = windows[0], windows[-1]
    metadata_result = {
        "evaluation": "retrospective one-hour-ahead evaluation on ERA5 reanalysis",
        "checkpoint_training_note": "The frozen checkpoint was not trained on this ERA5 pilot.",
        "evaluation_design": "External frozen-checkpoint rolling one-step evaluation; no tuning.",
        "input_history_note": "Each target uses six observed ERA5 fields immediately preceding it; this is not an autoregressive rollout.",
        "pilot_source": pilot.metadata.get("source", {}),
        "pilot_period": {
            "first_valid_time": pilot.timestamps[0],
            "last_valid_time": pilot.timestamps[-1],
            "frequency": "hourly",
        },
        "evaluation_window": {
            "first_input_indices": list(first_window.input_indices),
            "first_target_index": first_window.target_index,
            "first_target_timestamp": pilot.timestamps[first_window.target_index],
            "last_input_indices": list(last_window.input_indices),
            "last_target_index": last_window.target_index,
            "last_target_timestamp": pilot.timestamps[last_window.target_index],
            "timestamp_count": len(windows),
            "node_count": len(pilot.nodes),
            "node_field_prediction_count": len(windows) * len(pilot.nodes),
            "scalar_feature_prediction_count": len(windows) * len(pilot.nodes) * len(pilot.feature_names),
            "feature_order": list(pilot.feature_names),
            "feature_units": list(pilot.feature_units),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": hash_before,
            "sha256_after": None,
            "mtime_ns_before": before_stat.st_mtime_ns,
            "mtime_ns_after": None,
            "architecture": {
                "n_features": checkpoint["n_features"],
                "hidden_dim": checkpoint["hidden_dim"],
                "embed_dim": checkpoint["embed_dim"],
                "n_processor_steps": checkpoint["n_processor_steps"],
                "history_steps": checkpoint["t_in"],
                "trained_epoch": checkpoint.get("trained_epoch"),
            },
        },
        "baseline": {
            "persistence": "last observed physical ERA5 field",
            "graph_spatial_smoothing": "weights/gnn/03_baseline_predictor.py formula with its fixed ALPHA and normalized adjacency",
            "graph_smoothing_alpha": alpha,
            "metric_units": "physical feature units",
        },
        "outputs": {
            "metrics_csv": metrics_path.name,
            "domain_diagnostics_csv": diagnostics_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata_result, indent=2, allow_nan=False), encoding="utf-8")

    after_stat = checkpoint_path.stat()
    hash_after = _checkpoint_hash(checkpoint_path)
    metadata_result["checkpoint"]["sha256_after"] = hash_after
    metadata_result["checkpoint"]["mtime_ns_after"] = after_stat.st_mtime_ns
    if hash_before != hash_after or before_stat.st_mtime_ns != after_stat.st_mtime_ns:
        raise RuntimeError("Checkpoint content or modification time changed during evaluation")
    metadata_path.write_text(json.dumps(metadata_result, indent=2, allow_nan=False), encoding="utf-8")

    return {
        "metadata": metadata_result,
        "metrics": metric_rows,
        "domain_diagnostics": diagnostics,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "era5_pilot_evaluation")
    args = parser.parse_args()
    result = evaluate(args.pilot_dir, args.checkpoint, args.output_dir)
    print(result["metadata"]["evaluation"])
    print(
        "Targets: "
        f"{result['metadata']['evaluation_window']['timestamp_count']} "
        f"({result['metadata']['evaluation_window']['first_target_timestamp']} through "
        f"{result['metadata']['evaluation_window']['last_target_timestamp']})"
    )
    for row in result["metrics"]:
        print(
            f"{row['feature']} ({row['unit']}): "
            f"GNN RMSE={row['gnn_rmse']:.5g}, MAE={row['gnn_mae']:.5g}; "
            f"persistence RMSE={row['persistence_rmse']:.5g}, MAE={row['persistence_mae']:.5g}; "
            f"smoothing RMSE={row['smoothing_rmse']:.5g}, MAE={row['smoothing_mae']:.5g}"
        )
    print(f"Checkpoint SHA-256 unchanged: {result['metadata']['checkpoint']['sha256_before']}")
    print(f"Wrote compact results to {result['output_dir']}")


if __name__ == "__main__":
    main()
