"""Run the trained pilot GNN through anomaly detection and threat tracking.

From the repository root, run with ``conda run -n threat-x python
weights/gnn/run_gnn_tracking_demo.py``. This is a one-step prototype demo,
not an operational forecast or a climatology-validated anomaly product.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detection.anomaly_detection import detect_field_anomalies
from src.tracking.pipeline import run_tracking_pipeline

GNN_DIR = Path(__file__).resolve().parent
OUT_DIR = GNN_DIR / "outputs"
FEATURE_ORDER = ["temperature", "pressure", "humidity", "wind_speed", "precipitation"]


def _model_class(architecture: str = "plain"):
    """Return the architecture the checkpoint was fitted with.

    A checkpoint written by ``09_train_real_era5.py`` records its own
    ``architecture``. This matters: a *residual* checkpoint loaded into the plain
    class would apply no skip connection, so its decoder output -- an increment --
    would be returned as if it were a field. That would be wrong by hundreds of
    hPa and would fail silently, so the stored value is honoured instead.
    """
    module_path = GNN_DIR / "05_st_gnn_model.py"
    spec = importlib.util.spec_from_file_location("threat_x_st_gnn_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GNN model module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if architecture == "residual":
        return module.ResidualSTGNN
    if architecture == "plain":
        return module.SpatioTemporalGNN
    raise ValueError(f"Unknown checkpoint architecture {architecture!r}")


def predict_next_field(
    checkpoint_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    """Return physical prediction, six-step baseline, nodes, and UTC time."""
    pilot = OUT_DIR / "era5_pilot"
    tensor = np.load(pilot / "tensor.npy")
    if tensor.ndim != 3 or tensor.shape[1:] != (66, 5) or not np.isfinite(tensor).all():
        raise ValueError(f"Pilot tensor must be finite (T,66,5), got {tensor.shape}")
    with (OUT_DIR / "feature_order.txt").open(encoding="utf-8") as handle:
        feature_order = handle.read().splitlines()
    if feature_order != FEATURE_ORDER:
        raise ValueError(f"Unexpected feature order: {feature_order}")

    with (OUT_DIR / "era5_pilot" / "timestamps.csv").open(encoding="utf-8", newline="") as handle:
        timestamps = [row["valid_time"] for row in csv.DictReader(handle)]
    if len(timestamps) != tensor.shape[0]:
        raise ValueError("Pilot timestamp count does not match tensor length")

    with (OUT_DIR / "node_table.csv").open(encoding="utf-8", newline="") as handle:
        nodes = [{"node_id": int(row["node_id"]), "latitude": float(row["latitude"]),
                  "longitude": float(row["longitude"])} for row in csv.DictReader(handle)]
    if [node["node_id"] for node in nodes] != list(range(66)):
        raise ValueError("Node table must be in node_id order 0..65")

    resolved = Path(checkpoint_path) if checkpoint_path else OUT_DIR / "st_gnn_checkpoint.pt"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}\n"
            "Train the frozen checkpoint with:\n"
            "  python weights/gnn/05_st_gnn_model.py\n"
            "or the real-ERA5 one with:\n"
            "  python weights/gnn/09_train_real_era5.py"
        )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_names") != FEATURE_ORDER or checkpoint.get("t_in") != 6:
        raise ValueError("Checkpoint feature order or six-step input does not match the pilot")
    architecture = str(checkpoint.get("architecture", "plain"))
    mean = torch.as_tensor(checkpoint["normalization_mean"], dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(checkpoint["normalization_std"], dtype=torch.float32).reshape(-1)
    if mean.shape != (5,) or std.shape != (5,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Checkpoint normalization statistics are invalid")

    history = tensor[-6:]
    normalized = (torch.as_tensor(history, dtype=torch.float32) - mean) / std
    edge_index = torch.as_tensor(np.load(OUT_DIR / "edge_index.npy"), dtype=torch.long)
    edge_weight = torch.as_tensor(np.load(OUT_DIR / "edge_weight.npy"), dtype=torch.float32)
    model = _model_class(architecture)(
        n_features=checkpoint["n_features"], hidden_dim=checkpoint["hidden_dim"],
        embed_dim=checkpoint["embed_dim"], n_processor_steps=checkpoint["n_processor_steps"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction_norm, _ = model(normalized, edge_index, edge_weight)
    prediction = (prediction_norm * std + mean).cpu().numpy()
    if prediction.shape != (66, 5) or not np.isfinite(prediction).all():
        raise ValueError(f"GNN prediction must be finite (66,5), got {prediction.shape}")

    last_time = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
    next_time = (last_time + timedelta(hours=1)).astimezone(timezone.utc)
    return prediction, history, nodes, next_time.isoformat().replace("+00:00", "Z")


def run_demo(output_path: str | Path, checkpoint_path: str | Path | None = None) -> dict:
    prediction, history, nodes, timestamp = predict_next_field(checkpoint_path)
    frames = detect_field_anomalies(
        field=prediction[:, 4], baseline=history[:, :, 4], nodes=nodes,
        timestamp=timestamp,
    )
    trajectory_path = Path(output_path).with_name(Path(output_path).stem + "_trajectories.json")
    tracked = run_tracking_pipeline(
        frames, output_path=trajectory_path, default_year=2020, merge_existing=False,
    )
    payload = {
        "_meta": {
            "source": "VARNIKA pilot Spatio-Temporal GNN bridge",
            "timestamp": timestamp,
            "feature_order": FEATURE_ORDER,
            "reference": "six preceding pilot hours; prototype threshold from existing detector",
            "limitation": "prototype one-step field; not operational forecast skill or validated climatology",
        },
        "predicted_field": [
            {"node_id": node["node_id"], "latitude": node["latitude"],
             "longitude": node["longitude"],
             **{name: float(value) for name, value in zip(FEATURE_ORDER, prediction[node["node_id"]])}}
            for node in nodes
        ],
        "anomaly_frames": frames,
        "threat_ids": list(tracked["trajectories"]),
        "trajectories": tracked["trajectories"],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "gnn_tracking_demo.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help=(
            "Checkpoint to infer with. Defaults to the frozen checkpoint trained on "
            "dataset.csv; pass weights/gnn/outputs/real_era5/st_gnn_real_era5_checkpoint.pt "
            "for the model trained on real ERA5."
        ),
    )
    args = parser.parse_args()
    result = run_demo(args.output, args.checkpoint)
    print(f"Saved demo output to {args.output}")
    print(f"Anomaly boxes: {sum(len(frame['boxes']) for frame in result['anomaly_frames'])}")
    print(f"Threat IDs: {result['threat_ids']}")


if __name__ == "__main__":
    main()
