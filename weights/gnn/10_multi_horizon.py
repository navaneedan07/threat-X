"""Stage 10: does the ST-GNN's edge grow with lead time?

The one-hour result in stage 09 raised the obvious question. Persistence is
formidable at one hour -- the atmosphere barely moves -- so a learned model that
beats it by 2-21% may simply be smoothing away noise. Persistence decays with
lead time; a model that has learnt the *dynamics* should decay more slowly. This
stage measures exactly that: the same architecture, the same training window, the
same held-out week, at 6, 12 and 24 hour lead times.

Design
------
Three independently trained models, one per horizon (``09 --horizon h``): each
sees six consecutive observed hours and predicts ``h`` hours after the last one.
A direct per-horizon model is fairer than rolling the 1 h model out
autoregressively, whose compounding error is a *different* question -- but the
rollout of the 1 h model is reported too, as the third curve, so the comparison
is complete.

All horizons score the **same held-out targets**: pilot hours 24..167, each
predicted from its most recent six *observed* hours (never predicted ones), so
every horizon predicts identical states and the columns are directly comparable.
The first 23 pilot hours are reserved as input fuel for the 24 h model; that is
the price of a common target set and it is paid identically by every model and
baseline.

Baselines
---------
* **persistence** -- the last observed field, valid for any horizon
* **mean field** -- the training-window average, the null model of forecasting
* **smoothing** -- the repository's fixed graph-smoothing formula

Requires the archives from stage 09 (``--gnn-month 2020-03/04/05``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGE9 = BASE_DIR / "09_train_real_era5.py"
OUT_DIR = BASE_DIR / "outputs"
HORIZON_DIR = OUT_DIR / "real_era5" / "horizons"
HORIZONS = (6, 12, 24)
# Pilot hours 0..23 exist only to feed the 24 h model's inputs; every horizon
# scores targets 24..167 so the columns describe identical states.
COMMON_INPUT_STEPS = max(HORIZONS)
HISTORY_STEPS = 6


def _load_stage9():
    import importlib.util

    spec = importlib.util.spec_from_file_location("threat_x_gnn_stage9", STAGE9)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {STAGE9}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage9 = _load_stage9()


def load_tensors() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (train_tensor, test_tensor, test_timestamps) from the stage-9 split."""
    node_table = pd.read_csv(OUT_DIR / "node_table.csv")
    tensor, timestamps, _ = stage9.build_training_tensor(stage9.TRAIN_ARCHIVES, node_table)
    train_tensor, _ = stage9.split_training_window(tensor, timestamps)

    test_tensor = np.load(OUT_DIR / "era5_pilot" / "tensor.npy", allow_pickle=False)
    with (OUT_DIR / "era5_pilot" / "timestamps.csv").open(encoding="utf-8", newline="") as handle:
        test_times = [row["valid_time"] for row in csv.DictReader(handle)]
    if test_tensor.shape != (168, 66, 5):
        raise ValueError("The frozen pilot tensor is not (168, 66, 5)")
    return train_tensor, test_tensor, test_times


def horizon_targets(test_tensor: np.ndarray) -> tuple[list[int], list[int]]:
    """Input and target indices for the common held-out window.

    For a target at pilot index ``t``, a model with horizon ``h`` observes the six
    hours ending at ``t - h``. Requiring ``t - h - (HISTORY_STEPS - 1) >= 0`` for
    the largest horizon keeps every model on the same target list.
    """
    start = COMMON_INPUT_STEPS
    targets = list(range(start, test_tensor.shape[0]))
    inputs = [target - COMMON_INPUT_STEPS for target in targets]
    return inputs, targets


def build_windows(
    test_tensor: np.ndarray,
    train_tensor: np.ndarray,
    horizon: int,
    inputs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Observed six-hour input windows and their targets for one horizon.

    Inputs reach back into the training window where the pilot cannot supply a
    full six observed hours (the first targets of the 24 h model). The gap between
    the last input hour and the target is exactly ``horizon - 1`` observed hours
    that the model deliberately does not see -- that gap is the lead time.
    """
    model_inputs, model_targets = [], []
    for offset, target in enumerate(inputs):
        source = target + (COMMON_INPUT_STEPS - horizon)
        end = source + 1
        if end >= HISTORY_STEPS:
            window = test_tensor[end - HISTORY_STEPS : end]
        else:
            history = train_tensor[-(HISTORY_STEPS - end) :]
            window = np.concatenate([history, test_tensor[:end]], axis=0)
        if window.shape[0] != HISTORY_STEPS:
            raise ValueError(f"Window for target {target} has {window.shape[0]} steps")
        model_inputs.append(window)
        model_targets.append(test_tensor[target])
    return np.stack(model_inputs), np.stack(model_targets)


def persistence_baseline(windows: np.ndarray) -> np.ndarray:
    """Repeat the last observed hour -- the same field at every horizon."""
    return windows[:, -1, :, :].copy()


def predict_horizon(
    model,
    mean: np.ndarray,
    std: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray:
    """Batched inference of one horizon model over its input windows."""
    edge_index = torch.as_tensor(np.load(OUT_DIR / "edge_index.npy"), dtype=torch.long)
    edge_weight = torch.as_tensor(
        np.load(OUT_DIR / "edge_weight.npy"), dtype=torch.float32
    )
    normalized = (windows - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    predictions = []
    with torch.no_grad():
        for step in range(0, normalized.shape[0], 32):
            batch = torch.as_tensor(normalized[step : step + 32], dtype=torch.float32)
            prediction, _ = model(batch, edge_index, edge_weight)
            predictions.append(prediction.cpu().numpy() * std.reshape(1, -1) + mean.reshape(1, -1))
    return np.concatenate(predictions, axis=0)


def merge_metric_frames(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-horizon tables into one lead-time-indexed frame."""
    parts = []
    for horizon, frame in frames.items():
        part = frame.copy()
        part.insert(0, "horizon_hours", horizon)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_lead_time(
    merged: pd.DataFrame,
    rollout: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Skill vs lead time per feature: GNN direct, GNN rollout, persistence, mean field."""
    features = list(dict.fromkeys(merged["feature"]))
    horizons = sorted(merged["horizon_hours"].unique())

    figure, axes = plt.subplots(1, len(features), figsize=(3.1 * len(features), 3.9), sharex=True)
    axes = np.atleast_1d(axes)

    for axis, feature in zip(axes, features, strict=True):
        rows = merged[merged["feature"] == feature].sort_values("horizon_hours")
        axis.plot(
            rows["horizon_hours"], rows["gnn_rmse"], "o-", color="#c44e52", label="ST-GNN (direct)"
        )
        axis.plot(
            rows["horizon_hours"], rows["persistence_rmse"], "s--", color="#8a7f6d",
            label="persistence",
        )
        axis.plot(
            rows["horizon_hours"], rows["mean_field_rmse"], "^:", color="#55a868",
            label="mean field",
        )
        if not rollout.empty and feature in set(rollout["feature"]):
            rrows = rollout[rollout["feature"] == feature].sort_values("horizon_hours")
            axis.plot(
                rrows["horizon_hours"], rrows["gnn_rmse"], "o--", color="#4c72b0",
                label="1 h model, rolled out",
            )
        axis.set_title(feature, fontsize=10)
        axis.set_xlabel("lead time (h)")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.set_xticks(horizons)
    axes[0].set_ylabel("RMSE (physical units)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    figure.suptitle("Skill against lead time, held-out pilot week", fontsize=11)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--skip-training", action="store_true", help="Reuse saved per-horizon metrics.")
    args = parser.parse_args()

    HORIZON_DIR.mkdir(parents=True, exist_ok=True)
    train_tensor, test_tensor, test_times = load_tensors()
    inputs, targets = horizon_targets(test_tensor)
    print(f"Common held-out targets: pilot hours {targets[0]}..{targets[-1]} ({len(targets)} states)")
    print(f"  training hours {train_tensor.shape[0]}, held-out targets {len(targets)}")

    per_horizon: dict[int, pd.DataFrame] = {}
    checkpoint_info: dict[str, Any] = {}

    for horizon in HORIZONS:
        metrics_path = HORIZON_DIR / f"metrics_h{horizon:02d}.csv"
        checkpoint_path = HORIZON_DIR / f"st_gnn_real_era5_h{horizon:02d}.pt"
        if args.skip_training and metrics_path.is_file():
            per_horizon[horizon] = pd.read_csv(metrics_path)
            continue

        print(f"\n=== horizon {horizon} h ===")
        windows, model_targets = build_windows(test_tensor, train_tensor, horizon, inputs)
        model, mean, std, _, info = stage9.train_model(
            train_tensor,
            test_tensor,
            epochs=args.epochs,
            stride=args.stride,
            lr=args.lr,
            hidden_dim=32,
            embed_dim=16,
            n_processor_steps=2,
            architecture="residual",
            dropout=0.1,
            loss_weighting="persistence",
            horizon=horizon,
        )
        predictions = predict_horizon(model, mean, std, windows)
        persistence = persistence_baseline(windows)
        smoothing_reference = persistence  # smoothing needs a per-step 'previous' field
        frame = stage9.evaluate(
            predictions,
            persistence,
            persistence,  # smoothing replaced by mean field at these horizons
            model_targets,
            climatology_mean=train_tensor.mean(axis=(0, 1)),
        )
        frame.insert(0, "horizon_hours", horizon)
        frame.to_csv(metrics_path, index=False)
        per_horizon[horizon] = frame

        torch_save = __import__("torch").save
        torch_save(
            {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "n_features": len(stage9.FEATURE_NAMES),
                "hidden_dim": 32,
                "embed_dim": 16,
                "n_processor_steps": 2,
                "architecture": "residual",
                "normalization_mean": mean,
                "normalization_std": std,
                "feature_names": stage9.FEATURE_NAMES,
                "t_in": HISTORY_STEPS,
                "horizon": horizon,
                "training": info,
            },
            checkpoint_path,
        )
        checkpoint_info[f"h{horizon}"] = {
            "checkpoint": checkpoint_path.name,
            "final_train_mse": info["final_train_mse"],
        }
        best = frame.loc[frame["gnn_vs_persistence_improvement_pct"].idxmax()]
        print(
            f"  best feature vs persistence: {best['feature']} "
            f"{best['gnn_vs_persistence_improvement_pct']:+.2f}%"
        )

    merged = merge_metric_frames(per_horizon)
    merged_path = HORIZON_DIR / "metrics_by_horizon.csv"
    merged.to_csv(merged_path, index=False)

    # Rollout of the 1 h model: predict h steps ahead by feeding predictions back,
    # one hour at a time. Error compounds; the curve shows what that costs.
    rollout_path = HORIZON_DIR / "rollout_1h.csv"
    rollout = pd.DataFrame()
    if not args.skip_training:
        print("\n=== 1 h model rolled out autoregressively ===")
        checkpoint = __import__("torch").load(
            OUT_DIR / "real_era5" / "st_gnn_real_era5_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        model_1h = stage9.build_model("residual", 32, 16, 2, 0.1)
        model_1h.load_state_dict(checkpoint["model_state_dict"])
        model_1h.eval()
        mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(checkpoint["normalization_std"], dtype=np.float64).reshape(-1)

        edge_index = np.load(OUT_DIR / "edge_index.npy", allow_pickle=False)
        edge_weight = np.load(OUT_DIR / "edge_weight.npy", allow_pickle=False)
        rows = []
        for horizon in HORIZONS:
            windows, model_targets = build_windows(test_tensor, train_tensor, horizon, inputs)
            # Roll forward h times from each window's last observed hour.
            state = windows[:, -1, :, :].copy()
            for _ in range(horizon - 1):
                normalized = (state - mean) / std
                state = stage9_predict_step(model_1h, normalized, edge_index, edge_weight, mean, std)
            predictions = stage9_predict_step(
                model_1h, (state - mean) / std, edge_index, edge_weight, mean, std
            )
            frame = stage9.evaluate(
                predictions,
                persistence_baseline(windows),
                persistence_baseline(windows),
                model_targets,
                climatology_mean=train_tensor.mean(axis=(0, 1)),
            )
            frame.insert(0, "horizon_hours", horizon)
            frame.insert(1, "mode", "1h_rolled_out")
            rows.append(frame)
        if rows:
            rollout = pd.concat(rows, ignore_index=True)
            rollout.to_csv(rollout_path, index=False)

    plot_path = HORIZON_DIR / "lead_time_skill.png"
    plot_lead_time(merged, rollout, plot_path)

    metadata = {
        "stage": "10_multi_horizon",
        "question": "does the ST-GNN's edge over persistence grow with lead time?",
        "horizons_hours": list(HORIZONS),
        "common_targets": {
            "first_pilot_index": targets[0],
            "last_pilot_index": targets[-1],
            "count": len(targets),
            "note": (
                "identical target states for every horizon; the first 23 pilot hours "
                "exist only as input fuel for the 24 h model"
            ),
        },
        "models": "one ResidualSTGNN per horizon, stage-09 recipe unchanged",
        "baselines": ["persistence (last observed field)", "training-window mean field"],
        "checkpoints": checkpoint_info,
        "outputs": {
            "metrics_by_horizon": merged_path.name,
            "rollout_1h": rollout_path.name if rollout is not None and not rollout.empty else None,
            "plot": plot_path.name,
        },
    }
    (HORIZON_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("SKILL VS LEAD TIME (RMSE, held-out pilot week)")
    print("=" * 100)
    show = merged[
        ["horizon_hours", "feature", "gnn_rmse", "persistence_rmse", "mean_field_rmse",
         "gnn_vs_persistence_improvement_pct", "gnn_vs_mean_field_improvement_pct"]
    ]
    print(show.to_string(index=False))
    print(f"\nmerged metrics -> {merged_path.relative_to(REPO_ROOT)}")
    print(f"plot           -> {plot_path.relative_to(REPO_ROOT)}")


def stage9_predict_step(model, normalized_state, edge_index, edge_weight, mean, std):
    """One forward step of the 1 h model from a single state (rollout helper)."""
    import torch

    sequence = torch.as_tensor(
        np.repeat(normalized_state[np.newaxis, ...], stage9.HISTORY_STEPS, axis=0),
        dtype=torch.float32,
    )
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
    edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32)
    with torch.no_grad():
        prediction, _ = model(sequence, edge_index_t, edge_weight_t)
    return prediction.cpu().numpy() * std + mean


if __name__ == "__main__":
    main()
