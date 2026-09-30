"""Stage 9: train and evaluate the ST-GNN on REAL ERA5 reanalysis.

Why this file exists
--------------------
``05_st_gnn_model.py`` trains on ``st_tensor.npy``, which is built from the
repository-root ``dataset.csv``. That file's timestamps are not parseable into a
calendar (see the note in ``02_build_spatiotemporal_tensor.py``), and its
distributions do not match ERA5: its hourly precipitation mean is ~100x the
reanalysis mean and its pressure spread is ~20x the reanalysis spread over the
same 66-node box. The consequence is measured, not asserted -- the frozen
checkpoint evaluation in ``08_evaluate_era5_pilot.py`` records the scale
mismatch in ``domain_diagnostics.csv`` and the GNN loses to persistence on every
feature.

This stage removes that failure mode at its source by training on real ERA5.

Design
------
::

    train   2020-04-01 00:00Z .. 2020-05-15 23:00Z   (real ERA5, hourly)
    test    2020-05-16 00:00Z .. 2020-05-22 23:00Z   (the frozen pilot week)

The test window is deliberately identical to ``outputs/era5_pilot/tensor.npy``,
the same one ``08`` evaluates on, so the two numbers are directly comparable and
the pilot stays a genuinely held-out period: no pilot hour is used to fit the
weights or the normalisation statistics.

Preprocessing, feature order, units and node order all come from
``02_prepare_era5_tensor.prepare_dataset``; the architecture comes from
``05_st_gnn_model.SpatioTemporalGNN``; the baselines come from
``03_baseline_predictor``. Nothing is re-implemented here, so this file cannot
silently drift from the definitions already in the repository.

Requires real archives (fetched with ``src.data.cds_fetch``)::

    python -m src.data.cds_fetch --gnn-month 2020-04 --gnn-month 2020-05
    python weights/gnn/09_train_real_era5.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
import torch  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_dataset  # noqa: E402

OUT_DIR = BASE_DIR / "outputs"
REAL_OUT_DIR = OUT_DIR / "real_era5"
PILOT_DIR = OUT_DIR / "era5_pilot"
NODE_TABLE_PATH = OUT_DIR / "node_table.csv"

TRAIN_ARCHIVES = (
    REPO_ROOT / "data" / "raw" / "era5_gnn_2020_03.zip",
    REPO_ROOT / "data" / "raw" / "era5_gnn_2020_04.zip",
    REPO_ROOT / "data" / "raw" / "era5_gnn_2020_05.zip",
)
TRAIN_LAST_HOUR = "2020-05-15T23:00:00Z"
TEST_FIRST_HOUR = "2020-05-16T00:00:00Z"

HISTORY_STEPS = 6


def _load_sibling(name: str, filename: str):
    """Import a sibling stage module without executing its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename} from {BASE_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_sibling("threat_x_gnn_prepare", "02_prepare_era5_tensor.py")
model_module = _load_sibling("threat_x_gnn_model", "05_st_gnn_model.py")
baseline_module = _load_sibling("threat_x_gnn_baseline", "03_baseline_predictor.py")

FEATURE_NAMES = list(prep.FEATURE_NAMES)
FEATURE_UNITS = list(prep.FEATURE_UNITS)


def build_training_tensor(
    archives: tuple[Path, ...],
    node_table: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Concatenate the validated real-ERA5 monthly tensors into one time axis."""
    tensors: list[np.ndarray] = []
    timestamps: list[str] = []
    provenance: list[dict[str, Any]] = []

    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(
                f"Missing ERA5 archive {archive}. Fetch it with:\n"
                f"  python -m src.data.cds_fetch --gnn-month "
                f"{archive.stem.split('_')[-2]}-{archive.stem.split('_')[-1]}"
            )
        dataset = load_dataset(archive)
        tensor, index, metadata = prep.prepare_dataset(dataset, node_table, str(archive))
        dataset.close()
        tensors.append(tensor)
        block = [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in index]
        timestamps.extend(block)
        provenance.append(
            {
                "archive": archive.name,
                "timesteps": int(tensor.shape[0]),
                "first_valid_time": block[0],
                "last_valid_time": block[-1],
            }
        )

    combined = np.concatenate(tensors, axis=0)
    if len(timestamps) != combined.shape[0]:
        raise ValueError("Timestamp count does not match the concatenated tensor")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Concatenated real-ERA5 time axis contains duplicates")
    return combined, timestamps, provenance


def split_training_window(
    tensor: np.ndarray, timestamps: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Keep only the pre-pilot training hours, trimmed to a whole hour boundary."""
    if TEST_FIRST_HOUR not in timestamps:
        raise ValueError(
            f"The real-ERA5 archives do not cover the pilot start {TEST_FIRST_HOUR}"
        )
    cut = timestamps.index(TEST_FIRST_HOUR)
    train_tensor = tensor[:cut]
    train_times = timestamps[:cut]
    if train_times[-1] != TRAIN_LAST_HOUR:
        raise ValueError(
            f"Training window ends at {train_times[-1]}, expected {TRAIN_LAST_HOUR}. "
            "The archives and the pilot week are not contiguous."
        )
    return train_tensor, train_times


def build_model(
    architecture: str,
    hidden_dim: int,
    embed_dim: int,
    n_processor_steps: int,
    dropout: float,
):
    """Instantiate the requested architecture from the shared stage-5 definitions."""
    if architecture == "residual":
        return model_module.ResidualSTGNN(
            n_features=len(FEATURE_NAMES),
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_processor_steps=n_processor_steps,
            dropout=dropout,
        )
    if architecture == "plain":
        return model_module.SpatioTemporalGNN(
            n_features=len(FEATURE_NAMES),
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_processor_steps=n_processor_steps,
        )
    raise ValueError(f"Unknown architecture {architecture!r}; expected plain or residual")


def train_model(
    train_tensor: np.ndarray,
    test_tensor: np.ndarray,
    epochs: int,
    stride: int,
    lr: float,
    hidden_dim: int,
    embed_dim: int,
    n_processor_steps: int,
    architecture: str = "residual",
    dropout: float = 0.1,
    loss_weighting: str = "persistence",
    horizon: int = 1,
) -> tuple[Any, dict[str, Any]]:
    """Fit the ST-GNN on the training window; the test week is never touched.

    ``horizon`` is the lead time in hours between the last input field and the
    target. 1 is the default and reproduces stage 09's original protocol exactly;
    6/12/24 train the same architecture on the same window with the target simply
    further ahead, so per-horizon skill is attributable to the lead time alone.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = torch.as_tensor(np.load(OUT_DIR / "edge_index.npy"), dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(
        np.load(OUT_DIR / "edge_weight.npy"), dtype=torch.float32, device=device
    )

    # Normalisation statistics come from the TRAINING window only. Reusing them at
    # test time is what makes the two runs comparable; recomputing them on the test
    # week would leak its distribution into inference.
    mean = train_tensor.mean(axis=(0, 1), keepdims=True)
    std = train_tensor.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train = torch.as_tensor(
        (train_tensor - mean) / std, dtype=torch.float32, device=device
    )
    targets = list(range(HISTORY_STEPS, train_tensor.shape[0] - (horizon - 1), stride))

    torch.manual_seed(42)
    np.random.seed(42)
    model = build_model(
        architecture, hidden_dim, embed_dim, n_processor_steps, dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Per-feature loss weighting. A plain MSE over the five normalized features is
    # dominated by whichever feature is least predictable: one hour ahead, the
    # normalized persistence MSE is 0.99 for precipitation but only 0.07-0.10 for
    # temperature, pressure, humidity and wind. Under a flat loss the optimizer can
    # improve precipitation a little while *giving away* the four features where
    # persistence is already nearly perfect -- and that is exactly what the
    # unweighted runs did: they lost to persistence by 290-310% on those four.
    # Weighting each feature by 1/persistence_MSE makes every feature contribute
    # equally, so the model has to earn its gains without sacrificing the easy ones.
    loss_weights = torch.ones(len(FEATURE_NAMES), dtype=torch.float32, device=device)
    if loss_weighting == "persistence":
        increments = x_train[1:] - x_train[:-1]
        per_feature = (increments**2).mean(dim=(0, 1)).clamp_min(1e-8)
        loss_weights = (1.0 / per_feature)
        loss_weights = loss_weights / loss_weights.mean()
    elif loss_weighting != "uniform":
        raise ValueError(f"Unknown loss weighting {loss_weighting!r}")

    def criterion(prediction, target):
        return ((prediction - target) ** 2 * loss_weights).mean()

    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for target in targets:
            window = x_train[target - HISTORY_STEPS : target]
            optimizer.zero_grad()
            prediction, _ = model(window, edge_index, edge_weight)
            loss = criterion(prediction, x_train[target + horizon - 1])
            loss.backward()
            optimizer.step()
            running += loss.item()
        average = running / max(len(targets), 1)
        loss_history.append(average)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | train MSE {average:.5f}")

    # History the model is allowed to see for the first test target: the last six
    # training hours. Everything after that is the held-out pilot week.
    history = train_tensor[-HISTORY_STEPS:]
    info = {
        "device": str(device),
        "architecture": architecture,
        "dropout": dropout,
        "loss_weighting": loss_weighting,
        "loss_weights": loss_weights.detach().cpu().numpy().tolist(),
        "horizon_hours": horizon,
        "normalization_mean": mean.reshape(-1).tolist(),
        "normalization_std": std.reshape(-1).tolist(),
        "epochs": epochs,
        "stride": stride,
        "learning_rate": lr,
        "training_targets": len(targets),
        "train_timesteps": int(train_tensor.shape[0]),
        "test_timesteps": int(test_tensor.shape[0]),
        "final_train_mse": loss_history[-1],
        "loss_history": loss_history,
    }
    model.eval()
    return model, mean, std, history, info


def rollout_test_week(
    model,
    mean: np.ndarray,
    std: np.ndarray,
    history: np.ndarray,
    test_tensor: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the held-out pilot week one hour ahead, using observed history.

    Each target hour is predicted from the six *observed* ERA5 fields immediately
    preceding it -- the last six training hours for the first target, then the
    pilot's own observations. This is not an autoregressive rollout, so error does
    not compound; it is the same protocol ``08_evaluate_era5_pilot.py`` uses.
    """
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
    edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32)
    alpha = float(baseline_module.ALPHA)

    # Flatten the statistics to one value per feature before applying them to a
    # single (66, 5) prediction. Left as (1, 1, 5) they broadcast a leading
    # singleton axis into the result, silently turning (66, 5) into (1, 66, 5) and
    # corrupting every saved array downstream. The stored checkpoint keeps the
    # original shape; only the arithmetic here is flattened.
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean.shape != (len(FEATURE_NAMES),) or std.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Expected one statistic per feature, got {mean.shape} / {std.shape}")

    predictions, persistence, smoothing = [], [], []
    with torch.no_grad():
        for target in range(test_tensor.shape[0]):
            if target >= HISTORY_STEPS:
                # Six observed pilot hours immediately before the target hour.
                window = test_tensor[target - HISTORY_STEPS : target]
            else:
                # The first six targets reach back into the end of the training
                # window, which keeps the input sequence contiguous in time.
                window = np.concatenate(
                    [history[target:], test_tensor[:target]], axis=0
                )
            window = window[-HISTORY_STEPS:]
            normalized = (window - mean) / std
            prediction, _ = model(
                torch.as_tensor(normalized, dtype=torch.float32), edge_index_t, edge_weight_t
            )
            predictions.append(prediction.cpu().numpy() * std + mean)

            previous_normalized = normalized[-1]
            persistence.append(window[-1])
            smoothing.append(
                (alpha * previous_normalized + (1.0 - alpha) * (adjacency @ previous_normalized))
                * std
                + mean
            )

    return np.stack(predictions), np.stack(persistence), np.stack(smoothing)


def improvement_pct(baseline_rmse: float, candidate_rmse: float) -> float:
    """Percentage error reduction of ``candidate`` over ``baseline``.

    Returns ``0.0`` when the baseline is already perfect. A perfect baseline is a
    degenerate input -- there is no error left to reduce -- and returning a ratio
    there would either divide by zero or report a meaningless infinite gain, so the
    table records "no improvement available" instead of a fabricated number.
    """
    if baseline_rmse == 0.0:
        return 0.0
    return 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse


def evaluate(
    gnn: np.ndarray,
    persistence: np.ndarray,
    smoothing: np.ndarray,
    actual: np.ndarray,
    climatology_mean: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-feature RMSE/MAE in physical units, plus the improvement percentages.

    ``climatology_mean`` is the *training-window* mean field, added for the
    multi-horizon stage: at 24 h the mean field is itself a baseline ("climatology"
    at this short a window), and reporting the GNN against it stops a model that
    has merely learnt the average from looking skilful. When it is omitted the
    ``mean_field_*`` columns are simply left out, so the 1 h callers are unchanged.
    """
    rows = []
    for index, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)):
        gnn_error = gnn[..., index] - actual[..., index]
        persist_error = persistence[..., index] - actual[..., index]
        smooth_error = smoothing[..., index] - actual[..., index]

        gnn_rmse = float(np.sqrt(np.mean(gnn_error**2)))
        persist_rmse = float(np.sqrt(np.mean(persist_error**2)))
        smooth_rmse = float(np.sqrt(np.mean(smooth_error**2)))

        rows.append(
            {
                "feature": name,
                "unit": unit,
                "prediction_count": int(actual.shape[0] * actual.shape[1]),
                "gnn_rmse": round(gnn_rmse, 5),
                "gnn_mae": round(float(np.mean(np.abs(gnn_error))), 5),
                "persistence_rmse": round(persist_rmse, 5),
                "persistence_mae": round(float(np.mean(np.abs(persist_error))), 5),
                "smoothing_rmse": round(smooth_rmse, 5),
                "smoothing_mae": round(float(np.mean(np.abs(smooth_error))), 5),
                "gnn_vs_persistence_improvement_pct": round(
                    improvement_pct(persist_rmse, gnn_rmse), 2
                ),
                "gnn_vs_smoothing_improvement_pct": round(
                    improvement_pct(smooth_rmse, gnn_rmse), 2
                ),
            }
        )
        if climatology_mean is not None:
            reference = np.broadcast_to(
                np.asarray(climatology_mean, dtype=np.float64).reshape(-1)[index],
                actual[..., index].shape,
            )
            mean_rmse = float(np.sqrt(np.mean((reference - actual[..., index]) ** 2)))
            rows[-1]["mean_field_rmse"] = round(mean_rmse, 5)
            rows[-1]["gnn_vs_mean_field_improvement_pct"] = round(
                improvement_pct(mean_rmse, gnn_rmse), 2
            )
    return pd.DataFrame(rows)


def _attach(frame: pd.DataFrame, column: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        return frame
    reference = pd.read_csv(path).set_index("feature")["gnn_rmse"]
    frame = frame.copy()
    frame[column] = frame["feature"].map(reference)
    return frame


def add_reference_columns(frame: pd.DataFrame, architecture: str) -> pd.DataFrame:
    """Attach the other runs' RMSE on this same week, so the progression is visible.

    Three numbers end up side by side for an identical held-out week:

    * ``frozen_checkpoint_gnn_rmse`` -- fitted on ``dataset.csv`` by stage 5, scored
      by ``08_evaluate_era5_pilot.py``
    * ``plain_real_era5_gnn_rmse`` -- the plain decoder trained on real ERA5
    * ``gnn_rmse`` -- this run
    """
    frame = _attach(
        frame, "frozen_checkpoint_gnn_rmse", OUT_DIR / "era5_pilot_evaluation" / "metrics.csv"
    )
    if architecture != "plain":
        frame = _attach(
            frame, "plain_real_era5_gnn_rmse", REAL_OUT_DIR / "real_era5_metrics_plain.csv"
        )
    return frame


def plot_overview(frame: pd.DataFrame, out_path: Path) -> Path:
    """Baseline-vs-GNN RMSE per feature, plus the improvement summary."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})

    positions = np.arange(len(frame))
    width = 0.27
    left.bar(positions - width, frame["persistence_rmse"], width, label="persistence", color="#8a7f6d")
    left.bar(positions, frame["smoothing_rmse"], width, label="graph smoothing", color="#4c72b0")
    left.bar(positions + width, frame["gnn_rmse"], width, label="ST-GNN (real ERA5)", color="#c44e52")
    left.set_xticks(positions, frame["feature"], rotation=15)
    left.set_ylabel("RMSE (physical units)")
    left.set_title("Held-out pilot week 2020-05-16..22: RMSE by feature")
    left.grid(True, axis="y", linestyle=":", alpha=0.6)
    left.legend()

    right.axis("off")
    text = "\n".join(
        f"{row.feature:14s} {row.gnn_vs_persistence_improvement_pct:+6.2f}% vs persistence"
        for row in frame.itertuples()
    )
    right.text(
        0.0,
        1.0,
        "Improvement of the real-ERA5 ST-GNN\nover persistence, held-out pilot week\n\n"
        + text,
        va="top",
        family="monospace",
        fontsize=9,
    )

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2, help="Target subsampling stride.")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--architecture",
        choices=("residual", "plain"),
        default="residual",
        help=(
            "residual predicts the one-hour increment on top of the last observed "
            "state (starts at persistence); plain predicts the absolute field."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--loss-weighting",
        choices=("persistence", "uniform"),
        default="persistence",
        help=(
            "persistence weights each feature by 1/its persistence MSE so the "
            "least predictable feature cannot dominate the loss."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--processor-steps", type=int, default=2)
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=(1, 6, 12, 24),
        help=(
            "Lead time in hours between the last input field and the target. "
            "Stage 10 trains one model per horizon and compares them."
        ),
    )
    args = parser.parse_args()

    REAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    node_table = pd.read_csv(NODE_TABLE_PATH)

    print("Building the real-ERA5 training tensor ...")
    tensor, timestamps, provenance = build_training_tensor(TRAIN_ARCHIVES, node_table)
    train_tensor, train_times = split_training_window(tensor, timestamps)
    print(f"  training hours : {train_tensor.shape[0]}  ({train_times[0]} .. {train_times[-1]})")

    test_tensor = np.load(PILOT_DIR / "tensor.npy", allow_pickle=False)
    with (PILOT_DIR / "timestamps.csv").open(encoding="utf-8", newline="") as handle:
        test_times = [row["valid_time"] for row in csv.DictReader(handle)]
    if test_tensor.shape != (168, 66, 5) or len(test_times) != 168:
        raise ValueError("The frozen pilot tensor is not (168, 66, 5)")
    print(f"  held-out hours : {test_tensor.shape[0]}  ({test_times[0]} .. {test_times[-1]})")

    model, mean, std, history, info = train_model(
        train_tensor,
        test_tensor,
        epochs=args.epochs,
        stride=args.stride,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_processor_steps=args.processor_steps,
        architecture=args.architecture,
        dropout=args.dropout,
        loss_weighting=args.loss_weighting,
        horizon=args.horizon,
    )

    edge_index = np.load(OUT_DIR / "edge_index.npy", allow_pickle=False)
    edge_weight = np.load(OUT_DIR / "edge_weight.npy", allow_pickle=False)
    adjacency = baseline_module.build_normalized_adjacency(edge_index, edge_weight, 66)

    gnn, persistence, smoothing = rollout_test_week(
        model, mean, std, history, test_tensor, edge_index, edge_weight, adjacency
    )

    # Score from the first hour that has a full six-hour observed history, so the
    # feature set, the target count (162) and the window indices are byte-for-byte
    # the ones 08_evaluate_era5_pilot.py uses. That is what makes the two runs
    # comparable rather than merely similar.
    scored = slice(HISTORY_STEPS, None)
    frame = evaluate(
        gnn[scored], persistence[scored], smoothing[scored], test_tensor[scored]
    )

    checkpoint_path = REAL_OUT_DIR / (
        "st_gnn_real_era5_checkpoint.pt"
        if args.horizon == 1
        else f"st_gnn_real_era5_h{args.horizon:02d}.pt"
    )
    torch.save(
        {
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "n_features": len(FEATURE_NAMES),
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "n_processor_steps": args.processor_steps,
            # Recorded so consumers load this state dict into the matching class.
            # A residual checkpoint evaluated with the plain forward pass would
            # silently return an increment where a field is expected.
            "architecture": args.architecture,
            "normalization_mean": mean,
            "normalization_std": std,
            "feature_names": FEATURE_NAMES,
            "t_in": HISTORY_STEPS,
            "training": info,
        },
        checkpoint_path,
    )

    suffix = "" if args.horizon == 1 else f"_h{args.horizon:02d}"
    metrics_path = REAL_OUT_DIR / f"real_era5_metrics{suffix}.csv"
    per_architecture_path = REAL_OUT_DIR / f"real_era5_metrics_{args.architecture}{suffix}.csv"

    expected_test_shape = (test_tensor.shape[0], 66, len(FEATURE_NAMES))
    for name, values in (
        ("gnn", gnn),
        ("persistence", persistence),
        ("smoothing", smoothing),
    ):
        if values.shape != expected_test_shape:
            raise ValueError(f"{name} must be {expected_test_shape}; got {values.shape}")

    np.savez_compressed(
        REAL_OUT_DIR / f"real_era5_predictions{suffix}.npz",
        gnn=gnn.astype(np.float32),
        persistence=persistence.astype(np.float32),
        smoothing=smoothing.astype(np.float32),
        actual=test_tensor.astype(np.float32),
        timestamps=np.array(test_times),
        features=np.array(FEATURE_NAMES),
        units=np.array(FEATURE_UNITS),
    )

    frame.insert(2, "architecture", args.architecture)
    frame.to_csv(per_architecture_path, index=False)
    frame = add_reference_columns(frame, args.architecture)
    frame.to_csv(metrics_path, index=False)

    plot_path = plot_overview(frame, REAL_OUT_DIR / f"real_era5_overview{suffix}.png")

    metadata_path = REAL_OUT_DIR / f"real_era5_metadata{suffix}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "stage": "09_train_real_era5",
                "source": "ERA5 reanalysis, reanalysis-era5-single-levels",
                "training_archives": provenance,
                "normalisation": "statistics from the training window only; no leakage",
                "protocol": (
                    "one-hour-ahead prediction from six observed ERA5 fields; "
                    "not an autoregressive rollout"
                ),
                "held_out_window": {
                    "first_valid_time": test_times[0],
                    "last_valid_time": test_times[-1],
                    "timesteps": len(test_times),
                    "identical_to": "weights/gnn/outputs/era5_pilot (the frozen pilot)",
                    "lead_time_hours": args.horizon,
                },
                "architecture": {
                    "name": args.architecture,
                    "hidden_dim": args.hidden_dim,
                    "embed_dim": args.embed_dim,
                    "n_processor_steps": args.processor_steps,
                    "dropout": args.dropout,
                    "loss_weighting": args.loss_weighting,
                    "history_steps": HISTORY_STEPS,
                },
                "comparison": (
                    "frozen_checkpoint_gnn_rmse is the checkpoint fitted on dataset.csv "
                    "and scored on this same week by 08_evaluate_era5_pilot.py; "
                    "plain_real_era5_gnn_rmse is the non-residual decoder trained on "
                    "real ERA5 and scored on this same week"
                ),
                "metrics": frame.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("REAL ERA5 -- HELD-OUT PILOT WEEK, ONE-HOUR-AHEAD")
    print("=" * 88)
    print(frame.to_string(index=False))
    print()
    print(f"checkpoint -> {checkpoint_path.relative_to(REPO_ROOT)}")
    print(f"metrics    -> {metrics_path.relative_to(REPO_ROOT)}")
    print(f"overview   -> {plot_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
