"""
VARNIKA — Stage 4: Historical event replay + comparison plots

Finds the most extreme window in the actual dataset (highest grid-mean
precipitation) and produces the evidence plots required by the work card:
"Replay selected historical events and create comparison plots between
detected/model outputs and reference data."

Outputs
-------
- event_spatial_snapshot.png   grid heatmap of precipitation at the peak timestep
- event_timeseries.png         grid-mean precipitation & temperature around the event
- event_predicted_vs_actual.png  baseline-model prediction vs actual at the peak step
                                  (only covers the held-out region from stage 3;
                                   if the peak event falls outside that region we
                                   say so rather than silently reusing a different step)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUT_DIR = BASE_DIR / "outputs"


def main():
    tensor = np.load(f"{OUT_DIR}/st_tensor.npy")     # (T, N, F) raw units
    nodes = pd.read_csv(f"{OUT_DIR}/node_table.csv")
    with open(f"{OUT_DIR}/feature_order.txt") as f:
        feature_names = f.read().splitlines()
    precip_idx = feature_names.index("precipitation")
    temp_idx = feature_names.index("temperature")

    grid_mean_precip = tensor[:, :, precip_idx].mean(axis=1)   # (T,)
    peak_t = int(np.argmax(grid_mean_precip))
    print(f"Peak grid-mean precipitation at timestep {peak_t} "
          f"({grid_mean_precip[peak_t]:.2f} mm avg across the grid)")

    # --- Spatial snapshot at the peak step ---
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(nodes["longitude"], nodes["latitude"],
                     c=tensor[peak_t, :, precip_idx], cmap="Blues", s=400, marker="s")
    fig.colorbar(sc, ax=ax, label="Precipitation (mm)")
    ax.set_title(f"Historical extreme candidate — timestep {peak_t}\n"
                 f"(highest grid-mean precipitation in the record)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/event_spatial_snapshot.png", dpi=160)
    plt.close(fig)

    # --- Time series around the event ---
    window = 12
    lo, hi = max(0, peak_t - window), min(tensor.shape[0], peak_t + window)
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    t_axis = np.arange(lo, hi)
    ax1.plot(t_axis, grid_mean_precip[lo:hi], color="#1f4e8c", label="Grid-mean precipitation (mm)")
    ax1.axvline(peak_t, color="red", linestyle="--", label="Peak timestep")
    ax1.set_xlabel("Timestep (sequential hourly index)")
    ax1.set_ylabel("Precipitation (mm)", color="#1f4e8c")
    ax2 = ax1.twinx()
    ax2.plot(t_axis, tensor[lo:hi, :, temp_idx].mean(axis=1), color="#d9782f", label="Grid-mean temperature (°C)")
    ax2.set_ylabel("Temperature (°C)", color="#d9782f")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.set_title("Event window replay — precipitation build-up vs temperature")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/event_timeseries.png", dpi=160)
    plt.close(fig)

    # --- Predicted vs actual at the peak event ---
    # preds[i] (i = 0..T-2) is the model's prediction for timestep i+1, so the
    # prediction for peak_t lives at preds[peak_t - 1], as long as peak_t >= 1.
    preds = np.load(f"{OUT_DIR}/baseline_predictions_normalised.npy")   # (T-1, N, F)
    actuals = np.load(f"{OUT_DIR}/baseline_actuals_normalised.npy")
    mu = np.load(f"{OUT_DIR}/norm_mu.npy")
    sigma = np.load(f"{OUT_DIR}/norm_sigma.npy")
    held_out_start = int(np.load(f"{OUT_DIR}/held_out_start_timestep.npy")[0])

    pred_index = peak_t - 1
    if 0 <= pred_index < preds.shape[0]:
        pred_raw = preds[pred_index, :, precip_idx] * sigma[0, 0, precip_idx] + mu[0, 0, precip_idx]
        actual_raw = actuals[pred_index, :, precip_idx] * sigma[0, 0, precip_idx] + mu[0, 0, precip_idx]
        in_held_out = peak_t >= held_out_start + 1
        caveat = ("(this event is inside the held-out evaluation window, "
                   "so this is a genuinely out-of-sample comparison)" if in_held_out else
                   "(NOTE: this event falls inside the data used to fit the mu/sigma "
                   "normalisation, so treat this as a qualitative illustration, not a "
                   "held-out accuracy claim -- the headline RMSE table is the held-out number)")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, data, title in zip(axes, [actual_raw, pred_raw], ["Actual", "Baseline prediction"]):
            sc = ax.scatter(nodes["longitude"], nodes["latitude"], c=data, cmap="Blues",
                             s=350, marker="s", vmin=min(actual_raw.min(), pred_raw.min()),
                             vmax=max(actual_raw.max(), pred_raw.max()))
            ax.set_title(f"{title} precipitation, timestep {peak_t}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=axes, label="Precipitation (mm)", shrink=0.8)
        fig.suptitle("Baseline spatio-temporal predictor vs actual at the extreme event")
        fig.savefig(f"{OUT_DIR}/event_predicted_vs_actual.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        mae = float(np.mean(np.abs(pred_raw - actual_raw)))
        print(f"Saved predicted-vs-actual comparison. MAE at this event = {mae:.3f} mm {caveat}")
    else:
        print(f"Peak timestep {peak_t} has no valid prior step to predict from; "
              f"skipping predicted-vs-actual plot.")

    print(f"Saved plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
