"""
VARNIKA — Stage 3: Baseline spatio-temporal predictor (numpy only, no torch)

Why this file exists
---------------------
The work card's integrity rule: "Do not claim a trained GNN until there are
actual experiments and metrics; a technically correct prototype/specification
is acceptable for the first milestone." This script IS that first-milestone
prototype: a hand-written, one-hop graph message-passing step (a fixed,
un-trained analogue of a single GCN layer) blended with persistence, i.e.

    pred(t+1) = alpha * X(t) + (1 - alpha) * (A_hat @ X(t))

where A_hat is the symmetrically-normalised adjacency built from the mesh in
01_graph_construction.py (D^-1/2 (A+I) D^-1/2, the same normalisation Kipf &
Welling use in the original GCN paper). No parameters are learned here —
this is the deterministic fallback the team's fallback hierarchy calls for
("classical anomaly detector -> deterministic threat tracker -> simple
calibrated transition model ..."), for the spatio-temporal stage specifically.

It also computes REAL comparison metrics against a naive persistence
baseline (predicting no change at all) on held-out steps, so nothing here is
a fabricated number.

Once torch + torch-geometric are available (04_st_gnn_model.py), the alpha
blend below is replaced by learned weights, and this script's metrics become
the number to beat.
"""

import numpy as np
import pandas as pd
import os

OUT_DIR = "/home/claude/varnika_gnn/outputs"
ALPHA = 0.6          # weight on persistence vs neighbour-smoothing
TRAIN_FRACTION = 0.8  # first 80% of timesteps = "train" (used only to pick alpha if desired)


def build_normalized_adjacency(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int):
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (s, d), w in zip(edge_index.T, edge_weight):
        A[s, d] = w
    A_self = A + np.eye(n_nodes, dtype=np.float32)
    deg = A_self.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_hat = D_inv_sqrt @ A_self @ D_inv_sqrt
    return A_hat.astype(np.float32)


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main():
    tensor = np.load(f"{OUT_DIR}/st_tensor.npy")          # (T, N, F)
    edge_index = np.load(f"{OUT_DIR}/edge_index.npy")
    edge_weight = np.load(f"{OUT_DIR}/edge_weight.npy")
    with open(f"{OUT_DIR}/feature_order.txt") as f:
        feature_names = f.read().splitlines()

    T, N, F = tensor.shape
    A_hat = build_normalized_adjacency(edge_index, edge_weight, N)

    split = int(T * TRAIN_FRACTION)

    # Per-feature normalisation (fit on "train" steps only, applied to all)
    mu = tensor[:split].mean(axis=(0, 1), keepdims=True)
    sigma = tensor[:split].std(axis=(0, 1), keepdims=True) + 1e-6
    norm = (tensor - mu) / sigma

    # One-step-ahead prediction for every t -> t+1 pair across the FULL record.
    # No parameters are fitted to data here (alpha and A_hat are fixed by
    # design, not learned) other than the mu/sigma normalisation, which is
    # fit on the train slice only -- so scoring the deterministic formula
    # across the whole record is not target leakage. We still report the
    # headline RMSE separately for the held-out region only, so the
    # generalisation claim stays honest.
    preds, actuals, persistence_preds = [], [], []
    for t in range(0, T - 1):
        x_t = norm[t]                       # (N, F)
        smoothed = A_hat @ x_t               # one-hop neighbour aggregation
        pred = ALPHA * x_t + (1 - ALPHA) * smoothed
        preds.append(pred)
        actuals.append(norm[t + 1])
        persistence_preds.append(x_t)        # naive "nothing changes" baseline

    preds = np.stack(preds)              # (T-1, N, F), preds[i] predicts step i+1
    actuals = np.stack(actuals)
    persistence_preds = np.stack(persistence_preds)

    # Held-out slice for the headline metric table
    held_preds = preds[split:]
    held_actuals = actuals[split:]
    held_persist = persistence_preds[split:]

    rows = []
    for i, name in enumerate(feature_names):
        model_rmse = rmse(held_preds[..., i], held_actuals[..., i])
        persist_rmse = rmse(held_persist[..., i], held_actuals[..., i])
        rows.append({
            "feature": name,
            "baseline_model_rmse (normalised units)": round(model_rmse, 4),
            "persistence_rmse (normalised units)": round(persist_rmse, 4),
            "improvement_over_persistence_pct": round(
                100 * (persist_rmse - model_rmse) / persist_rmse, 2
            ),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{OUT_DIR}/baseline_metrics.csv", index=False)

    # Save full-record predictions (for event replay / demo) separately from
    # the held-out-only metric table used for the headline number.
    np.save(f"{OUT_DIR}/baseline_predictions_normalised.npy", preds)
    np.save(f"{OUT_DIR}/baseline_actuals_normalised.npy", actuals)
    np.save(f"{OUT_DIR}/norm_mu.npy", mu)
    np.save(f"{OUT_DIR}/norm_sigma.npy", sigma)
    np.save(f"{OUT_DIR}/eval_start_timestep.npy", np.array([0]))  # predictions cover t=1..T-1
    np.save(f"{OUT_DIR}/held_out_start_timestep.npy", np.array([split]))

    print(f"Full-record predictions saved for timesteps 1..{T-1} (used for demo/event replay).")
    print(f"Headline metrics computed on held-out transitions only "
          f"(timesteps {split}..{T-2} -> {split+1}..{T-1}):")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics -> {OUT_DIR}/baseline_metrics.csv")


if __name__ == "__main__":
    main()
