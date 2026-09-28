"""
VARNIKA — Stage 2: Spatio-temporal tensor construction

Turns the flat CSV (one row per grid-cell-per-hour) into a proper
(T, N, F) tensor: T timesteps x N=66 nodes x F=5 features, aligned to the
node ordering fixed in 01_graph_construction.py. This tensor is the shared
input format for both the numpy baseline (03) and the PyTorch-Geometric GNN
(04, run on a GPU machine).

Note on timestamps: the Date/Time columns in the source file mix
DD/MM and MM/DD-looking values and are not reliably parseable into one
calendar. Rather than guess, we treat the file's native row order as the
true time axis (it comes in complete, ordered blocks of 66 rows = one grid
snapshot each), and index timesteps 0..T-1. This is documented so nobody
downstream mistakes the timestep index for a verified wall-clock time.
"""

import pandas as pd
import numpy as np
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DATA_PATH = REPO_ROOT / "dataset.csv"
OUT_DIR = BASE_DIR / "outputs"
FEATURES = ["temperature", "pressure", "humidity", "wind_speed", "precipitation"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    nodes = pd.read_csv(f"{OUT_DIR}/node_table.csv")
    coord_to_id = {
        (round(r.latitude, 4), round(r.longitude, 4)): int(r.node_id)
        for r in nodes.itertuples()
    }
    n_nodes = len(nodes)

    # keep only complete 66-row blocks
    n_complete_steps = len(df) // n_nodes
    df = df.iloc[: n_complete_steps * n_nodes].copy()

    df["node_id"] = df.apply(
        lambda r: coord_to_id[(round(r["latitude"], 4), round(r["longitude"], 4))],
        axis=1,
    )
    df["timestep"] = np.arange(len(df)) // n_nodes

    tensor = np.zeros((n_complete_steps, n_nodes, len(FEATURES)), dtype=np.float32)
    for _, row in df.iterrows():
        t, n = int(row["timestep"]), int(row["node_id"])
        tensor[t, n, :] = row[FEATURES].values

    np.save(f"{OUT_DIR}/st_tensor.npy", tensor)
    with open(f"{OUT_DIR}/feature_order.txt", "w") as f:
        f.write("\n".join(FEATURES))

    print(f"Built spatio-temporal tensor: T={tensor.shape[0]} steps, "
          f"N={tensor.shape[1]} nodes, F={tensor.shape[2]} features")
    print(f"Saved -> {OUT_DIR}/st_tensor.npy")


if __name__ == "__main__":
    main()
