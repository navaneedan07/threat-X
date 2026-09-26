"""
VARNIKA — GNN & Spatio-Temporal Modelling
Stage 1: Graph / mesh construction

What this does
--------------
Reads the shared grid dataset and builds the spatial graph that every later
stage (GNN prototype, tracker interface, precursor features) will reuse.

Nodes   = each unique (latitude, longitude) grid cell -> 66 nodes here
Edges   = rook adjacency on the lat/lon grid (a node is connected to its
          immediate north/south/east/west neighbour on the grid). This is a
          deliberately simple, defensible mesh — NOT yet the multi-scale
          icosahedral mesh used in GraphCast-style models (Lam et al.,
          "Learning skillful medium-range global weather forecasting",
          Science, 2023). Documenting this honestly matters for the
          "research-backed feasibility" slide: we start from a plain grid
          mesh and note the multi-scale mesh as a future upgrade, not a
          claim we have already implemented it.

Outputs (saved to outputs/)
----------------------------
- node_table.csv       node_id, latitude, longitude
- edge_index.npy       (2, E) array in PyTorch-Geometric COO format
- edge_weight.npy      (E,)   inverse-distance edge weights
- mesh_diagram.png     the evidence plot for Slide 3 / Slide 6 (GNN mesh diagram)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/mnt/user-data/uploads/dataset.csv"
OUT_DIR = "/home/claude/varnika_gnn/outputs"


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique grid cell, sorted so node_id is deterministic."""
    nodes = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )
    nodes["node_id"] = nodes.index
    return nodes[["node_id", "latitude", "longitude"]]


def build_edges(nodes: pd.DataFrame, lat_step: float = 0.5, lon_step: float = 1.0):
    """
    Rook adjacency: connect a node to the neighbour one lat_step north/south
    and one lon_step east/west, if that neighbour exists in the grid.
    Returns edge_index (2, E) undirected (both directions included, as PyG
    expects) and edge_weight (E,) = 1 / euclidean distance.
    """
    coord_to_id = {
        (round(r.latitude, 4), round(r.longitude, 4)): r.node_id
        for r in nodes.itertuples()
    }

    src, dst, w = [], [], []
    for r in nodes.itertuples():
        lat, lon, nid = r.latitude, r.longitude, r.node_id
        neighbours = [
            (round(lat + lat_step, 4), round(lon, 4)),
            (round(lat - lat_step, 4), round(lon, 4)),
            (round(lat, 4), round(lon + lon_step, 4)),
            (round(lat, 4), round(lon - lon_step, 4)),
        ]
        for nb in neighbours:
            if nb in coord_to_id:
                nb_id = coord_to_id[nb]
                dist = np.hypot(lat - nb[0], lon - nb[1])
                src.append(nid)
                dst.append(nb_id)
                w.append(1.0 / dist)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(w, dtype=np.float32)
    return edge_index, edge_weight


def plot_mesh(nodes: pd.DataFrame, edge_index: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for s, d in edge_index.T:
        lat_s, lon_s = nodes.loc[s, ["latitude", "longitude"]]
        lat_d, lon_d = nodes.loc[d, ["latitude", "longitude"]]
        ax.plot([lon_s, lon_d], [lat_s, lat_d], color="#8fb3d9", linewidth=1, zorder=1)
    ax.scatter(nodes["longitude"], nodes["latitude"], s=80, color="#1f4e8c", zorder=2)
    for r in nodes.itertuples():
        ax.annotate(str(r.node_id), (r.longitude, r.latitude), fontsize=6,
                    color="white", ha="center", va="center", zorder=3)
    ax.set_title("Spatial mesh graph — 11x6 grid, 66 nodes, rook adjacency")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    nodes = build_node_table(df)
    edge_index, edge_weight = build_edges(nodes)

    nodes.to_csv(f"{OUT_DIR}/node_table.csv", index=False)
    np.save(f"{OUT_DIR}/edge_index.npy", edge_index)
    np.save(f"{OUT_DIR}/edge_weight.npy", edge_weight)
    plot_mesh(nodes, edge_index, f"{OUT_DIR}/mesh_diagram.png")

    print(f"Nodes: {len(nodes)}")
    print(f"Directed edges (both directions counted): {edge_index.shape[1]}")
    print(f"Saved node_table.csv, edge_index.npy, edge_weight.npy, mesh_diagram.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
