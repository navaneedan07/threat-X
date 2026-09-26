"""
VARNIKA — Stage 5: Spatio-temporal GNN prototype (PyTorch + PyTorch Geometric)

*** This file needs an environment with torch and torch-geometric. ***
It could not be executed in the container used to prepare this handoff
(no network access there to install torch), so run it on your own
GPU machine / Colab:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torch-geometric

Design
------
A small encoder-processor-decoder GNN, the same lineage as GraphCast (Lam et
al., "Learning skillful medium-range global weather forecasting", Science,
2023) and MeshGraphNets (Pfaff et al., "Learning Mesh-Based Simulation with
Graph Networks", ICLR 2021), scaled down to fit a 66-node regional mesh:

  1. Encoder:   per-node MLP lifts the raw 5 features into a hidden embedding
  2. Processor: K rounds of GCNConv (spatial message passing over the mesh
                built in 01_graph_construction.py) + a GRUCell applied
                per-node across the input time window (temporal recurrence,
                in the spirit of DCRNN / Graph WaveNet-style spatio-temporal
                architectures)
  3. Decoder:   per-node MLP maps the final hidden state to a predicted
                next-step feature vector AND to a fixed-size "threat
                embedding" -- this embedding is what gets handed to
                Sachin's tracker and Hariharan's precursor stage (see
                06_export_interface.py for the JSON contract).

This is a PROTOTYPE / SPECIFICATION. Per the work card's integrity rule, do
not present this as "trained" until it has actually been trained and
evaluated with real held-out metrics -- swap in real numbers in
outputs/baseline_metrics.csv-style table once you've run it.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from torch_geometric.nn import GCNConv
except ImportError as e:
    raise ImportError(
        "torch_geometric is required for this file. Install with:\n"
        "  pip install torch-geometric\n"
        "This script is meant to run on your own GPU machine, not in the "
        "data-prep container."
    ) from e


class STGNNCell(nn.Module):
    """One round of spatial message passing + per-node temporal recurrence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, h, edge_index, edge_weight):
        # x: (N, hidden_dim) current-step embedding
        # h: (N, hidden_dim) recurrent state from previous step
        msg = torch.relu(self.gcn(x, edge_index, edge_weight))
        h_new = self.gru(msg, h)
        return h_new


class SpatioTemporalGNN(nn.Module):
    """
    Encoder-processor-decoder ST-GNN.

    Inputs:
        x_seq: (T_in, N, F)  a window of T_in past timesteps
        edge_index, edge_weight: from 01_graph_construction.py

    Outputs:
        next_step_pred: (N, F)          predicted features at t = T_in
        threat_embedding: (N, embed_dim) per-node embedding for downstream
                                          Threat Object / tracker / precursor use
    """

    def __init__(self, n_features: int, hidden_dim: int = 32, embed_dim: int = 16,
                 n_processor_steps: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cells = nn.ModuleList(
            [STGNNCell(hidden_dim) for _ in range(n_processor_steps)]
        )
        self.decoder_next_step = nn.Linear(hidden_dim, n_features)
        self.decoder_embedding = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x_seq, edge_index, edge_weight):
        T_in, N, F = x_seq.shape
        h = [torch.zeros(N, self.hidden_dim, device=x_seq.device) for _ in self.cells]

        for t in range(T_in):
            x_t = self.encoder(x_seq[t])           # (N, hidden_dim)
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t if i == 0 else h[i - 1], h[i], edge_index, edge_weight)

        final_state = h[-1]
        next_step_pred = self.decoder_next_step(final_state)
        threat_embedding = self.decoder_embedding(final_state)
        return next_step_pred, threat_embedding


def train_example():
    """
    Minimal training loop skeleton against the tensor produced by
    02_build_spatiotemporal_tensor.py. Fill in real train/val/test splits
    and log real metrics before claiming this is "trained" anywhere.
    """
    OUT_DIR = "/home/claude/varnika_gnn/outputs"  # adjust to your local path
    tensor = np.load(f"{OUT_DIR}/st_tensor.npy")           # (T, N, F)
    edge_index_np = np.load(f"{OUT_DIR}/edge_index.npy")   # (2, E)
    edge_weight_np = np.load(f"{OUT_DIR}/edge_weight.npy")  # (E,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32, device=device)

    T, N, F = tensor.shape
    mu, sigma = tensor.mean(axis=(0, 1)), tensor.std(axis=(0, 1)) + 1e-6
    norm_tensor = (tensor - mu) / sigma
    x_all = torch.tensor(norm_tensor, dtype=torch.float32, device=device)

    model = SpatioTemporalGNN(n_features=F).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    T_IN = 6
    split = int(T * 0.8)
    n_epochs = 20  # small demo run; increase once you have real time budget

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for t in range(T_IN, split - 1):
            x_seq = x_all[t - T_IN: t]      # (T_IN, N, F)
            target = x_all[t]               # (N, F)
            opt.zero_grad()
            pred, _ = model(x_seq, edge_index, edge_weight)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch:02d}  train_mse={total_loss / max(n_batches,1):.4f}")

    # --- held-out evaluation (report this number honestly, don't skip it) ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for t in range(max(split, T_IN), T - 1):
            x_seq = x_all[t - T_IN: t]
            target = x_all[t]
            pred, _ = model(x_seq, edge_index, edge_weight)
            val_losses.append(torch.mean((pred - target) ** 2).item())
    print(f"held-out val_mse={np.mean(val_losses):.4f} over {len(val_losses)} steps")
    print("Compare this against outputs/baseline_metrics.csv (the numpy baseline) "
          "before claiming the GNN improves on it.")


if __name__ == "__main__":
    train_example()
