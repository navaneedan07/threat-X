"""
VARNIKA — Stage 5: Spatio-temporal GNN prototype (PyTorch + PyTorch Geometric)

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
                07_export_interface.py for the JSON contract).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except ImportError as e:
    raise ImportError(
        "torch_geometric is required for this file. Install with:\n"
        "  pip install torch-geometric\n"
    ) from e

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUT_DIR = BASE_DIR / "outputs"


class STGNNCell(nn.Module):
    """
    One round of spatial message passing (GCNConv) followed by a GRU update.
    Models spatial correlation across the mesh and update across time.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, h, edge_index, edge_weight):
        # x: (N, hidden_dim) node representation at this time step
        # h: (N, hidden_dim) previous hidden state of the node
        # spatial message passing over the mesh
        m = torch.relu(self.gcn(x, edge_index, edge_weight))
        # temporal recurrence
        h_next = self.gru(m, h)
        return h_next


class SpatioTemporalGNN(nn.Module):
    """
    Encoder-processor-decoder ST-GNN:
      - Encoder:   lifts (N, F) raw features to (N, hidden_dim)
      - Processor: K rounds of STGNNCell across T_in time steps
      - Decoder:   predicts (N, F) next-step features AND (N, embed_dim) threat embedding
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


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def train_and_evaluate(
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 32,
    embed_dim: int = 16,
    n_processor_steps: int = 2,
    t_in: int = 6,
    split: int = 120,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts
    tensor = np.load(OUT_DIR / "st_tensor.npy")  # (T=151, N=66, F=5)
    edge_index_np = np.load(OUT_DIR / "edge_index.npy")  # (2, E)
    edge_weight_np = np.load(OUT_DIR / "edge_weight.npy")  # (E,)
    with open(OUT_DIR / "feature_order.txt") as f:
        feature_names = f.read().splitlines()

    T, N, F = tensor.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset shape: T={T}, N={N}, F={F}")

    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32, device=device)

    # 2. Strict training partition normalization (no leakage)
    # Train partition: timesteps 0 ... split-1 (t=0..119)
    train_tensor = tensor[:split]
    mu = train_tensor.mean(axis=(0, 1), keepdims=True)  # (1, 1, F)
    sigma = train_tensor.std(axis=(0, 1), keepdims=True) + 1e-6  # (1, 1, F)

    norm_tensor = (tensor - mu) / sigma
    x_all = torch.tensor(norm_tensor, dtype=torch.float32, device=device)

    # 3. Model, optimizer, loss
    torch.manual_seed(42)
    np.random.seed(42)
    model = SpatioTemporalGNN(
        n_features=F,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        n_processor_steps=n_processor_steps,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1

    # Target ranges:
    # Training targets: t = t_in ... split - 1  (t = 6 ... 119, 114 targets)
    # Evaluation targets: t = split ... T - 1   (t = 120 ... 150, 31 targets)
    train_target_range = range(t_in, split)
    val_target_range = range(split, T)

    print(f"Training targets: {list(train_target_range)[0]} to {list(train_target_range)[-1]} ({len(train_target_range)} steps)")
    print(f"Evaluation targets: {list(val_target_range)[0]} to {list(val_target_range)[-1]} ({len(val_target_range)} steps)")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_steps = 0

        for t in train_target_range:
            x_seq = x_all[t - t_in : t]  # (T_in, N, F)
            target = x_all[t]            # (N, F)

            optimizer.zero_grad()
            pred, _ = model(x_seq, edge_index, edge_weight)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_steps += 1

        avg_train_loss = total_train_loss / max(n_train_steps, 1)
        train_losses.append(avg_train_loss)

        # Validation on held-out partition
        model.eval()
        total_val_loss = 0.0
        n_val_steps = 0
        with torch.no_grad():
            for t in val_target_range:
                x_seq = x_all[t - t_in : t]
                target = x_all[t]
                pred, _ = model(x_seq, edge_index, edge_weight)
                val_loss = criterion(pred, target)
                total_val_loss += val_loss.item()
                n_val_steps += 1

        avg_val_loss = total_val_loss / max(n_val_steps, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {avg_train_loss:.5f} | Held-out Val MSE: {avg_val_loss:.5f}")

    print(f"Best Held-out Val MSE: {best_val_loss:.5f} at Epoch {best_epoch:02d}")

    # Load best model weights for inference & evaluation
    model.load_state_dict(best_model_state)
    model.eval()

    # 4. Save Checkpoint (Task 7)
    checkpoint_path = OUT_DIR / "st_gnn_checkpoint.pt"
    checkpoint = {
        "model_state_dict": best_model_state,
        "n_features": F,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "n_processor_steps": n_processor_steps,
        "normalization_mean": mu,
        "normalization_std": sigma,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "trained_epoch": best_epoch,
        "feature_names": feature_names,
        "t_in": t_in,
        "split": split,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved trained model checkpoint -> {checkpoint_path}")

    # Save training loss plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, epochs + 1), train_losses, label="Train MSE (t=6..119)", color="#1f4e8c", lw=2)
    ax.plot(range(1, epochs + 1), val_losses, label="Held-out Val MSE (t=120..150)", color="#d95f02", lw=2)
    ax.axvline(best_epoch, color="#7570b3", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Normalized)")
    ax.set_title("ST-GNN Training and Held-Out Validation Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_loss.png", dpi=160)
    plt.close(fig)
    print(f"Saved training loss plot -> {OUT_DIR / 'training_loss.png'}")

    # 5. Full Record & Held-Out Predictions + Latent Representations (Task 10)
    all_preds_norm = np.zeros((T - 1, N, F), dtype=np.float32)
    all_embeddings = np.zeros((T - 1, N, embed_dim), dtype=np.float32)

    with torch.no_grad():
        for t in range(1, T):
            if t >= t_in:
                x_seq = x_all[t - t_in : t]
                pred_t, emb_t = model(x_seq, edge_index, edge_weight)
                all_preds_norm[t - 1] = pred_t.cpu().numpy()
                all_embeddings[t - 1] = emb_t.cpu().numpy()
            else:
                all_preds_norm[t - 1] = norm_tensor[t - 1]
                pad_seq = torch.stack([x_all[0]] * (t_in - t) + [x_all[i] for i in range(t)])
                _, emb_t = model(pad_seq, edge_index, edge_weight)
                all_embeddings[t - 1] = emb_t.cpu().numpy()

    # Physical units for full record
    all_preds_phys = all_preds_norm * sigma + mu

    # Save full predictions and embeddings
    np.save(OUT_DIR / "gnn_predictions_normalised.npy", all_preds_norm)
    np.save(OUT_DIR / "gnn_predictions.npy", all_preds_phys)
    np.save(OUT_DIR / "threat_embeddings.npy", all_embeddings)

    # Held-out predictions (31 steps, t = 120..150, indices 119..149 in all_preds)
    held_gnn_norm = all_preds_norm[split - 1 : T - 1]      # (31, 66, 5)
    held_gnn_phys = all_preds_phys[split - 1 : T - 1]      # (31, 66, 5)
    held_emb = all_embeddings[split - 1 : T - 1]           # (31, 66, 16)
    np.save(OUT_DIR / "gnn_held_out_predictions_norm.npy", held_gnn_norm)
    np.save(OUT_DIR / "gnn_held_out_predictions_physical.npy", held_gnn_phys)
    np.save(OUT_DIR / "gnn_held_out_threat_embeddings.npy", held_emb)

    # 6. Held-out Evaluation vs Baselines (Task 8 & Task 9)
    baseline_preds_norm = np.load(OUT_DIR / "baseline_predictions_normalised.npy")  # (T-1, N, F)
    baseline_actuals_norm = np.load(OUT_DIR / "baseline_actuals_normalised.npy")    # (T-1, N, F)

    held_smoothing_norm = baseline_preds_norm[split - 1 : T - 1]  # (31, 66, 5)
    held_actuals_norm = baseline_actuals_norm[split - 1 : T - 1]  # (31, 66, 5)

    # Persistence predictions on held-out slice (target t is predicted by t-1)
    held_persist_norm = norm_tensor[split - 1 : T - 1]           # (31, 66, 5)

    # Physical actuals and baselines
    held_actuals_phys = tensor[split : T]                         # (31, 66, 5)
    held_smoothing_phys = held_smoothing_norm * sigma + mu
    held_persist_phys = tensor[split - 1 : T - 1]

    units_map = {
        "temperature": "°C",
        "pressure": "hPa",
        "humidity": "%",
        "wind_speed": "m/s",
        "precipitation": "mm",
    }

    eval_rows = []
    print("\n" + "=" * 80)
    print("HELD-OUT EVALUATION METRICS (31 transitions: timesteps 120 -> 150)")
    print("=" * 80)

    for i, name in enumerate(feature_names):
        unit = units_map.get(name, "a.u.")

        # Normalized RMSE
        gnn_r_norm = rmse(held_gnn_norm[..., i], held_actuals_norm[..., i])
        smooth_r_norm = rmse(held_smoothing_norm[..., i], held_actuals_norm[..., i])
        persist_r_norm = rmse(held_persist_norm[..., i], held_actuals_norm[..., i])

        # Physical RMSE
        gnn_r_phys = rmse(held_gnn_phys[..., i], held_actuals_phys[..., i])
        smooth_r_phys = rmse(held_smoothing_phys[..., i], held_actuals_phys[..., i])
        persist_r_phys = rmse(held_persist_phys[..., i], held_actuals_phys[..., i])

        imp_over_persist = 100.0 * (persist_r_norm - gnn_r_norm) / persist_r_norm
        imp_over_smooth = 100.0 * (smooth_r_norm - gnn_r_norm) / smooth_r_norm

        eval_rows.append({
            "feature": name,
            "unit": unit,
            "gnn_rmse_norm": round(gnn_r_norm, 4),
            "smoothing_baseline_rmse_norm": round(smooth_r_norm, 4),
            "persistence_rmse_norm": round(persist_r_norm, 4),
            "gnn_rmse_physical": round(gnn_r_phys, 4),
            "smoothing_rmse_physical": round(smooth_r_phys, 4),
            "persistence_rmse_physical": round(persist_r_phys, 4),
            "gnn_vs_persistence_improvement_pct": round(imp_over_persist, 2),
            "gnn_vs_smoothing_improvement_pct": round(imp_over_smooth, 2),
        })

    eval_df = pd.DataFrame(eval_rows)
    metrics_csv_path = OUT_DIR / "gnn_vs_baseline_metrics.csv"
    eval_df.to_csv(metrics_csv_path, index=False)
    print(eval_df.to_string(index=False))
    print(f"\nSaved held-out evaluation comparison -> {metrics_csv_path}")

    return eval_df


if __name__ == "__main__":
    train_and_evaluate()
