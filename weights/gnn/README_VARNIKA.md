# Varnika — GNN & Spatio-Temporal Modelling + Historical Event Analysis

This package is your work-card deliverable set, built and test-run against
the team's `dataset.csv` (11×6 lat/lon grid, hourly, 5 variables: temperature,
pressure, humidity, wind_speed, precipitation — 10,000 rows = 151 complete
grid snapshots).

Everything below already ran successfully on this data. The only thing you
still need to do on your own machine is Stage 5 (the actual GNN), because
this container has no internet access to install `torch` /
`torch-geometric`. Everything else is done and its outputs are in `outputs/`.

## What each file is, and why it exists

| # | File | What it does | Runs here? |
|---|------|--------------|------------|
| 1 | `01_graph_construction.py` | Builds the 66-node spatial mesh (rook adjacency) from the grid, saves `edge_index.npy`/`edge_weight.npy`, draws `mesh_diagram.png` | ✅ done |
| 2 | `02_build_spatiotemporal_tensor.py` | Reshapes the flat CSV into a `(T=151, N=66, F=5)` tensor aligned to the mesh's node ordering | ✅ done |
| 3 | `03_baseline_predictor.py` | A hand-written, **un-trained** one-hop graph-smoothing baseline (normalised-adjacency message pass, à la a single fixed GCN layer) — this is your Milestone-1 "technically correct prototype." Evaluates real RMSE vs a naive persistence baseline on a held-out slice | ✅ done |
| 4 | `04_historical_event_replay.py` | Finds the real highest-precipitation window in the data and produces the spatial snapshot, time-series, and predicted-vs-actual comparison plots | ✅ done |
| 5 | `05_st_gnn_model.py` | The actual encoder–processor–decoder spatio-temporal GNN (PyTorch + PyTorch Geometric): GCNConv for spatial message passing + GRUCell for temporal recurrence, in the lineage of GraphCast / MeshGraphNets | ⚠️ **run this yourself** — needs `torch`, `torch-geometric` |
| 6 | `06_architecture_diagram.py` | Draws the encoder/processor/decoder diagram for the PPT | ✅ done |
| 7 | `07_export_interface.py` | Produces the JSON contract that hands your output to Sachin's tracker and Hariharan's precursor stage | ✅ done |

## Run order (once you add a GPU/Colab environment for step 5)

```bash
python3 01_graph_construction.py
python3 02_build_spatiotemporal_tensor.py
python3 03_baseline_predictor.py
python3 04_historical_event_replay.py
python3 06_architecture_diagram.py
python3 07_export_interface.py

# on a machine with torch + torch-geometric installed:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
python3 05_st_gnn_model.py
```

Files 1–4, 6–7 only need `pandas`, `numpy`, `matplotlib` (already available).
Copy `dataset.csv` to `/mnt/user-data/uploads/dataset.csv` relative to wherever
you run them, or edit the `DATA_PATH`/`OUT_DIR` constants at the top of each
file.

## What's in `outputs/` after running everything

- `node_table.csv`, `edge_index.npy`, `edge_weight.npy` — the mesh, ready for both the baseline and the real GNN
- `mesh_diagram.png` — **your Slide 3/6 "spherical mesh/GNN diagram"**
- `gnn_architecture_diagram.png` — **your Slide 2 "conceptual GNN contribution"** visual
- `st_tensor.npy`, `feature_order.txt` — the shared spatio-temporal tensor
- `baseline_metrics.csv` — real, computed RMSE table (baseline vs persistence, held-out only) — this is what you put on a slide **as a baseline number**, not as "GNN accuracy"
- `event_spatial_snapshot.png`, `event_timeseries.png`, `event_predicted_vs_actual.png` — **your "historical replay / comparison plots"** deliverable
- `interface_export_example.json`, `interface_schema.json` — the handoff contract for Sachin/Hariharan

## Honesty checklist before this goes on a slide

- The baseline (`03_baseline_predictor.py`) has **no learned parameters**. Label it "deterministic graph-smoothing baseline," never "GNN result."
- Say plainly that Stage 5 is a **prototype/specification** until you have actually trained it and logged a held-out metric the same way `baseline_metrics.csv` does. The work card explicitly allows this for Milestone 1 — don't feel pressure to overclaim.
- The extreme event found in the data (timestep 25) is a **synthetic dataset's** highest-precipitation window, not a verified named real-world event — say "highest-precipitation window in the supplied dataset," not "Cyclone X" or similar, unless the team swaps in real ERA5/IMDAA data later.
- `interface_export_example.json`'s `threat_embedding` field is currently a **placeholder** re-using the predicted feature vector — flag this to Sachin/Hariharan explicitly so nobody assumes it's a trained embedding.

## Research references for Slide 6 (verified, not fabricated)

- Lam, R. et al. (2023). *Learning skillful medium-range global weather forecasting.* Science, 382(6677), 1416–1421. — origin of the encoder/processor/decoder GNN-on-a-mesh design pattern this prototype scales down.
- Pfaff, T. et al. (2021). *Learning Mesh-Based Simulation with Graph Networks.* ICLR. — origin of the mesh-graph message-passing idea used in the processor stage.
- For the spatio-temporal recurrence design (GCN + GRU/temporal conv combined), the general architecture family is documented in the traffic-forecasting GNN literature (e.g. DCRNN, Graph WaveNet) — cite these as "architecture family," not as papers you've reproduced results from, since this prototype hasn't been benchmarked against them.

## Demo responsibility (yours, per the run-of-show)

00–20s and 70–95s: you help select/replay the historical case and show the
persistent Threat ID being created from the graph/mesh representation. Use
`mesh_diagram.png` and `event_spatial_snapshot.png` live, and have
`event_predicted_vs_actual.png` ready in case a judge asks "does the model
do anything yet" — you can honestly say "yes, here's a deterministic
baseline with a measured accuracy gain over doing nothing, and here's the
trained-GNN upgrade path," which is exactly the fallback-hierarchy story
the whole team is telling.
