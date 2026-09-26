"""
VARNIKA — Stage 6: Interface export

Defines and produces the actual handoff artifact between the GNN/spatio-
temporal stage (Varnika) and the Threat Object / tracker (Sachin +
Navaneedan) and precursor stage (Hariharan), per the pipeline table in the
team plan (Stage 4 GNN -> Stage 5 Tracking / Stage 6 Precursors).

Contract (documented here so all three of you can code against it without
waiting on each other):

{
  "timestep": int,
  "nodes": [
    {
      "node_id": int,
      "latitude": float,
      "longitude": float,
      "raw_features": {"temperature":.., "pressure":.., "humidity":..,
                        "wind_speed":.., "precipitation":..},
      "predicted_next_features": {...same keys...},   # from the model, one step ahead
      "threat_embedding": [float, ...]                 # fixed-length vector;
                                                         # currently the un-trained
                                                         # baseline's smoothed state,
                                                         # replaced by decoder_embedding
                                                         # once 05_st_gnn_model.py is trained
    },
    ...
  ]
}

This uses the numpy baseline's outputs (real numbers, computed today).
Swap in the trained GNN's `threat_embedding` output once available -- the
schema does not need to change.
"""

import numpy as np
import pandas as pd
import json

OUT_DIR = "/home/claude/varnika_gnn/outputs"


def main():
    nodes = pd.read_csv(f"{OUT_DIR}/node_table.csv")
    tensor = np.load(f"{OUT_DIR}/st_tensor.npy")                          # raw units (T,N,F)
    preds_norm = np.load(f"{OUT_DIR}/baseline_predictions_normalised.npy")  # (T-1,N,F), preds[i] -> step i+1
    mu = np.load(f"{OUT_DIR}/norm_mu.npy")
    sigma = np.load(f"{OUT_DIR}/norm_sigma.npy")
    with open(f"{OUT_DIR}/feature_order.txt") as f:
        feature_names = f.read().splitlines()

    # Export one representative timestep (the peak event from stage 4) as a
    # worked example of the schema, plus the schema itself as a separate file.
    example_t = 25  # matches the peak timestep found in 04_historical_event_replay.py
    pred_idx = example_t - 1

    record = {"timestep": example_t, "nodes": []}
    for _, row in nodes.iterrows():
        nid = int(row["node_id"])
        raw = tensor[example_t, nid, :]
        pred_raw = preds_norm[pred_idx, nid, :] * sigma[0, 0, :] + mu[0, 0, :]

        record["nodes"].append({
            "node_id": nid,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "raw_features": {name: float(v) for name, v in zip(feature_names, raw)},
            "predicted_next_features": {name: float(v) for name, v in zip(feature_names, pred_raw)},
            "threat_embedding": [round(float(v), 4) for v in pred_raw],  # placeholder:
            # baseline has no learned embedding, so we reuse the predicted
            # feature vector as a stand-in of the right shape. Replace this
            # list with model.decoder_embedding output once 05 is trained.
        })

    with open(f"{OUT_DIR}/interface_export_example.json", "w") as f:
        json.dump(record, f, indent=2)

    schema_doc = {
        "description": "Contract between Varnika's spatio-temporal stage and "
                        "Sachin's tracker / Hariharan's precursor stage.",
        "fields": {
            "timestep": "sequential hourly index into the shared grid record",
            "nodes[].node_id": "0..65, fixed by 01_graph_construction.py node_table.csv",
            "nodes[].raw_features": "actual observed grid values at this timestep",
            "nodes[].predicted_next_features": "one-step-ahead prediction for this node",
            "nodes[].threat_embedding": "fixed-length per-node vector for the tracker "
                                        "to use as similarity/matching input across time; "
                                        "currently a placeholder (see comments in source)",
        },
    }
    with open(f"{OUT_DIR}/interface_schema.json", "w") as f:
        json.dump(schema_doc, f, indent=2)

    print(f"Wrote example export for timestep {example_t} -> "
          f"{OUT_DIR}/interface_export_example.json")
    print(f"Wrote schema doc -> {OUT_DIR}/interface_schema.json")


if __name__ == "__main__":
    main()
