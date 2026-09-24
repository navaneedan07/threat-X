# VARNIKA — THREAT-X

> **Primary:** GNN & Spatio-Temporal Modelling
> **Secondary:** Historical Event Analysis + Visualization Support
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 4 GNN / SPATIO-TEMPORAL |
| **Milestones** | **M4**, M10 (led by Sachin / Navaneedan / Varnika) |
| **Main artifact** | Graph construction · GNN prototype/experiment · metrics **if trained** · historical replay scripts · comparison plots · research evidence |
| **PPT** | Slides 2–6: GNN / event / research visuals (Slide 6 co-owner with Pushpa, Slide 5 co-owner) |
| **Demo segments** | 00–20s select case · 70–95s create/track threat (support) |
| **Stack** | Python, PyTorch, PyTorch Geometric/DGL, Xarray, NumPy, CUDA, NetworkX, Matplotlib, Cartopy |

You own the spatio-temporal representation. **Your credibility comes from being honest about
what is a prototype and what is trained** — the plan is explicit that a technically correct
prototype/specification is an acceptable first milestone. A fabricated metric is not.

---

## 1. Graph / mesh — what to actually build (`src/models/gnn/`)

### Step 1 — Define nodes (`mesh.py`)

Potential node features:

```text
latitude · longitude · weather variables · anomaly values · time · derived features
```

Document precisely which features a node carries, and at what level (surface / 850 / 500 hPa).
A node definition that is vague here makes every later result unreproducible.

### Step 2 — Define edges (`graph_builder.py`)

```text
spatial neighbours · geographic distance · mesh connectivity · temporal relationships
```

- Build the **spherical / mesh representation** — the weather field is spherical, so a
  naive lat/lon grid distorts distances near the poles.
- Decide the spatial neighbourhood radius and record it. `configs/model.yaml` →
  `graph.spatial_edge_radius_deg`, `temporal_edges`, `mesh_level` are `null` — fill them in
  with a reason.
- **Exit condition (M4):** graph/mesh construction is demonstrable and its interface to the
  threat state is defined.

### Step 3 — GNN prototype (`gnn.py`)

```text
Weather Field → Graph/Mesh → Message Passing → Spatio-Temporal Representation → Threat Output
```

- Prototype the architecture and **document what it should learn from the anomaly field**.
  "It learns features" is not an answer.
- Use `NetworkX` for graph inspection/visualization even if the model is PyG/DGL.

### Step 4 — Interface to the Threat Object

Define how GNN output connects to the Thing the team actually tracks:

- Publish your output schema in coordination with Navaneedan (`src/shared/contracts.py`).
- If the GNN is not trained, the interface still exists and the tracker consumes the
  deterministic path — say so plainly.

---

## 2. Historical event analysis and replay

- Replay selected historical events end to end.
- Produce **comparison plots** between detected/model outputs and reference data — this is
  your strongest evidence and works even if the GNN is not trained.
- Build the comparison figures Sachin needs for the final visual system.

**Exit condition:** replay scripts committed, comparison plots saved.

---

## 3. Research evidence

- Maintain the citation matrix in `docs/references.md` (GNN + global weather models +
  event-tracking literature). Add a row **before** citing it anywhere.
- Support slide 3 (architecture) and slide 4 (feasibility) with research-backed reasoning.
- **Record actual GPU/CUDA requirements** — this feeds the slide 4 feasibility argument.

> ⚠️ **Environment note:** this machine has Python **3.13**, but `environment.yml` pins
> **3.11** for good reason — PyTorch and PyTorch Geometric wheels lag on 3.13. Use the conda
> environment (`conda env create -f environment.yml`) for GNN work, not system Python.

---

## Integrity rules (these are yours to uphold)

- **Do not claim a trained GNN until there are actual experiments and metrics.**
- Progress labelling is mandatory in every artifact you produce:

| Label | Means |
|---|---|
| **Implemented** | Exists and has been tested |
| **Experimental** | Under development or being evaluated |
| **Planned** | Described, not yet written |
| **Target** | A requirement from the problem statement |

- If the GNN is **Experimental** or **Planned**, the reference supports the **approach**, not
  a result. Say that on the slide.

---

## Evidence you must save

- [ ] Graph construction code (committed)
- [ ] Graph/mesh visualization
- [ ] Node/edge definition documented
- [ ] GNN prototype (architecture + I/O documented)
- [ ] Actual metrics — **only if trained**
- [ ] Historical replay scripts
- [ ] Comparison plots
- [ ] Research/citation matrix in `docs/references.md`
- [ ] GPU/CUDA requirement estimate

---

## PPT contribution

| Slide | Your input |
|---|---|
| 2 IDEA | Conceptual GNN contribution — what the graph representation buys the team |
| 3 TECHNICAL | Spherical mesh / GNN diagram |
| 4 FEASIBILITY | Research-backed feasibility **and honest GPU requirements** |
| 5 IMPACT | Historical replay visualization (co-owner with Sachin + Navaneedan) |
| 6 REFERENCES | **Co-owner with Pushpa.** GNN / event-analysis literature matrix |

**Required evidence (from the team checklist):** GNN mesh diagram · historical replay ·
comparison plots · final reference matrix.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 00–20s | Select case (with Sachin) | Prepared historical extreme-weather event / forecast replay |
| 70–95s | Create/track threat (support Sachin) | Persistent Threat ID + path + centroid + footprint |

**If the GNN is implemented:** show the graph/mesh, the model output, and a real metric.
**If it is not:** show the graph construction, the architecture, and the historical replay —
and label it clearly as a prototype/concept.

> Either path is acceptable in the plan. **Inventing model results is not.**

---

## Fallback

| If this fails | Fall back to |
|---|---|
| GNN not trainable in time | Graph/mesh construction + documented architecture + interface spec — labelled Experimental |
| No GPU available | CPU prototype on a small subdomain; state the reduced scope |
| PyTorch / PyG install problems | Use the pinned conda env (`environment.yml`), not system Python 3.13 |
| Trained metrics unavailable | Report no metrics. Never estimate, interpolate or borrow a number. |
| Historical replay data missing | Use the prepared sample case in `data/samples/` |

---

## Definition of done

- **GNN:** graph/mesh construction is demonstrable; **trained claims have actual metrics**
- The GNN representation must be **reproducible**, its interface to the Threat Object must be
  **clear**, and every claimed result must have **real experimental evidence**
- Plus the team-wide rule: *runs from a reproducible input, produces a saved output, has at
  least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Graph
- [ ] Node definition completed
- [ ] Node features defined (with levels)
- [ ] Spatial edges defined (radius recorded)
- [ ] Temporal edges defined
- [ ] Mesh / spherical representation documented
- [ ] Graph construction code works
- [ ] Graph visualization created

### GNN
- [ ] Architecture defined
- [ ] Prototype implemented
- [ ] Input / output documented
- [ ] What it should learn from the anomaly field documented
- [ ] Training attempted (if feasible)
- [ ] Evaluation metric calculated **if trained**
- [ ] Results saved
- [ ] Status label applied (Implemented / Experimental / Planned)
- [ ] No unsupported performance claim
- [ ] GPU/CUDA requirement recorded

### Historical analysis
- [ ] Event selected
- [ ] Event timeline created
- [ ] Weather maps created
- [ ] Anomaly maps created
- [ ] Replay script created
- [ ] Comparison plots created

### Integration
- [ ] GNN output schema defined
- [ ] Tracker interface defined (with Sachin)
- [ ] Threat Object interface defined (with Navaneedan)
- [ ] GNN output can be visualized

### Research
- [ ] GNN papers collected in `docs/references.md`
- [ ] Global weather-model papers collected
- [ ] Event-tracking references collected
- [ ] Citation matrix created
- [ ] Every row marked verified before use
- [ ] PPT claims checked against sources

### PPT
- [ ] Slide 2 GNN visual supplied
- [ ] Slide 3 GNN/mesh architecture supplied
- [ ] Slide 4 feasibility + GPU evidence supplied
- [ ] Slide 5 historical replay supplied
- [ ] Slide 6 literature matrix supplied

### Demo
- [ ] Graph/mesh can be shown
- [ ] Model output shown **if implemented**
- [ ] Prototype clearly labelled if not implemented
- [ ] Historical replay works
- [ ] Static fallback saved
