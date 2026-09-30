# 🌦️ AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies

> **Threat-X** — Smart India Hackathon 2026  
> **Theme:** Smart Automation  
> **Category:** Software  
> **Organization:** Ministry of Earth Sciences (MoES)  
> **Team:** **Bots** — Navaneedan · Aravinth · Hariharan · Sachin · Pushpa · Varnika  
> **See [`# 🧭 Start Here`](#-start-here) for what runs, what was measured, and what is not done.**

An end-to-end AI system for detecting, tracking, explaining, and localizing extreme-weather anomalies in medium-range numerical weather prediction (NWP) forecasts.

The system moves beyond a static "extreme weather detected" alert. It creates a **persistent Threat Object** for every detected anomaly and follows its lifecycle across space and time — measuring its movement, intensity, footprint, persistence, atmospheric precursors, and transition toward higher severity.

The intended output is a **hyper-local, uncertainty-aware weather threat layer** that can support earlier and more targeted warnings.

---

# 🧭 Start Here

> **Read this section first if you are evaluating the repository.** It states what runs,
> what was measured, and what is honestly not done. Every number below is reproduced by a
> command in this file or linked to the experiment that produced it
> ([`docs/experiments.md`](docs/experiments.md)).

## What this is

Threat-X turns a medium-range forecast field into a **persistent Threat Object** — an
identity that survives across time, a trajectory, atmospheric precursor context, an
estimate of whether it is moving toward a higher-severity state, and a localised footprint
at finer resolution than the input. The point is to move past a one-off *"anomaly
detected"* flag.

One command runs the whole chain on the real archives:

```bash
python -m src.shared.demo --event amphan     # or: --event heatwave
```

On a machine with the data fetched (see [What ships in the repo](#what-ships-in-the-repo-and-what-does-not)),
both events complete **all twelve steps**. A step that cannot be computed on the data
actually present says so and is skipped — it is never replaced by a synthetic stand-in.

## What runs

| Stage | Where | Status | Run it |
|---|---|---|---|
| Data ingestion (ERA5, ERA5-Land, ECMWF open data, IBTrACS) | `src/data/` | implemented | `python -m src.data.cds_fetch --help` |
| Climatological baseline, 1991–2020 per region | `src/data/`, `src/detection/` | implemented | `python -m src.data.cds_fetch --climatology --all-regions` |
| Extreme anomaly detection | `src/detection/` | implemented — runs on **both** events | `python -m src.detection.anomaly_detection --event amphan --variable mean_sea_level_pressure` |
| Threat Object + tracking | `src/tracking/` | implemented | `python -m src.tracking.pipeline --help` |
| Atmospheric precursors | `src/precursors/` | implemented | `python -m src.precursors.pipeline --all-events` |
| Threat lifecycle (deterministic) | `src/transition/` | implemented | `python -m src.shared.visualization` |
| Transition intelligence | `src/models/transition/` | implemented — **result is negative**, below | `python -m src.models.transition.transition_model` |
| Downscaling, interpolation baseline | `src/downscaling/` | implemented and scored | `python -m src.downscaling.real_pair --all-events` |
| Downscaling, learned filter + calibration | `src/models/downscaling/` | implemented and scored | `python -m src.models.downscaling.super_resolution --all-events` |
| Validation & gate | `src/validation/` | implemented — thresholds deliberately `null` | library, invoked by the demo (step 11) |
| REST API, 8 endpoints | `backend/` | implemented, partly fixture-backed | `uvicorn backend.main:app --reload` |
| Real-map dashboard (Cartopy + Folium) | `frontend/dashboard.py` | implemented — **this is the demo** | `python frontend/dashboard.py --all-events --with-folium --with-gnn` |
| GNN → detector → tracker bridge | `weights/gnn/run_gnn_tracking_demo.py` | implemented, checkpoint-aware | `python weights/gnn/run_gnn_tracking_demo.py` |
| Interactive map only | `frontend/visulisation.py` | implemented | `python frontend/visulisation.py --event amphan` |
| Spatio-temporal GNN on real ERA5 | `weights/gnn/09_train_real_era5.py` | implemented and scored | `python weights/gnn/09_train_real_era5.py` |
| GNN → detector → tracker bridge | `weights/gnn/run_gnn_tracking_demo.py` | implemented | `python weights/gnn/run_gnn_tracking_demo.py` |
| Web dashboard (React) | — | **not started** | — |

> `src/validation/evaluation.py` and `src/tracking/tracker.py` are libraries, not
> commands: the README previously advertised `python -m src.data.preprocessing` and
> `python -m src.detection.anomaly_detector`, neither of which exists. The table above
> lists only entry points that were executed while writing this section.

## What ships in the repo (and what does not)

`.gitignore` excludes large weather archives, so a fresh clone contains **18 MB** of data —
event archives for both cases, the Bay of Bengal climatology, and the committed API
fixtures. That is deliberately *not* everything the full demo touches:

| Needed by | File | Size | In the repo? | If missing |
|---|---|---|---|---|
| Steps 1–9, both events | `data/raw/era5_amphan.nc`, `era5_heatwave.nc` | 6 MB | ✅ yes | — |
| Step 4, Amphan | `data/climatology/era5_clim_1991..2020.nc` | 12 MB | ✅ yes | — |
| Step 4, heatwave | `data/climatology/era5_clim_north_india_*.nc` | 331 MB | ❌ no | `python -m src.data.cds_fetch --climatology --region north_india --years 1991-2020` |
| Step 10, both events | `data/raw/era5_land_*.nc` | 22 MB | ❌ no | `python -m src.data.land_fetch --event <event>` |
| Precursor upper-air fields | `era5_*_plev.nc` | — | ❌ no | `python -m src.data.cds_fetch --event <event> --pressure-levels` |
| GNN pilot (frozen week) | `data/raw/era5_gnn_pilot_20200516_20200522.zip` | 1 MB | ❌ no | `python -m src.data.cds_fetch --gnn-pilot` |
| GNN training months | `data/raw/era5_gnn_2020_03..05.zip` | 13 MB | ❌ no | `python -m src.data.cds_fetch --gnn-month 2020-03 --gnn-month 2020-04 --gnn-month 2020-05` |
| GNN derived tensor + timestamps | `weights/gnn/outputs/era5_pilot/` | 230 KB | ✅ yes | — (so `08` and the GNN demo run on a clone) |
| GNN checkpoints (2 × 75 KB) | `weights/gnn/outputs/**/*.pt` | 150 KB | ✅ yes | — (so the two GNN test files run on a clone) |

**Resulting demo coverage:** with only what ships, Amphan runs **11 / 12** (step 10 needs
ERA5-Land) and the heatwave runs **9 / 12** (step 4 needs the North India climatology, and
steps 5–7 depend on it). Fetch the two rows above and both reach **12 / 12**.

Every gap is reported as a named gap with the exact command to close it — never as a
silently skipped step.

> Both fetches need Copernicus CDS credentials in a gitignored `.env`. If you are
> evaluating this repository and cannot obtain them, the committed
> [`docs/experiments.md`](docs/experiments.md) carries the measured outputs of all of the
> above, and Amphan's detection, tracking, precursors, transition and API stages run
> immediately on clone.

## What was measured

All figures are real runs on the committed archives. Gates stay `undecided` because
`configs/validation.yaml` thresholds are `null` by design — see [Research Integrity](#-research-integrity).

**Downscaling — ERA5 0.25° → ERA5-Land 0.10°, spatial holdout (D2–D5)**

| Model | RMSE | MAE | Peak preservation | Extreme bias | IoU (p99) | Dice (p99) |
|---|---|---|---|---|---|---|
| Interpolation baseline | 1.672 K | 1.342 K | 1.0002 | +0.064 K | **0.2874** | **0.4032** |
| Learned filter only | **1.441 K** | **1.105 K** | 0.9977 | −0.702 K | 0.2435 | 0.3312 |
| Learned + quantile calibration | 1.466 K | 1.137 K | 0.9994 | −0.179 K | 0.2737 | 0.3832 |

No single model wins: bilinear wins footprint overlap, the learned model wins error and
the tail percentiles. **Both are reported rather than one being chosen by preference.**
Amphan is *not* scored for the learned model (only 24.9 % of its holdout has a reference);
its baseline keeps just **9 %** of the fine precipitation peak with zero footprint
overlap — the clearest evidence in the repo that interpolation fails.

**Detection (DET1–DET2)**

| Event | Variable | Frames | With regions | Strongest |
|---|---|---|---|---|
| Amphan | mean sea-level pressure | 48 | 43 | **19.36 σ** — 942.7 hPa against a 1003.8 hPa, 3.15 hPa σ baseline |
| Heatwave | 2 m temperature | 80 | 18 | **3.71 σ** at 22.00 N 72.50 E |

**Transition intelligence (T1–T2) — a negative result, reported as measured**

Brier skill against the constant base-rate forecast is **−0.47 … +0.36** (Amphan) and
**+0.01 … +0.09** (heatwave) across 6/12/18/24 h horizons. At 12 h the Amphan model is
*worse* than always predicting the base rate, and the heatwave label is 86.8 % positive at
24 h, so "always yes" scores well while discriminating nothing. The publishing gate
withholds 3 of 7 fitted horizons. **With two events and 27–57 rows per horizon, no credible
transition model can be trained** — the machinery is correct, the data is not sufficient.

**GNN on real ERA5 (`weights/gnn/`, rows G1–G4)**

This stage was rebuilt and corrected during this session, and the corrections are part of
the result. Three defects were found, measured and fixed:

1. **The frozen checkpoint was fitted on the wrong distribution.** The previously committed
   result compared a checkpoint trained on `dataset.csv` against the ERA5 pilot — and that
   training data has an hourly **precipitation mean of 7.40 mm/h against the pilot's
   0.075 mm/h** (~100×) and a **pressure σ of 60 hPa against 2.97 hPa**. `08` records the
   mismatch in `domain_diagnostics.csv` (`pilot_std_over_checkpoint_std`: pressure **0.049**,
   precipitation **0.025**). The features the checkpoint appeared to win on were the ones
   that do not transfer.
2. **A flat loss let one feature dominate.** One hour ahead, the normalised persistence MSE
   is **0.99 for precipitation but 0.07–0.10 for the other four**; unweighted, the optimiser
   bought a little precipitation skill by giving away the features persistence already
   solves. The loss is now weighted by 1/persistence MSE, which makes a weighted score of
   1.0 *mean* "equal to persistence".
3. **An evaluation broadcasting bug** briefly compared every prediction hour against every
   target hour; the fix (and a shape guard that makes it unrepeatable) is in stage `09`.

The corrected model (`09 --architecture residual --loss-weighting persistence`) trains on
**real ERA5** — 1824 hourly steps, 2020-03-01 → 05-15, the same 66-node mesh — and is scored
on the held-out pilot week with the identical window indices as `08`:

| Feature | **ST-GNN RMSE** | Persistence RMSE | Improvement | vs frozen ckpt (G1) |
|---|---|---|---|---|
| temperature (°C) | **0.896** | 1.011 | **+11.4 %** | 4.2× better |
| pressure (hPa) | **0.559** | 0.706 | **+20.8 %** | 20× better |
| humidity (%) | **4.242** | 4.557 | **+6.9 %** | 4.9× better |
| wind speed (m/s) | **0.568** | 0.580 | **+2.1 %** | 5.0× better |
| precipitation (mm/h) | **0.276** | 0.281 | **+2.0 %** | 30× better |

**Beats persistence on all five features**, with an aggregate weighted MSE of **0.0865
against persistence's 0.1030** (train 0.0534 — modest, explicable overfitting; the gain
survives it). The ablation is in `docs/experiments.md`: the plain decoder loses to
persistence on all five, so both the residual head and the weighting are doing real work.
The residual head is also *provable* — a zero increment reproduces persistence exactly, and
a test asserts it at epoch 0.

The stage is wired into the pipeline: `run_gnn_tracking_demo.py` loads the checkpoint
(architecture-aware, so an increment is never silently returned as a field), predicts one
hour ahead, feeds the field through the **real detector** and the **real tracker**, and
produces threat ID **`THR-2020-0001`** in `gnn_tracking_demo.json`.

## What is not done

Listed so nothing here is a surprise:

- **No React dashboard.** The frontend is a static Folium HTML map, not the interactive UI
  the original plan describes.
- **Gate thresholds are `null`.** Every PASS / DEGRADE / SUPPRESS verdict reports
  `undecided`. Filling them in requires measured distributions that one event cannot give.
- **No reference labels**, so detection precision/recall and tracking error are implemented
  but **not measured**.
- **No ensemble data**, so `ensemble_agreement` is `null` end to end.
- **Diffusion downscaling not attempted** — it needs a validation split with more than two
  events, and an unfalsifiable model is worse than an absent one.
- **The GNN's first training set is not ERA5 and does not transfer to it.** The checkpoint
  at `weights/gnn/outputs/st_gnn_checkpoint.pt` is fitted on `dataset.csv`, whose hourly
  precipitation mean is ~100× the reanalysis mean over the same box. `08` measures that
  mismatch rather than hiding it; stage `09` is the repair. Both numbers are reported.
- **The GNN is a 66-node regional mesh**, not a global or operational model, and its
  training window is 76 days of one spring. It is a method demonstration, not a product.
- **The Bay of Bengal climatology is a single-day window** (240 samples/cell) while the
  North India one is full May (7440). Known gap, documented in
  [`docs/dataset.md`](docs/dataset.md), not silently patched.

## Tests

```bash
pytest
# 514 passed
```

The full suite now includes the two GNN suites (`test_gnn_era5_evaluation.py`,
`test_gnn_tracking_demo.py`) and a third (`test_gnn_real_era5.py`) for the real-ERA5 stage.
The first two previously could not even be collected because PyTorch was not installed.
They need the committed pilot tensor and checkpoints listed in the shipping table above, so
they pass on a clone with no network access.

## Where the detail lives

| I want… | Read |
|---|---|
| every experiment, number and correction | [`docs/experiments.md`](docs/experiments.md) |
| architecture and module ownership | [`docs/architecture.md`](docs/architecture.md) |
| dataset decisions and known data gaps | [`docs/dataset.md`](docs/dataset.md) |
| the API contract and null conventions | [`docs/api.md`](docs/api.md) |
| transition target definition and gate | [`docs/transition.md`](docs/transition.md) |
| per-member work cards and checklists | [`team/`](team/) |

---

## 📌 Problem Statement

Medium-range weather forecasting over **3–10 days** provides valuable lead time, but extreme-weather events are difficult to identify and track precisely inside large, multidimensional NWP datasets.

A conventional workflow may identify a broad region experiencing heavy rainfall, heat, cold, or another anomaly, but operational decision-making often needs more specific answers:

- Where exactly is the threat?
- Where is it moving?
- Is its footprint expanding?
- Is its intensity increasing?
- What atmospheric conditions are driving the evolution?
- How long is the threat expected to persist?
- Is it likely to transition to a more severe state?
- How confident is the forecast?
- Which smaller geographic region is most affected?

The problem statement specifically calls for an AI-driven pipeline capable of tracking extreme anomalies in medium-range forecast data and improving localization from a coarse global field toward a finer local representation.

---

# 💡 Our Proposed Solution

We propose a modular **Extreme Weather Threat Intelligence Pipeline**:

```text
NWP / Ensemble Forecast Data
            │
            ▼
   Data Preprocessing
            │
            ▼
   Climatological Baseline
      ERA5 / IMDAA
            │
            ▼
   Extreme Anomaly Detection
            │
            ▼
     Persistent Threat Object
            │
            ▼
 GNN / Spatio-Temporal Tracking
            │
            ├───────────────┐
            ▼               ▼
  Threat Trajectory    Atmospheric
                       Precursor Analysis
            │               │
            └───────┬───────┘
                    ▼
      Threat Transition Intelligence
                    │
                    ▼
       12 km → 5 km Localization
                    │
                    ▼
          Validation & Evaluation
                    │
                    ▼
              REST API
                    │
                    ▼
          GIS / Web Dashboard
                    │
                    ▼
          Localized Threat Alert
```

---

# 🚀 Core Innovation: Threat Transition Intelligence

The prescribed technologies in the problem statement — GNNs, anomaly detection, and AI-based downscaling — are established approaches and should not themselves be presented as the project's novelty.

Our differentiating layer is the **Threat Transition Intelligence Engine (TTIE)**.

Instead of treating an anomaly as a static pixel or isolated forecast value, the system represents it as a persistent **Threat Object**.

Each Threat Object maintains a lifecycle such as:

```text
DETECTION
    ↓
FORMATION
    ↓
GROWTH
    ↓
INTENSIFICATION
    ↓
PEAK
    ↓
DECAY
```

Additional lifecycle events can include:

```text
MERGER
SPLIT
WEAKENING
RELOCATION
```

For every time step, the system can maintain:

- Threat ID
- threat type
- severity
- geographic centroid
- spatial footprint
- movement direction
- movement speed
- intensity
- intensity growth rate
- persistence duration
- ensemble agreement
- atmospheric precursor features
- transition probability
- expected transition window
- uncertainty information

This changes the question from:

> **"Is there an extreme event?"**

to:

> **"What is this threat, how is it evolving, why is it evolving, where is it going, and what transition is it likely to make?"**

---

# 🎯 System Objectives

The project is designed around five major objectives.

### 1. Detect

Identify extreme weather anomalies relative to an appropriate climatological baseline.

### 2. Track

Maintain a persistent identity for an evolving threat across forecast timesteps.

### 3. Explain

Analyze atmospheric variables that accompany changes in the threat.

### 4. Predict Transition

Estimate the likelihood and expected time window of a transition toward a higher-severity state.

### 5. Localize

Convert a coarse-scale threat representation into a finer local impact footprint while evaluating whether important extremes are preserved.

---

# 🧠 Technical Architecture

## Stage 1 — Data Ingestion

The system accepts multidimensional meteorological data containing variables such as:

- temperature
- precipitation
- geopotential height
- sea-level pressure
- relative humidity
- specific humidity
- wind components
- vorticity
- other event-specific variables

Potential data sources include:

- ERA5 reanalysis
- IMDAA reanalysis
- NWP forecast products
- ensemble prediction system data
- historical extreme-event cases

Large NetCDF/GRIB datasets are processed using chunked and lazy-loading approaches.

### Main tools

```text
Python
xarray
Dask
NetCDF / GRIB2
NumPy
pandas
```

---

# 📊 Stage 2 — Climatological Baseline

Extreme conditions should not be identified only from absolute values.

A value that is extreme in one region or season may be normal in another.

Therefore, the system uses a historical climatological baseline to estimate anomalous conditions.

Conceptually:

```text
Forecast Value
      │
      ▼
Historical Climatology
      │
      ▼
Anomaly / Percentile / EFI-style Signal
      │
      ▼
Extreme Candidate
```

Possible approaches include:

- standardized anomaly / z-score
- percentile thresholds
- seasonal climatology
- rolling climatological statistics
- Extreme Forecast Index (EFI)-style comparison

The implementation can use the simplest validated detector first and progressively introduce more sophisticated metrics.

---

# 🔥 Stage 3 — Extreme Anomaly Detection

The anomaly detector converts continuous meteorological fields into candidate threat regions.

Example:

```text
Weather Field
     │
     ▼
Anomaly Calculation
     │
     ▼
Thresholding
     │
     ▼
Spatial Filtering
     │
     ▼
Connected Components / Clustering
     │
     ▼
Candidate Threat Regions
```

For each candidate region, the system calculates features such as:

```text
area
centroid
maximum intensity
mean intensity
shape
duration
growth
movement
```

The output is an anomaly mask and a set of candidate regions.

---

# 🌐 Stage 4 — Spatio-Temporal Representation

The weather field is inherently spatial and temporal.

Instead of treating every grid cell independently, the system represents relationships between neighboring spatial locations.

A graph representation can contain:

```text
Node
 ├── latitude
 ├── longitude
 ├── temperature
 ├── pressure
 ├── humidity
 ├── wind
 ├── precipitation
 └── anomaly features

Edge
 ├── spatial adjacency
 ├── geographic distance
 └── temporal relationship
```

For a spherical/global representation, the project can use a mesh-based graph architecture.

A GNN can then learn or represent:

```text
Local interactions
        +
Spatial structure
        +
Temporal evolution
        ↓
Threat representation
```

Possible technology:

```text
PyTorch
PyTorch Geometric / DGL
Graph Neural Networks
Message Passing
```

The exact GNN architecture should only be claimed as implemented after it has been trained and evaluated.

---

# 🛰️ Stage 5 — Persistent Threat Object

The core tracking abstraction is a **Threat Object**.

A detected anomaly receives a persistent identifier.

Example schema:

```json
{
  "threat_id": "THR-2026-0001",
  "type": "extreme_rainfall",
  "severity": "moderate",
  "timestamp": "2026-09-23T12:00:00Z",
  "centroid": {
    "lat": 13.08,
    "lon": 80.27
  },
  "footprint": {},
  "movement": {
    "direction_deg": 72,
    "speed_kmh": 18.4
  },
  "intensity": 0.71,
  "growth_rate": 0.18,
  "persistence_hours": 30,
  "ensemble_agreement": 0.78
}
```

> The numerical values above are illustrative schema examples, not claimed model outputs.

---

# 🧭 Stage 6 — Threat Tracking

The tracker associates candidate regions between consecutive timesteps.

A threat can be matched using combinations of:

- centroid distance
- overlap / IoU
- footprint similarity
- intensity similarity
- motion consistency
- temporal continuity

Conceptually:

```text
Threat(t)
   │
   ├── nearest candidate
   ├── footprint overlap
   ├── motion consistency
   └── intensity continuity
             │
             ▼
        Threat(t+1)
```

This produces a trajectory:

```text
T0 → T1 → T2 → T3 → T4
```

From this trajectory we derive:

- displacement
- velocity
- direction
- growth rate
- duration
- spatial expansion
- intensification / weakening

---

# 🌬️ Stage 7 — Atmospheric Precursor Analysis

The system should not only observe that a threat is becoming stronger.

It should investigate the atmospheric conditions accompanying that change.

Potential precursor variables include:

### Moisture

- relative humidity
- specific humidity
- moisture convergence
- precipitable water

### Pressure

- sea-level pressure
- pressure tendency
- geopotential height changes

### Temperature

- temperature anomaly
- temperature gradient
- vertical temperature structure

### Wind

- wind speed
- wind direction
- convergence
- vertical shear
- vorticity

These features form an explanatory state:

```text
Pressure tendency
       +
Moisture increase
       +
Wind convergence
       +
Temperature anomaly
       ↓
Threat evolution
```

The system can expose these signals through time-series plots.

---

# 🧠 Stage 8 — Threat Transition Intelligence Engine

This is the project's primary differentiating layer.

The engine receives:

```text
Threat history
      +
Current intensity
      +
Footprint evolution
      +
Trajectory
      +
Atmospheric precursors
      +
Ensemble uncertainty
```

and produces:

```text
Current state
      +
Transition probability
      +
Expected transition window
      +
Contributing signals
      +
Confidence / uncertainty
```

Example output format:

```json
{
  "current_state": "moderate",
  "target_state": "severe",
  "transition_probability": 0.76,
  "expected_window_hours": [8, 14],
  "drivers": [
    "increasing moisture",
    "pressure decrease",
    "wind convergence"
  ]
}
```

Again, these values are an example of the output format. Actual probabilities must come from the trained and validated model.

---

# 🧬 Threat Lifecycle Model

The dashboard can represent the lifecycle as:

```text
┌────────────┐
│  DETECTED  │
└─────┬──────┘
      ▼
┌────────────┐
│  FORMING   │
└─────┬──────┘
      ▼
┌────────────┐
│   GROWING  │
└─────┬──────┘
      ▼
┌────────────┐
│INTENSIFYING│
└─────┬──────┘
      ▼
┌────────────┐
│    PEAK    │
└─────┬──────┘
      ▼
┌────────────┐
│   DECAY    │
└────────────┘
```

This lifecycle representation is useful for both human interpretation and machine-readable alerting.

---

# 🔬 Stage 9 — 12 km → 5 km Localization / Downscaling

The SIH problem calls for localization from a coarse global forecast representation toward a finer local grid.

The intended pipeline is:

```text
12 km NWP Field
       │
       ▼
Threat Bounding Box
       │
       ▼
Local Crop
       │
       ▼
Downscaling Model
       │
       ▼
~5 km Local Field
       │
       ▼
Localized Threat Footprint
```

The project can use a staged implementation.

## Baseline

Start with a deterministic interpolation method such as:

- bilinear interpolation
- nearest-neighbor interpolation

This creates a reproducible baseline.

## Advanced model

If sufficient paired data and compute are available, investigate:

- conditional diffusion
- super-resolution networks
- generative downscaling
- physics-informed constraints

The advanced model must be compared against the baseline.

---

# ⚖️ Extreme Preservation

A major problem with ordinary spatial smoothing is that extreme values can be weakened.

Therefore, evaluation should not rely only on generic image similarity.

The system should measure whether important extremes survive downscaling.

Possible metrics include:

### Peak preservation

```text
Peak Preservation =
Fine-grid peak / Reference peak
```

### Extreme bias

```text
Extreme Bias =
Predicted extreme - Reference extreme
```

### Spatial overlap

Use:

- IoU
- Dice coefficient
- threat-footprint overlap

### Distribution metrics

Compare:

- percentile distributions
- upper-tail statistics
- mean absolute error
- RMSE

The final validation module should report a metric table rather than visually claiming that the downscaled output is better.

---

# 🧪 Stage 10 — Validation & Evaluation

Validation is performed at multiple levels.

## Detection

Measure:

- precision
- recall
- F1-score
- false alarm rate
- miss rate

## Tracking

Measure:

- centroid error
- trajectory error
- track continuity
- ID consistency
- duration error

## Transition intelligence

Measure:

- Brier score
- ROC-AUC where appropriate
- calibration
- transition detection lead time

## Downscaling

Measure:

- RMSE / MAE
- peak-value error
- extreme percentile error
- spatial overlap
- extreme preservation

## Operational validation

Evaluate:

```text
Detection
   ↓
Tracking
   ↓
Transition
   ↓
Localization
   ↓
Alert
```

A final validation gate can classify a pipeline stage as:

```text
PASS
DEGRADE
FAIL
```

based on predefined thresholds.

---

# 🌪️ Historical Event Replay

The system should be demonstrated on historical events rather than only synthetic examples.

Possible case studies include documented extreme-weather events for which suitable historical data are available.

The replay pipeline is:

```text
Historical Forecast
        │
        ▼
Run Detection
        │
        ▼
Create Threat Object
        │
        ▼
Track Event
        │
        ▼
Analyze Precursors
        │
        ▼
Estimate Transition
        │
        ▼
Localize Impact
        │
        ▼
Compare With Reference
```

This makes the demo reproducible.

---

# 🌐 Ensemble Uncertainty

Where ensemble forecast data are available, the system can use member-to-member variation to estimate uncertainty.

Instead of showing only:

```text
Threat trajectory
```

the dashboard can show:

```text
Central trajectory
       +
Possible trajectory corridor
       +
Ensemble agreement
```

This gives decision-makers information about forecast confidence rather than presenting a single deterministic path as certain.

---

# 🗺️ GIS Dashboard

The dashboard is designed around the questions:

> **WHERE?**  
> **WHERE IS IT GOING?**  
> **WHY IS IT CHANGING?**  
> **WHEN COULD IT ESCALATE?**  
> **HOW CERTAIN IS THE FORECAST?**

### What the dashboard actually draws

Built by
`python frontend/dashboard.py --all-events --with-folium --with-gnn`, which writes one
`data/processed/plots/dashboard/index.html` with every panel embedded, plus the PNGs.

| Layer | Source | Real? |
|---|---|---|
| ERA5 background field (MSLP for Amphan, 2 m temperature for the heatwave) | `data/raw/era5_<event>.nc` | real reanalysis |
| Coastlines, borders, land/ocean, labelled graticule | Cartopy Natural Earth | real geography |
| Detected anomaly rectangles (all frames faint, strongest frame red) | `data/processed/detection/anomalies_<event>.json` | real detector output |
| Tracked trajectory + strongest-anomaly marker | same file | real tracker output |
| Severity series and anomalous-cell count per frame | same file | measured |
| ST-GNN prediction vs ERA5 observation vs persistence on the 66-node mesh | `weights/gnn/outputs/real_era5/` | real ERA5 + real trained model |
| Interactive Folium map with box and trajectory popups | same detection file | real |

**Deliberately absent:** the uncertainty corridor, the localised impact zone, the
historical/reference footprint overlay and ensemble agreement. Nothing in the repository
computes them — `ensemble_agreement` is `null` end to end — so drawing them would mean
drawing invented data.

### Threat card

The card below is the **contract** ([`src/shared/contracts.py`](src/shared/contracts.py)).
Fields are rendered from the pipeline where it produces them and as `—` where it does not.
The tier line used to read `MODERATE`; it was removed because
`configs/tracking.yaml -> severity_bands` is `null`, so no severity tier is assigned to any
threat.

### Threat card

```text
┌────────────────────────────────────┐
│ THR-2026-0001                      │
│ Extreme Rainfall                   │
│                                    │
│ Severity: —  (severity_bands null) │
│ Movement: NE                       │
│ Speed: -- km/h                     │
│ Persistence: -- hours              │
│                                    │
│ Transition Risk: --                │
│ Expected Window: --                │
│ Ensemble Agreement: --             │
└────────────────────────────────────┘
```

Every displayed number comes from the pipeline. A field the pipeline does not compute is
shown as `—`, never estimated for the sake of a fuller card.

---

# 🔌 REST API

A lightweight API exposes machine-readable threat information.

Example endpoint structure:

```text
GET /api/v1/threats
GET /api/v1/threats/{threat_id}
GET /api/v1/threats/{threat_id}/trajectory
GET /api/v1/threats/{threat_id}/precursors
GET /api/v1/threats/{threat_id}/transition
GET /api/v1/threats/{threat_id}/footprint
GET /api/v1/alerts
```

Example response:

```json
{
  "threat_id": "THR-2026-0001",
  "type": "extreme_rainfall",
  "severity": "moderate",
  "location": {
    "latitude": 13.08,
    "longitude": 80.27
  },
  "trajectory": [],
  "transition": {
    "target_severity": "severe",
    "probability": null,
    "expected_window": null
  },
  "uncertainty": {
    "ensemble_agreement": null
  }
}
```

`null` values should be used until a corresponding model output is actually available.

---

# 🧱 Project Structure

Generated from `git ls-files`, so everything listed below is what a clone actually
receives. Derived artefacts under `data/processed/` and the ERA5-Land / north-India
climatology fetches are **not** tracked — see
[What ships in the repo](#what-ships-in-the-repo-and-what-does-not).

```text
threat-X/
│
├── README.md
├── .gitignore
├── .env.example                  # CDS API credential template
├── requirements.txt
├── environment.yml
├── pyproject.toml                # pytest addopts + ruff rule set
├── dataset.csv                   # flat 66-node CSV, 151 steps; the GNN's first
│                                 # training set. Its Date/Time columns cannot be
│                                 # parsed into one calendar (see 02_build_spatio-
│                                 # temporal_tensor.py), and its precipitation and
│                                 # pressure spreads are far from ERA5 -- which is
│                                 # why stage 09 retrains on real reanalysis.
├── inspect_dataset.py
│
├── team/                         # plan + per-member work cards
│   ├── README.md                 # pipeline, milestones, demo run of show, DoD
│   ├── Aravinth.md  Hariharan.md  Navaneedan.md
│   ├── Pushpa.md    Sachin.md     Varnika.md
│
├── configs/
│   ├── data.yaml                 # regions, events, variable sets
│   ├── model.yaml                # downscaler + calibration + GNN blocks
│   ├── tracking.yaml             # severity_bands: null (no tuned thresholds)
│   └── validation.yaml           # score gates: all null -> verdicts "undecided"
│
├── data/
│   ├── README.md
│   ├── raw/                      # era5_amphan.nc, era5_heatwave.nc      (tracked)
│   ├── climatology/              # 30 x era5_clim_<year>.nc, Bay of Bengal (tracked)
│   ├── processed/                # every derived artefact               (gitignored)
│   └── samples/                  # 6 fixture JSONs, also served by the API
│
├── src/                          # pipeline logic
│   ├── shared/
│   │   ├── contracts.py          # Threat Object schema shared by every layer
│   │   ├── config.py             # typed YAML config access
│   │   ├── demo.py               # the twelve-step demo, end to end
│   │   ├── fields.py  geo.py  metrics.py  synthetic.py
│   │   └── visualization.py
│   │
│   ├── data/
│   │   ├── cds_fetch.py          # ERA5 + per-region climatology fetch
│   │   ├── land_fetch.py         # ERA5-Land
│   │   ├── open_data_fetch.py    # GFS / IMDAA alternatives
│   │   ├── ibtracs.py            # cyclone best-track reference
│   │   └── loader.py
│   │
│   ├── detection/
│   │   └── anomaly_detection.py  # streaming z-score, region clustering, CLI
│   │
│   ├── tracking/
│   │   ├── tracker.py  trajectory.py
│   │   └── pipeline.py           # runnable stage: python -m src.tracking.pipeline
│   │
│   ├── precursors/
│   │   └── engine.py  pipeline.py  plots.py
│   │
│   ├── transition/
│   │   └── lifecycle.py          # deterministic threat state machine
│   │
│   ├── downscaling/              # deterministic rungs + real-pair harness
│   │   ├── downscaling.py  baseline.py  metrics.py
│   │   └── real_pair.py          # ERA5-Land vs ERA5 paired evaluation
│   │
│   ├── validation/
│   │   ├── detection_metrics.py  tracking_metrics.py
│   │   ├── transition_metrics.py
│   │   └── evaluation.py         # library -- no __main__
│   │
│   └── models/                   # learned models ONLY
│       ├── downscaling/super_resolution.py  # learned filter + quantile calibration
│       ├── gnn/                  # package home for learned graph modules
│       └── transition/transition_model.py
│
├── backend/                      # FastAPI service
│   ├── main.py  alert_api.py
│   ├── routers/                  # threats, trajectory, footprint, alerts,
│   │                             # precursors, transition
│   ├── schemas/
│   └── services/                 # read paths -- fixture-backed, not live-wired
│
├── frontend/
│   ├── dashboard.py              # real-map Cartopy panels + Folium + one index.html
│   └── visulisation.py           # the interactive Folium map on its own
│
├── weights/                      # trained artefacts (data, not source)
│   ├── gnn/                      # 01..09 scripts + committed ERA5 pilot outputs
│   │   ├── 09_train_real_era5.py # trains and scores the ST-GNN on real ERA5
│   │   └── run_gnn_tracking_demo.py
│   ├── anomaly/  downscaling/  transition/
│
├── tests/                        # 514 passing tests
│
└── docs/
    ├── architecture.md  dataset.md  dataset_sources.md
    ├── api.md  experiments.md  precursors.md
    └── transition.md  references.md
```

---

# 🛠️ Technology Stack

Split into **what the code actually imports** and **what was declared in the original plan
but is not installed**. Nothing in the second list is on the critical path; a few of them
are named in older sections of this document as future options.

## In use (installed by `requirements.txt`)

| Area | Libraries |
|---|---|
| Data & scientific computing | Python 3.11, NumPy, pandas, xarray, Dask, NetCDF4, h5netcdf, zarr, SciPy |
| Meteorological formats | cfgrib + eccodes (GRIB2), CDS API (ERA5), ECMWF open-data client, MetPy |
| Machine learning | scikit-learn (ridge/logistic models), scikit-image (connected components) |
| Geospatial | GeoPandas, Shapely, PyProj, haversine helpers in `src/shared/geo.py` |
| Visualization | matplotlib, Folium (interactive HTML map) |
| Backend | FastAPI, Pydantic, Uvicorn |
| Testing & quality | pytest, pytest-cov, httpx, Ruff |

## Declared in the plan, **not** installed

| Library | Why it is absent |
|---|---|
| PyTorch / PyTorch Geometric | Commented out in `requirements.txt`. The downscaler is ridge-fitted and the transition model is logistic regression, so nothing needs it. **Two GNN test files do** — see [Tests](#tests). |
| Cartopy, Rasterio | Conda-only on Windows; maps use Folium + haversine instead. |
| React / Plotly frontend | **Not built.** The dashboard is a static Folium HTML file. |
| Docker, GPU | No `Dockerfile` and no CUDA path in the repo. |
| IMDAA, ensemble (EPS) data | Unverified access; `ensemble_agreement` is `null` end to end. |

> Earlier revisions of this file listed all of the above as if they were in use. They are
> recorded here as intentions so the gap is visible rather than discovered mid-evaluation.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd threat-X
```

## 2. Create a Python environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Configuration

Create an environment file:

```bash
cp .env.example .env
```

Example:

```env
DATA_ROOT=./data
WEIGHTS_ROOT=./weights
API_HOST=0.0.0.0
API_PORT=8000
```

Do not commit:

- API keys
- private datasets
- credentials
- large raw weather archives
- trained model weights unless intentionally released

---

# ▶️ Running the Pipeline

> Every command below was executed while writing this README. Two commands that used to
> appear here — `python -m src.data.preprocessing` and `python -m src.detection.anomaly_detector` —
> refer to modules that do not exist; they have been replaced with the entry points that do.

## Fetch data (needs CDS credentials in a gitignored `.env`)

```bash
python -m src.data.cds_fetch --event amphan
python -m src.data.cds_fetch --climatology --all-regions --years 1991-2020
python -m src.data.land_fetch --event heatwave
```

## Run anomaly detection

```bash
python -m src.detection.anomaly_detection --event amphan --variable mean_sea_level_pressure
python -m src.detection.anomaly_detection --event heatwave --variable 2m_temperature
```

Writes `data/processed/detection/anomalies_<event>.json`. Each event is scored against
**its own** region's climatology; a baseline that does not cover 90 % of the event domain
is refused rather than reported as "no anomaly".

## Run threat tracking

```bash
python -m src.tracking.pipeline --input data/processed/detection/anomalies_amphan.json
```

## Run the atmospheric precursor analysis

```bash
python -m src.precursors.pipeline
```

## Run transition intelligence

```bash
python -m src.models.transition.transition_model
```

## Run downscaling (interpolation baseline)

```bash
python -m src.downscaling.real_pair --all-events
```

`python -m src.downscaling.baseline` runs the synthetic interface check instead — useful
for seeing the metric respond to resolution loss without any real data.

## Run downscaling (learned filter + quantile calibration)

```bash
python -m src.models.downscaling.super_resolution --all-events --sweep-radii 1,2,3,4
```

This fits the learned filter on the real ERA5 → ERA5-Land pair, scores it against the
interpolation baseline on the same holdout cells, and writes one artefact per event under
`data/processed/validation/downscaling/`. Measured results are in `docs/experiments.md`
(D2–D4).

## Generate evidence plots

```bash
python -m src.shared.visualization
```

## Build the GIS map

```bash
python -m src.detection.anomaly_detection --event amphan --variable mean_sea_level_pressure
python frontend/visulisation.py --event amphan
```

## Run validation

`src/validation/evaluation.py` is a **library**, not a command — it has no `__main__`
entry point. It is invoked by the demo (step 11), and its gate reports `undecided` for
every stage while `configs/validation.yaml` thresholds are `null`.

## Start API

```bash
uvicorn backend.main:app --reload
```

The API will be available locally at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
/docs
```

---

# 🧪 Reproducible Demo

The SIH demonstration should use a prepared historical case.

Recommended sequence:

```text
1. Select historical event
        ↓
2. Load forecast/reanalysis sample
        ↓
3. Display meteorological field
        ↓
4. Detect anomaly
        ↓
5. Generate Threat ID
        ↓
6. Show threat trajectory
        ↓
7. Show intensity + footprint evolution
        ↓
8. Show atmospheric precursor signals
        ↓
9. Show transition intelligence
        ↓
10. Show 12 km → local downscaling
        ↓
11. Show validation metrics
        ↓
12. Display final GIS alert
```

The sequence above is **executable**, not a script for a human to chain by hand:

```bash
python -m src.shared.demo --event amphan       # all twelve steps, real archives
python -m src.shared.demo --event heatwave     # all twelve steps, real archives
python -m src.shared.demo --event amphan --only 1,2,4
```

`src/shared/demo.py` runs each step against the local ERA5 / ERA5-Land archives and writes
its artefact to `data/processed/demo/<event>/` and its figures to
`data/processed/plots/demo/<event>/`. A step that cannot be computed on the data that is
actually present says so and is skipped — it is never replaced by a synthetic stand-in.
Both cases complete all twelve steps. Each event is scored against **its own** region's
1991–2020 climatology (`docs/dataset.md`), and the detector's `MIN_CLIMATOLOGY_COVERAGE`
guard stays armed: it refuses a baseline that does not cover the event instead of
reporting "no anomaly" where no comparison was possible.

A recorded/static fallback should be maintained in case live inference or external data access fails during the presentation.

---

# 📈 Example Threat Output

> ⚠️ **The JSON below is a shape illustration, not a pipeline output.** Every value in it is
> a placeholder written when the contract was drafted, including `"severity": "moderate"` —
> which the pipeline cannot emit, because `configs/tracking.yaml -> severity_bands` is
> `null`. The field list matches [`src/shared/contracts.py`](src/shared/contracts.py); the
> numbers do not come from anywhere.
>
> For records the pipeline **actually** produced, read:
>
> - `data/processed/detection/anomalies_<event>.json` — real detection + trajectory output
> - `weights/gnn/outputs/gnn_tracking_demo.json` — a real Threat ID derived from a
>   GNN-predicted field
> - `data/samples/threats.json` — the API fixture, whose own `_meta` marks it
>   *"NOT real model output"*

```json
{
  "threat_id": "THR-2026-0001",
  "type": "extreme_rainfall",
  "severity": "moderate",

  "location": {
    "centroid": [13.08, 80.27]
  },

  "movement": {
    "direction_deg": 72,
    "speed_kmh": 18.4
  },

  "evolution": {
    "intensity": 0.71,
    "growth_rate": 0.18,
    "persistence_hours": 30
  },

  "precursors": {
    "humidity": 0.84,
    "pressure_change": 0.62,
    "wind_convergence": 0.71
  },

  "transition": {
    "target": "severe",
    "probability": 0.76,
    "window_hours": [8, 14]
  },

  "uncertainty": {
    "ensemble_agreement": 0.78
  }
}
```

These numbers are **illustrative only**. The deployed system must populate them from actual model outputs.

---

# 🧩 Model Development Strategy

The project should be developed incrementally.

## Phase 1 — Working Baseline

Implement:

```text
Data loading
    ↓
Climatology
    ↓
Z-score / percentile anomaly
    ↓
Spatial clustering
    ↓
Threat object
    ↓
Centroid tracking
    ↓
Basic dashboard
```

This establishes a complete end-to-end system early.

## Phase 2 — Spatio-Temporal Intelligence

Add:

```text
Graph representation
    ↓
GNN
    ↓
Improved trajectory representation
```

## Phase 3 — Transition Intelligence

Add:

```text
Threat history
+
Atmospheric precursors
+
Ensemble features
        ↓
Transition model
```

## Phase 4 — Fine Localization

Start with:

```text
12 km → interpolation → local grid
```

Then evaluate:

```text
12 km → learned downscaling
```

Only retain the advanced model if it demonstrates measurable improvement.

## Phase 5 — Operational Integration

Connect:

```text
Models
 ↓
API
 ↓
GIS dashboard
 ↓
Alert format
```

---

# 🛡️ Physics-Aware Constraints

For advanced downscaling or transition models, purely statistical predictions can produce physically implausible states.

Where appropriate, physical constraints can be incorporated into the objective function.

Conceptually:

```text
Total Loss =
Data Loss
+
Spatial Loss
+
Temporal Loss
+
Extreme Preservation Loss
+
Physics Loss
```

Possible physical consistency checks include relationships involving:

- moisture
- pressure
- wind
- temperature
- precipitation

The exact constraints should be selected according to the modeled variables and available observations.

---

# 📚 Datasets

Potential datasets include:

### ERA5

Used as a historical atmospheric reanalysis and climatological reference.

### IMDAA

Indian-region atmospheric reanalysis suitable for regional historical analysis.

### NWP / Ensemble Forecast Data

Used to represent the medium-range forecast input.

### Historical Extreme Events

Selected cases should have sufficient observations/reanalysis/forecast information to support reproducible evaluation.

> Dataset availability, licensing, spatial resolution, temporal resolution, ensemble configuration, and access restrictions must be verified for the exact version used by the team.

---

# 🔍 Why This Architecture?

The architecture separates the problem into interpretable components.

```text
Detection
```

answers:

> Is there an anomaly?

```text
Tracking
```

answers:

> Is it the same threat over time?

```text
Precursor Analysis
```

answers:

> What atmospheric signals accompany its evolution?

```text
Transition Intelligence
```

answers:

> What state could it transition toward and when?

```text
Downscaling
```

answers:

> Where is the localized high-resolution impact?

```text
Validation
```

answers:

> How accurate is the system?

This separation also makes the system easier to debug and evaluate.

---

# 🏆 What Makes the Project Different?

The project's differentiation is not simply:

```text
GNN
+
Diffusion
+
Weather Data
```

because those are components of the SIH problem and established research directions.

The proposed differentiation is the **persistent threat lifecycle abstraction**:

```text
Anomaly
   ↓
Threat Object
   ↓
Trajectory
   ↓
Evolution
   ↓
Precursors
   ↓
Transition
   ↓
Localized Impact
   ↓
Uncertainty-Aware Alert
```

The system therefore attempts to connect **detection → understanding → transition → localization → decision support** in one pipeline.

---

# 📊 Key Performance Indicators

The project should report measurable results rather than only visual demonstrations.

| Component | Example KPI |
|---|---|
| Anomaly Detection | Precision / Recall / F1 |
| Threat Tracking | Centroid Error |
| Trajectory | Position / displacement error |
| Persistence | Track continuity |
| Transition Model | Brier Score / calibration |
| Transition Timing | Lead-time error |
| Downscaling | RMSE / MAE |
| Extreme Preservation | Peak / upper-tail error |
| Spatial Localization | IoU / Dice |
| API | Response latency |
| Pipeline | End-to-end inference time |

Final values should be populated only after experiments are completed.

---

# ⚠️ Limitations

The system has several important limitations.

### Data availability

High-resolution paired forecast/reference datasets may be difficult to obtain.

### Rare events

Extreme events are naturally imbalanced, making model training and evaluation difficult.

### Forecast uncertainty

Medium-range predictions become increasingly uncertain with lead time.

### Downscaling uncertainty

A higher-resolution output does not automatically mean higher forecast accuracy.

### Physical consistency

Generative models can produce statistically plausible but physically inconsistent states unless appropriately constrained.

### Computational requirements

Large 4D weather datasets can require substantial storage, memory, and GPU compute.

### Generalization

A model trained on one type of extreme event may not generalize to a different event type or geographic region.

---

# 🔧 Risk Mitigation

| Risk | Mitigation |
|---|---|
| Large datasets | xarray + Dask + chunking |
| Limited GPU | Start with classical baselines |
| Missing high-resolution data | Use interpolation baseline |
| GNN training difficulty | Implement graph representation + baseline tracker first |
| Diffusion too expensive | Compare against deterministic downscaling |
| Rare-event imbalance | Event-based sampling / appropriate metrics |
| Live API failure | Prepared demo data |
| External data unavailable | Local historical sample |
| False confidence | Display uncertainty / ensemble agreement |
| Model not sufficiently accurate | Retain validated baseline rather than overclaiming |

---

# 🔒 Responsible Use

This system is intended as a **research and decision-support prototype**, not as an autonomous replacement for official meteorological warning systems.

Forecast probabilities and localized threat footprints should be interpreted with their uncertainty and validation context.

The project should avoid:

- presenting experimental predictions as official warnings
- inventing probability values
- hiding uncertainty
- claiming operational accuracy without evaluation
- claiming a model is trained when only the architecture exists

---

# 👥 Team Responsibilities

| Member | Primary Responsibility | Secondary Responsibility |
|---|---|---|
| **Navaneedan** | 12 km → 5 km Downscaling | Validation + Transition Intelligence |
| **Aravinth** | 12 km → 5 km Downscaling | Weather Data + Preprocessing |
| **Hariharan** | Atmospheric Precursor Analysis | FastAPI / REST API |
| **Sachin** | Threat Tracking + Trajectory | GIS Dashboard + PPT Architecture |
| **Pushpa** | Extreme Anomaly Detection | Weather Data + Preprocessing |
| **Varnika** | GNN + Spatio-Temporal Modeling | Historical Event Analysis + Visualization |

---

# 🎤 SIH Demo Narrative

The final demonstration should tell one continuous story.

### 1. Detect

> "The system receives a medium-range weather forecast and identifies an abnormal region."

### 2. Track

> "Instead of treating this as an isolated anomaly, we create a persistent Threat Object and track it through time."

### 3. Understand

> "We monitor its intensity, footprint, movement, and atmospheric precursor signals."

### 4. Transition

> "The Threat Transition Intelligence layer estimates whether the threat is evolving toward a higher-severity state and provides an expected time window."

### 5. Localize

> "The identified threat region is passed to the localization/downscaling stage to obtain a finer-scale impact footprint."

### 6. Validate

> "We compare the result against a reference and report quantitative metrics."

### 7. Alert

> "The final result is exposed through an API and visualized on a GIS dashboard."

---

# 🧭 Final End-to-End Pipeline

```text
                    ┌──────────────────────────┐
                    │ NWP / Ensemble Forecast │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Data Preprocessing       │
                    │ xarray + Dask            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Climatological Baseline │
                    │ ERA5 / IMDAA             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Anomaly Detection       │
                    │ Z-score / Percentile    │
                    │ / EFI-style features    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Threat Object Creation  │
                    │ Persistent Threat ID    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    ▼                          ▼
          ┌─────────────────┐        ┌──────────────────┐
          │ GNN / Tracking  │        │ Precursor Engine │
          │ trajectory      │        │ P/T/RH/Wind etc. │
          └────────┬────────┘        └─────────┬────────┘
                   │                           │
                   └────────────┬──────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ Threat Transition       │
                    │ Intelligence             │
                    │ lifecycle + probability  │
                    │ + expected window        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ 12 km → Local Grid      │
                    │ Downscaling / Baseline  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Validation               │
                    │ Accuracy + Extremes     │
                    │ + Spatial Overlap       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ FastAPI REST Layer       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ GIS Dashboard            │
                    │ Map + Trajectory         │
                    │ Threat + Uncertainty     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Localized Threat Alert  │
                    └──────────────────────────┘
```

---

# 📖 References & Research

The implementation should maintain a dedicated research log and cite the exact versions of datasets, libraries, and papers used.

Important research directions include:

1. **Extreme Forecast Index (EFI)** — ECMWF methodology for identifying forecast distributions that depart from a model climate distribution.
2. **Ensemble Prediction Systems** — probabilistic weather forecasting and uncertainty representation.
3. **Object-based weather-event tracking** — including TITAN-style storm/object tracking.
4. **Graph-based weather forecasting** — graph and mesh representations for atmospheric dynamics.
5. **Diffusion-based weather downscaling** — conditional generative approaches for high-resolution weather fields.
6. **Extreme-event evaluation** — metrics designed for rare and high-impact weather events.
7. **Indian atmospheric datasets** — ERA5, IMDAA, NWP and ensemble forecast products relevant to the target region.

Representative research resources:

- ECMWF Extreme Forecast Index documentation
- Lalaurette, C. (2003), Extreme Forecast Index methodology
- TITAN object-based storm tracking literature
- GraphCast / graph-based global weather forecasting research
- GenCast probabilistic weather forecasting research
- Recent diffusion-based weather downscaling literature
- IMD / NCMRWF documentation and operational forecasting material

Exact citations and dataset links should be maintained in `docs/references.md`.

---

# 📝 Research Integrity

This repository distinguishes between:

### Implemented

A component that exists in the codebase and has been tested.

### Experimental

A component under development or being evaluated.

### Planned

A component described by the architecture but not yet implemented.

### Target

A requirement from the SIH problem statement that the project intends to address.

This distinction prevents the README from claiming capabilities that the prototype has not yet demonstrated.

### Where each stage stands today

| Stage | Status | Evidence |
|---|---|---|
| Data acquisition (ERA5, ERA5-Land, IBTrACS, GFS/IMDAA) | **Implemented** | `src/data/`, tracked `era5_amphan.nc` / `era5_heatwave.nc` |
| Climatological baseline | **Implemented** | 30 tracked Bay-of-Bengal files + 30 fetched north-India files |
| Extreme anomaly detection | **Implemented** | `src/detection/anomaly_detection.py`, `anomalies_*.json` artefacts |
| Spatio-temporal representation | **Implemented** | `src/shared/fields.py`, threat-object contracts |
| Persistent threat object + tracking | **Implemented** | `src/tracking/`, `tests/test_tracking*.py` |
| Atmospheric precursor analysis | **Implemented** | `src/precursors/`, `docs/precursors.md` |
| Transition intelligence | **Implemented, gate withholding** | Brier scores exist; 3 of 7 horizons fail the publishing gate |
| 12 km baseline localization | **Implemented** | `src/downscaling/` |
| Learned 5 km downscaling | **Experimental** | spatial-holdout numbers below; no independent high-res reference |
| Extreme-preservation metric | **Implemented** | peak ratio, exceedance rate, tail RMSE |
| Validation & evaluation | **Implemented, gates unconfigured** | every `configs/validation.yaml` gate is `null` -> verdict `undecided` |
| GNN spatio-temporal model, frozen checkpoint | **Implemented, scale mismatch measured** | `08_evaluate_era5_pilot.py` scores it against persistence on the held-out week and records the training/pilot mismatch in `domain_diagnostics.csv` |
| GNN spatio-temporal model on real ERA5 | **Implemented — beats persistence on all five features** | `weights/gnn/09_train_real_era5.py`, 1824 training hours, held-out pilot week, `real_era5_metrics.csv` |
| GNN → detection → tracking bridge | **Implemented** | `weights/gnn/run_gnn_tracking_demo.py` writes `gnn_tracking_demo.json` + a tracked Threat ID |
| REST API | **Implemented, fixture-backed** | routers run; read paths serve `data/samples/`, not the pipeline |
| GIS dashboard | **Implemented** | `frontend/dashboard.py` -> real-map Cartopy panels + Folium + one `index.html` |
| Web dashboard (React) | **Not started** | planned only |
| Historical event replay | **Implemented** | `weights/gnn/04_historical_event_replay.py`, 12-step demo |
| Recorded fallback demo, six-slide deck | **Not started** | delivery artefacts, no code |

---

# 🧪 Definition of Done

Status against the checklist the team set for itself. "Done" means the command in
[Start Here](#-start-here) runs and the number in
[the experiment log](docs/experiments.md) reproduces it.

- [x] Reproducible weather-data sample — tracked ERA5 + six API fixtures
- [x] Working preprocessing pipeline — `src/data/loader.py`, region-scoped climatology
- [x] Working anomaly detector — 48 frames / 51 regions (Amphan), 80 / 30 (heatwave)
- [x] Candidate anomaly mask — region boxes + cell counts in `anomalies_*.json`
- [x] Persistent Threat ID — `src/tracking/tracker.py`
- [x] Threat trajectory — `trajectory` field, `tests/test_trajectory.py`
- [x] Threat footprint evolution — footprint endpoint + fixtures
- [x] Atmospheric precursor features — `src/precursors/engine.py`
- [x] Transition model or validated transition baseline — deterministic baseline scored
- [x] Transition probability calibration/evaluation — Brier + skill computed; **3 of 7 horizons withheld**
- [x] 12 km baseline localization — bilinear reference, RMSE 1.5703 K
- [x] Advanced downscaling experiment — learned filter, spatial holdout, radius + alpha sweep
- [x] Extreme-preservation metric — peak, exceedance, p95/p99 tail RMSE
- [x] Validation table — `docs/experiments.md` rows D1–D5, DET1–DET2, T1–T2, S1
- [x] REST API — `backend/main.py` runs, 6 routers
- [x] Real-map dashboard — Cartopy ERA5 panels + Folium layer, one `index.html`, no API key required
- [x] GNN staged on a real ERA5 background — 1824 training hours, the pilot week held out, **beats persistence on all five features**
- [x] GNN → detection → tracking bridge — `gnn_tracking_demo.json` yields a tracked Threat ID
- [x] PyTorch + PyTorch Geometric installed and pinned — the two GNN test files now run
- [x] Historical event replay — both events run 12/12 locally
- [x] Research/reference documentation — `docs/`
- [ ] Recorded fallback demo — the dashboard is ready to screen-record; the recording itself is not made
- [ ] Final six-slide SIH presentation — not started
- [ ] Severity bands / validation gates filled in — deliberately `null`, needs a justified threshold
- [ ] Detection precision & recall — no reference labels exist to score against
- [ ] API read paths wired to the pipeline — `/health` reports stage flags as `False`
- [ ] Live React web dashboard — not started

---

# ⭐ Project Vision

The long-term goal is to move from:

```text
Forecast
   ↓
Generic Warning
```

toward:

```text
Forecast
   ↓
Detect
   ↓
Track
   ↓
Understand
   ↓
Predict Transition
   ↓
Localize
   ↓
Quantify Uncertainty
   ↓
Deliver Actionable Threat Information
```

The central idea is simple:

> **An extreme weather event should not be treated as a static point on a map. It should be treated as a dynamic threat with a lifecycle.**

---

## Team

**Team: Bots** — Smart India Hackathon 2026

**Project:** AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

**Per-member responsibilities, deliverable checklists and PPT duties:** see [`team/`](team/)

**Theme:** Smart Automation

**Category:** Software

---

## Status

🚧 **Research & Development Prototype — evaluable end to end**

Everything in [Start Here](#-start-here) runs from a clone, the full test suite passes
(**514 passed**), and every measured number is reproducible from a command in
this file. The dashboard at `data/processed/plots/dashboard/index.html` is the artefact to
screen-record for a demo.

The repository is updated continuously as individual modules move from planned →
experimental → implemented → validated. Sections that still describe a *target* rather than
a capability are marked as such; where a result is negative it is reported as negative
(see the transition model and the frozen GNN checkpoint).

---

## License

Add the project's selected open-source license here once the team finalizes the repository licensing strategy.

---

**Built for SIH 2026 with a focus on reproducibility, measurable validation, uncertainty awareness, and interpretable extreme-weather threat intelligence.**
