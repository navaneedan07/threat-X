# 🌦️ AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies

> **Threat-X** — Smart India Hackathon 2026  
> **Theme:** Smart Automation  
> **Category:** Software  
> **Organization:** Ministry of Earth Sciences (MoES)

An end-to-end AI system for detecting, tracking, explaining, and localizing extreme-weather anomalies in medium-range numerical weather prediction (NWP) forecasts.

The system moves beyond a static "extreme weather detected" alert. It creates a **persistent Threat Object** for every detected anomaly and follows its lifecycle across space and time — measuring its movement, intensity, footprint, persistence, atmospheric precursors, and transition toward higher severity.

The intended output is a **hyper-local, uncertainty-aware weather threat layer** that can support earlier and more targeted warnings.

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

### Main map layers

- anomaly field
- threat footprint
- threat centroid
- trajectory
- uncertainty corridor
- localized impact zone
- administrative boundaries
- historical/reference footprint

### Threat card

```text
┌────────────────────────────────────┐
│ THR-2026-0001                      │
│ Extreme Rainfall                   │
│                                    │
│ Severity: MODERATE                 │
│ Movement: NE                       │
│ Speed: -- km/h                     │
│ Persistence: -- hours              │
│                                    │
│ Transition Risk: --                │
│ Expected Window: --                │
│ Ensemble Agreement: --             │
└────────────────────────────────────┘
```

All displayed numerical values should come from the actual inference pipeline.

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

# 🧱 Recommended Project Structure

```text
threat-X/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── docker-compose.yml
├── .env.example
│
├── team/                          # locked plan + per-member work cards
│   ├── README.md                 # pipeline, milestones, demo run of show, DoD
│   ├── Aravinth.md
│   ├── Hariharan.md
│   ├── Navaneedan.md
│   ├── Pushpa.md
│   ├── Sachin.md
│   └── Varnika.md
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── tracking.yaml
│   └── validation.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── climatology/
│   └── samples/
│
├── frontend/                     # UI layer -- GIS map, threat cards, timeline
│   ├── components/
│   ├── pages/
│   └── maps/
│
├── backend/                      # service layer -- FastAPI
│   ├── main.py
│   ├── routers/
│   └── schemas/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_climatology.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_tracking.ipynb
│   ├── 05_precursors.ipynb
│   ├── 06_transition_intelligence.ipynb
│   └── 07_validation.ipynb
│
├── src/                          # pipeline logic
│   ├── shared/                   # contracts + helpers used across layers
│   │   ├── contracts.py          # Threat Object schema (shared by all layers)
│   │   ├── geo.py
│   │   ├── logging.py
│   │   └── visualization.py
│   │
│   ├── data/
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── climatology.py
│   │
│   ├── detection/
│   │   ├── anomaly_detector.py
│   │   ├── thresholding.py
│   │   └── clustering.py
│   │
│   ├── tracking/
│   │   ├── threat_object.py
│   │   ├── tracker.py
│   │   └── trajectory.py
│   │
│   ├── precursors/
│   │   ├── feature_engineering.py
│   │   └── analysis.py
│   │
│   ├── transition/
│   │   ├── lifecycle.py          # deterministic state machine
│   │   └── uncertainty.py
│   │
│   ├── downscaling/
│   │   ├── baseline.py           # deterministic interpolation
│   │   └── metrics.py
│   │
│   ├── validation/
│   │   ├── detection_metrics.py
│   │   ├── tracking_metrics.py
│   │   ├── transition_metrics.py
│   │   └── evaluation.py
│   │
│   └── models/                   # learned models ONLY (ML + DL)
│       ├── gnn/
│       │   ├── mesh.py
│       │   ├── graph_builder.py
│       │   └── gnn.py
│       ├── downscaling/
│       │   └── diffusion.py
│       └── transition/
│           └── transition_model.py
│
├── weights/                      # trained artefacts (data, not source)
│   ├── anomaly/
│   ├── gnn/
│   ├── transition/
│   └── downscaling/
│
├── tests/
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_transition.py
│   └── test_validation.py
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── api.md
    ├── experiments.md
    └── references.md
```

---

# 🛠️ Technology Stack

## Data & Scientific Computing

```text
Python
NumPy
pandas
xarray
Dask
NetCDF
GRIB2
```

## Meteorological Processing

```text
MetPy
ERA5
IMDAA
NWP / EPS data
```

## Machine Learning

```text
PyTorch
scikit-learn
PyTorch Geometric / DGL
```

## Advanced Modeling

```text
Graph Neural Networks
Conditional Diffusion
Spatio-Temporal Modeling
Probabilistic Forecasting
```

## Geospatial

```text
GeoPandas
Shapely
Cartopy
Rasterio
```

## Backend

```text
FastAPI
Pydantic
Uvicorn
```

## Frontend / Visualization

```text
React
Leaflet / MapLibre / equivalent GIS library
Plotly / equivalent charting library
```

## Infrastructure

```text
Docker
Git
GitHub
GPU acceleration where available
```

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

## Run preprocessing

```bash
python -m src.data.preprocessing
```

## Run anomaly detection

```bash
python -m src.detection.anomaly_detector
```

## Run threat tracking

```bash
python -m src.tracking.tracker
```

## Run transition intelligence

```bash
python -m src.models.transition.transition_model
```

## Run downscaling (interpolation baseline)

```bash
python -m src.downscaling.baseline
```

## Run validation

```bash
python -m src.validation.evaluation
```

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

A recorded/static fallback should be maintained in case live inference or external data access fails during the presentation.

---

# 📈 Example Threat Output

A conceptual threat record:

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

---

# 🧪 Definition of Done

The project should not be considered complete until the team can demonstrate:

- [ ] Reproducible weather-data sample
- [ ] Working preprocessing pipeline
- [ ] Working anomaly detector
- [ ] Candidate anomaly mask
- [ ] Persistent Threat ID
- [ ] Threat trajectory
- [ ] Threat footprint evolution
- [ ] Atmospheric precursor features
- [ ] Transition model or validated transition baseline
- [ ] Transition probability calibration/evaluation
- [ ] 12 km baseline localization
- [ ] Advanced downscaling experiment, if feasible
- [ ] Extreme-preservation metric
- [ ] Validation table
- [ ] REST API
- [ ] GIS dashboard
- [ ] Historical event replay
- [ ] Recorded fallback demo
- [ ] Final six-slide SIH presentation
- [ ] Research/reference documentation

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

**Threat-X — Smart India Hackathon 2026**

**Project:** AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

**Per-member responsibilities, deliverable checklists and PPT duties:** see [`team/`](team/)

**Theme:** Smart Automation

**Category:** Software

---

## Status

🚧 **Research & Development Prototype**

The repository should be updated continuously as individual modules move from planned → experimental → implemented → validated.

---

## License

Add the project's selected open-source license here once the team finalizes the repository licensing strategy.

---

**Built for SIH 2026 with a focus on reproducibility, measurable validation, uncertainty awareness, and interpretable extreme-weather threat intelligence.**
