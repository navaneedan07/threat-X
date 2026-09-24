# Threat-X — Team Plan

> **One locked working document** for implementation, evidence, the six-slide PPT,
> the 3–5 minute demo and final integration.
>
> Derived from the *SIH26078 Final Team Workflow & PPT Contribution Plan* (13 pages).
> Per-member work cards live in this folder — see [Work cards](#work-cards).

**Project:** AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts
**Team:** Navaneedan · Aravinth · Hariharan · Sachin · Pushpa · Varnika
**Theme:** Smart Automation · **Category:** Software · **Organisation:** Ministry of Earth Sciences (MoES)

---

## Core rule

> **Every member must produce an actual artifact** — code, experiment, dataset/preprocessing
> output, model result, diagram, API, UI, validation result, reference matrix or demo asset.

A component with no saved output is not done, regardless of how much was discussed.

---

## 1. What we are actually building

**Problem-statement core:** detect extreme-weather anomalies in medium-range forecasts, track
them across time, localize the threat footprint, preserve extreme amplitudes during refinement,
validate the result, and expose a decision-facing alert.

**Team-added intelligence layer:** turn each detected anomaly into a **persistent Threat
Object** with an identity, evolving state, trajectory, atmospheric precursor context and a
transition estimate toward higher severity.

### Do NOT present as novelty

GNNs, anomaly detection, basic tracking and 12 km → ~5 km diffusion/downscaling are either
**prescribed by the problem statement or established research**. The defensible contribution is
their **integration into a persistent threat lifecycle and a transition-decision layer**.

### The judge-facing story

| Question | Answered by |
|---|---|
| WHERE is the threat? | Detection + tracking |
| WHERE is it going? | Trajectory |
| WHY is it evolving? | Precursor analysis |
| WHEN could it intensify? | Threat Transition Intelligence |
| HOW CERTAIN is the estimate? | Ensemble agreement / calibration |
| WHAT precise area should receive the alert? | Downscaling + validation gate |

---

## 2. End-to-end pipeline and stage ownership

| # | Stage | Output | Owner |
|---|---|---|---|
| 1 | DATA | ERA5 / IMDAA / selected forecast case | Aravinth + Pushpa |
| 2 | ANOMALY | Climatology comparison → anomaly mask / candidate region | Pushpa |
| 3 | THREAT OBJECT | Persistent Threat ID + state | Sachin + Navaneedan |
| 4 | GNN / SPATIO-TEMPORAL | Graph/mesh representation and temporal context | Varnika |
| 5 | TRACKING | Centroid, footprint, velocity, direction, duration | Sachin |
| 6 | PRECURSORS | Pressure, moisture, temperature, wind, vorticity/gradients | Hariharan |
| 7 | TRANSITION | Formation → intensification → expansion → peak → decay; escalation probability/window | Navaneedan |
| 8 | DOWNSCALING | 12 km → ~5 km localized field + extreme preservation | Navaneedan + Aravinth |
| 9 | VALIDATION | Spatial/intensity/extreme/transition metrics + PASS/DEGRADE | Navaneedan + Pushpa |
| 10 | API + GIS | JSON alert + map + trajectory + localized footprint | Hariharan + Sachin |

---

## 3. Final team allocation

| Member | Primary | Secondary / Support | Main artifact | PPT responsibility |
|---|---|---|---|---|
| **Navaneedan** | 12 km → ~5 km Downscaling | Validation & Evaluation + Threat Transition Intelligence | Downscaled field, extreme-preservation metrics, validation table, transition output | Slides 2–6 + final technical integration |
| **Aravinth** | 12 km → ~5 km Downscaling | Weather Data & Preprocessing | Paired coarse/fine data, preprocessing, baseline downscaler, event samples | Slides 2–6: data/downscaling evidence |
| **Hariharan** | Atmospheric Precursor Analysis | Backend & REST API | Precursor features/plots, FastAPI endpoints, JSON schema, integration tests | Slides 2–6: explainability/API/deployment |
| **Sachin** | Threat Tracking & Trajectory | Interactive Dashboard/UI + PPT/Architecture | Persistent tracker, trajectories, GIS dashboard, architecture visuals | Slides 1–6: visual system, architecture, dashboard, demo |
| **Pushpa** | Extreme Weather Anomaly Detection | Weather Data & Preprocessing | Anomaly detector, masks, clusters, thresholds, event maps | Slides 2–6: anomaly evidence/references |
| **Varnika** | GNN & Spatio-Temporal Modelling | Historical Event Analysis + Visualization Support | Graph/mesh prototype, GNN evidence, event replay/comparison plots | Slides 2–6: GNN/event/research visuals |

> **Ownership clarification:** downscaling is intentionally **co-owned** by Navaneedan + Aravinth.
> Varnika owns the GNN / spatio-temporal track. Sachin owns threat tracking and the visual layer.

---

## 4. Six-slide PPT — ownership

The supplied SIH template is preserved exactly as the presentation structure.

| # | Template slide | Lead owner(s) | What must appear |
|---|---|---|---|
| 1 | TITLE PAGE | Sachin + Navaneedan | PS ID, exact title, theme/category, Team ID/Name; clean title visual; **no architecture dump** |
| 2 | IDEA TITLE / PROPOSED SOLUTION | Navaneedan + Sachin | Problem visual → Detect → Track → Localize → Validate → Alert. Highlight persistent Threat Object, transition intelligence, extreme-preservation checks, localized decisioning |
| 3 | TECHNICAL APPROACH | Sachin (diagram) + all technical owners | One readable pipeline: Data → anomaly → spherical/GNN → Threat Object/tracking → precursor → transition → downscaling → validation → API/GIS. Show outputs and **actual** model/baseline status |
| 4 | FEASIBILITY & VIABILITY | Hariharan + Navaneedan | Data/compute assumptions, risks, mitigation, fallbacks. Baseline PoC first; advanced GNN/diffusion only where implemented/tested |
| 5 | IMPACT & BENEFITS | Sachin + Navaneedan + Varnika | Dashboard/evidence visual. Measurable prototype metrics: spatial overlap, peak preservation, false alarms, precision/recall, calibration or latency **where available** |
| 6 | RESEARCH & REFERENCES | Varnika + Pushpa | Compact, high-value references. Each technical claim has a source or is clearly labelled as our proposed design |

### PPT discipline

- **No invented benchmark numbers.**
- No "first / only / no prior art" claim without evidence.
- Advanced components are labelled **implemented** only after actual testing.
- Every slide must show what the team can **defend in judge Q&A**.

### Per-member PPT evidence checklist

| Member | Required PPT evidence |
|---|---|
| Navaneedan | Threat lifecycle visual; transition output; downscaling comparison; validation metric/gate |
| Aravinth | Dataset/data-flow evidence; coarse/fine pair; baseline comparison; experiment settings |
| Hariharan | Precursor plot; API schema/JSON; deployment/data-flow block; degraded mode |
| Sachin | Architecture diagram; trajectory map; dashboard screenshot; final visual system |
| Pushpa | Anomaly mask; threshold explanation; event map; validation/reference evidence |
| Varnika | GNN mesh diagram; historical replay; comparison plots; final reference matrix |

---

## 5. Build order and dependencies

**Do not start with the most expensive model.**

| Milestone | Stage | Owner(s) | Exit condition |
|---|---|---|---|
| M1 | Data + historical case | Aravinth + Pushpa | Clean inputs, climatology baseline, selected event, reproducible loader |
| M2 | Anomaly detection | Pushpa | Mask + candidate regions + threshold configuration |
| M3 | Threat identity + tracking | Sachin + Navaneedan | Persistent IDs, centroid/footprint/velocity/duration |
| M4 | GNN representation | Varnika | Graph/mesh prototype and interface to threat state |
| M5 | Precursor features | Hariharan | Feature table + explanatory plots |
| M6 | Transition intelligence | Navaneedan + Hariharan | Transition dataset + probability/window output |
| M7 | Downscaling baseline | Navaneedan + Aravinth | Interpolation/learned baseline + extreme-preservation metric |
| M8 | Validation + gating | Navaneedan + Pushpa | Metric table + PASS/DEGRADE/SUPPRESS |
| M9 | API + dashboard | Hariharan + Sachin | JSON alert + map + trajectory + localized footprint |
| M10 | Integration + PPT + demo | All; led by Sachin/Navaneedan/Varnika | One deterministic end-to-end case + six-slide evidence |

### Fallback hierarchy

```text
classical anomaly detector
  → deterministic threat tracker
    → simple calibrated transition model
      → interpolation / learned downscaling baseline
        → advanced GNN / diffusion as upgrades
```

**The demo must work without depending on the most expensive component.**

---

## 6. Demo — final run of show (3–5 minutes)

Use a **frozen historical event** for the primary demo; keep a recorded/static fallback.

| Time | Action | Owner | Judge sees |
|---|---|---|---|
| 00–20s | Select case | Sachin / Varnika | Prepared historical extreme-weather event / forecast replay |
| 20–45s | Load data + climatology | Aravinth / Pushpa | Raw/coarse field + baseline climatology |
| 45–70s | Detect anomaly | Pushpa | Anomaly mask + candidate region |
| 70–95s | Create/track threat | Sachin / Varnika | Persistent Threat ID + path + centroid + footprint |
| 95–125s | Explain evolution | Hariharan | Precursor feature plot + change over time |
| 125–150s | Transition intelligence | Navaneedan | Severity transition probability + expected window |
| 150–175s | Localize/downscale | Navaneedan / Aravinth | 12 km → ~5 km field + extreme-preservation metric |
| 175–195s | Validate + alert | Navaneedan / Hariharan / Sachin | PASS/DEGRADE + JSON alert + map |

**Closing sentence:**

> "The system is not only identifying an extreme region; it maintains a threat identity,
> explains its evolution, estimates its transition risk, verifies the localized field and
> exposes a precise alert."

---

## 7. Team-wide definition of done

| Module | Definition of done |
|---|---|
| DATA | Script/notebook loads the same case reproducibly; variables/time window documented |
| ANOMALY | Mask and candidate region saved; threshold/normalization choices documented |
| THREAT TRACKING | Threat ID persists across time; centroid, footprint, direction, velocity, duration saved |
| GNN | Graph/mesh construction demonstrable; trained claims have actual metrics |
| PRECURSORS | Feature table and at least one explanatory plot saved |
| TRANSITION | Target definition explicit; probability produced by a tested model, **not a hard-coded number** |
| DOWNSCALING | Baseline exists; coarse/refined comparison and extreme-preservation metric exist |
| VALIDATION | At least one quantitative metric reported; degraded cases visible |
| API | One documented JSON response works from model output to endpoint |
| GIS | Map shows threat, trajectory and localized footprint |
| PPT | Every technical claim is supported by an artifact or reference |
| DEMO | One deterministic path runs end-to-end; recorded/static fallback exists |

### Critical integrity rule

> **Never fabricate probabilities, metrics, training results or operational performance.**
> If a component is a prototype, baseline or conceptual architecture, **label it that way**.

Status vocabulary used throughout this repo:

| Label | Meaning |
|---|---|
| **Implemented** | Exists in the codebase and has been tested |
| **Experimental** | Under development or being evaluated |
| **Planned** | Described by the architecture, not yet written |
| **Target** | A requirement from the problem statement we intend to address |

---

## Where things live

| Path | Contents |
|---|---|
| `src/data/` | Data loading, preprocessing, climatology |
| `src/detection/` | Anomaly detector, thresholding, clustering |
| `src/tracking/` | Threat Object, tracker, trajectory |
| `src/models/gnn/` | Graph/mesh construction and GNN |
| `src/precursors/` | Precursor feature engineering and analysis |
| `src/transition/` | Lifecycle state machine, uncertainty |
| `src/models/transition/` | Learned transition model |
| `src/downscaling/` | Interpolation baseline + metrics |
| `src/models/downscaling/` | Learned super-resolution / diffusion |
| `src/validation/` | Metric modules + PASS/DEGRADE/SUPPRESS gate |
| `src/shared/` | Threat Object contract, geo, logging, plotting |
| `backend/` | FastAPI service, routers, Pydantic schemas |
| `frontend/` | React + GIS dashboard |
| `weights/` | Trained artefacts (code lives in `src/models/`) |
| `configs/` | All parameters — never hard-code them |
| `docs/` | Architecture, dataset, API, experiments, references |

---

## Work cards

| Member | Card |
|---|---|
| Navaneedan | [Navaneedan.md](Navaneedan.md) |
| Aravinth | [Aravinth.md](Aravinth.md) |
| Hariharan | [Hariharan.md](Hariharan.md) |
| Sachin | [Sachin.md](Sachin.md) |
| Pushpa | [Pushpa.md](Pushpa.md) |
| Varnika | [Varnika.md](Varnika.md) |
