# NAVANEEDAN — THREAT-X

> **Primary:** 12 km → ~5 km Downscaling
> **Secondary:** Validation & Evaluation · Threat Transition Intelligence · End-to-end integration
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 3 (with Sachin), 7 Transition, 8 Downscaling (with Aravinth), 9 Validation (with Pushpa) |
| **Milestones** | M3, M6 (with Hariharan), M7 (with Aravinth), M8 (with Pushpa), M10 (lead) |
| **Main artifact** | Downscaled field · coarse-vs-refined comparison · extreme-preservation metric table · transition output · validation gate |
| **PPT** | Slide 1 (with Sachin), Slide 2 (lead with Sachin), Slide 3, Slide 4 (with Hariharan), Slide 5 (with Sachin + Varnika), Slide 6 |
| **Demo segments** | 125–150s transition · 150–175s localize/downscale · 175–195s validate + alert |
| **Stack** | Python, PyTorch, Diffusers, Xarray, NumPy, Dask, CUDA, Cartopy/GeoPandas, scikit-learn |

You are the **integration owner**. Downscaling is your headline, but you also own the layer
that makes the project defensible — transition intelligence and the validation gate.

---

## 1. Downscaling — what to actually build

**Rule: do not start with the most expensive model.** Build in this order and stop at the
first rung that works within the time available.

### Step 1 — Interpolation baseline (`src/downscaling/baseline.py`)

- Take a coarse `xarray.DataArray` and a **target grid** as parameters.
- Implement bilinear (`order=1`) and nearest-neighbour (`order=0`) resampling.
- **Do not hard-code the upscale factor.** `configs/data.yaml` has
  `coarse_resolution_deg` and `fine_resolution_deg` as `null` because the dataset is not
  confirmed. Derive the factor from the target grid so the module works the day Aravinth
  confirms the resolutions — no rewrite.
- **Exit condition:** runs on any `DataArray`, including synthetic ones. Testable without data.
- **Reads:** `configs/model.yaml` → `downscaling.interpolation.order`

### Step 2 — Extreme-preservation metrics (`src/downscaling/metrics.py`)

This is the module that makes your claim defensible. Generic image similarity is **not**
sufficient — a bilinear baseline that preserves the peak can legitimately beat a learned
model that smooths it away.

| Metric | Definition |
|---|---|
| Peak preservation | `fine_peak / reference_peak` |
| Extreme bias | `predicted_extreme − reference_extreme` |
| Threshold exceedance | fraction of cells above the extreme threshold, before vs after |
| Percentile error | error at the **95th** and **99th** percentiles |
| Spatial overlap | IoU and Dice on thresholded masks |
| RMSE / MAE | whole-field error, for completeness |

- **Exit condition:** a test case where interpolation *obviously* fails must actually be
  flagged as a failure. If your metric passes a smoothed field, the metric is wrong.
- **Why it matters:** this table is your slide 5 evidence and your gate input.

### Step 3 — Learned baseline (`src/models/downscaling/`)

Only after Steps 1–2 are stable **and** Aravinth has paired coarse/fine data.

- CNN / super-resolution first. Conditional diffusion **only** if time, data and compute permit.
- Score against the interpolation baseline on the metrics from Step 2, not on visual appeal.
- **Retain the baseline as the fallback** and report both. If the learned model loses on
  peak preservation, say so.

---

## 2. Threat Transition Intelligence — what to actually build

### Step 1 — Lifecycle (`src/transition/lifecycle.py`)

Define the deterministic state machine:

```text
FORMATION → INTENSIFICATION → EXPANSION → PEAK → DECAY
```

Plus the events from `configs/tracking.yaml`: `MERGER`, `SPLIT`, `WEAKENING`, `RELOCATION`.
Input is a threat history; output is a state label per time step. No training needed —
this is a rule-based state machine and is fully testable with synthetic histories.

### Step 2 — Transition output (`src/models/transition/transition_model.py`)

- Horizons: **6 / 12 / 18 / 24 hours**, "where the data supports it".
- Output per the README's format: current state, target state, transition probability,
  expected window in hours, contributing drivers, confidence.
- Baseline model first: logistic regression on Hariharan's precursor features is enough.
- **Target definition must be explicit** and written down before training.
- **Exit condition:** probability comes from a **tested model**, never a hard-coded number.

### Step 3 — Uncertainty (`src/transition/uncertainty.py`)

Feed `ensemble_agreement` onto the Threat Object. If ensemble data is unavailable, the field
stays `null` and the dashboard renders "—".

---

## 3. Validation and gating (`src/validation/`)

| File | Metrics |
|---|---|
| `detection_metrics.py` | precision, recall, F1, false alarm rate, miss rate *(with Pushpa)* |
| `tracking_metrics.py` | centroid error, trajectory error, track continuity, ID consistency, duration error |
| `transition_metrics.py` | Brier score, ROC-AUC, calibration, lead-time error |
| `evaluation.py` | the gate + the final metric table |

### The gate

Implement **PASS / DEGRADE / SUPPRESS** based on actual metrics, uncertainty and model
availability.

**The thresholds stay `null`** in `configs/validation.yaml` until real metric distributions
have been measured. Justify them in `docs/experiments.md`, never invent them — a gate with a
guessed threshold launders a guess into an apparent pass.

Remember the base rate: extreme events are rare, so a great Brier score can come from
predicting "no transition" everywhere. Always report the base rate alongside it.

---

## 4. Integration — you own the contracts

- Define module **input/output contracts** and keep them in `src/shared/contracts.py`.
- The **Threat Object** is the load-bearing one: Sachin, Hariharan and Varnika all serialize
  it. Field list lives in `configs/tracking.yaml` → `threat_object_fields`.
- Build the **Threat Object schema at M3**, with Sachin. Four people are blocked on it.
- Fold GNN, tracking, precursor and downscaling outputs into a single threat state.

> ⚠️ **Unblocked work starts here.** The schema, the interpolation baseline, the metrics and
> the lifecycle state machine need **nothing** from anyone else. See "Start here" below.

---

## Start here (no dependencies)

| Order | Artifact | Path | Needs from others |
|---|---|---|---|
| 1 | Threat Object schema | `src/shared/contracts.py` | **nothing** |
| 2 | Synthetic field generator | `src/shared/synthetic.py` | **nothing** |
| 3 | Interpolation baseline | `src/downscaling/baseline.py` | **nothing** |
| 4 | Extreme-preservation metrics | `src/downscaling/metrics.py` | **nothing** |
| 5 | Gate + lifecycle | `src/validation/evaluation.py`, `src/transition/lifecycle.py` | **nothing** |

Items 1–5 are testable with synthetic arrays today. Blocked: learned downscaler (needs
paired data → Aravinth), trained transition model (needs precursors → Hariharan, threat
history → Sachin), historical validation (needs the chosen case).

---

## Evidence you must save

- [ ] Downscaled field (saved array/file, not a screenshot of one)
- [ ] Coarse vs refined visual comparison
- [ ] Extreme-preservation metric table
- [ ] Transition model output with a real probability
- [ ] Validation metric table + gate verdict
- [ ] Threat lifecycle visual

---

## PPT contribution

| Slide | Your input |
|---|---|
| 1 TITLE | Co-lead. PS ID, exact title, theme/category, Team ID/Name. **No architecture dump.** |
| 2 IDEA | **Lead.** The differentiator: persistent Threat Object + lifecycle + transition intelligence. Problem visual → Detect → Track → Localize → Validate → Alert. |
| 3 TECHNICAL | Transition + downscaling + validation blocks, with actual baseline/model status |
| 4 FEASIBILITY | Co-owner with Hariharan. Fallbacks and compute risk for learned downscaling/diffusion |
| 5 IMPACT | Co-owner. Quantitative benefits: peak preservation, spatial overlap, transition calibration |
| 6 REFERENCES | Downscaling/validation references — add rows to `docs/references.md` before citing |

**Required evidence (from the team checklist):** threat lifecycle visual · transition output ·
downscaling comparison · validation metric/gate.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 125–150s | Transition intelligence | Severity transition probability + expected window |
| 150–175s | Localize/downscale (with Aravinth) | 12 km → ~5 km field + extreme-preservation metric |
| 175–195s | Validate + alert (with Hariharan, Sachin) | PASS/DEGRADE + JSON alert + map |

**You must show escalation output and coarse-vs-refined evidence.** Have both on disk and
openable offline.

---

## Fallback

| If this fails | Fall back to |
|---|---|
| Learned / diffusion downscaler | Interpolation baseline (already built) — report it as the delivered method |
| Transition model unfitted | Deterministic lifecycle state machine + `null` probability |
| Metric not measurable | Report the metric as unavailable; never substitute an estimate |
| Live API or data access fails | Recorded/static demo assets in `data/samples/` |

---

## Definition of done

- **DOWNSCALING:** baseline exists; coarse/refined comparison and extreme-preservation metric exist
- **TRANSITION:** target definition explicit; probability produced by a **tested model**, not a hard-coded number
- **VALIDATION:** at least one quantitative metric reported; degraded cases visible
- Plus the team-wide rule: *the artifact runs from a reproducible input, produces a saved
  output, has at least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Downscaling
- [ ] Coarse input pipeline works
- [ ] Interpolation baseline works (bilinear + nearest)
- [ ] Target grid is a parameter, not a hard-coded factor
- [ ] Coarse-vs-refined plot created
- [ ] Extreme-preservation metric calculated
- [ ] Metric tested against a known-failure case
- [ ] Learned baseline tested
- [ ] Diffusion investigated (only if feasible)
- [ ] Results saved

### Threat Transition Intelligence
- [ ] Threat Object schema created (with Sachin)
- [ ] Lifecycle defined
- [ ] Transition target defined explicitly
- [ ] Transition dataset prepared
- [ ] Transition model implemented
- [ ] Real probability output generated
- [ ] Model evaluated (Brier + calibration + base rate)
- [ ] Expected transition window implemented
- [ ] Horizons 6/12/18/24h handled where data supports

### Validation & Integration
- [ ] Validation metrics implemented (detection/tracking/transition/downscaling)
- [ ] PASS/DEGRADE/SUPPRESS rule defined
- [ ] Gate thresholds justified in `docs/experiments.md` (not guessed)
- [ ] Historical validation completed
- [ ] Threat Object connected to all modules
- [ ] API receives final state

### PPT
- [ ] Slide 2 content + differentiator visual supplied
- [ ] Slide 3 transition/downscaling/validation blocks supplied
- [ ] Slide 4 feasibility/fallback supplied
- [ ] Slide 5 quantitative metrics supplied
- [ ] Slide 6 references supplied
- [ ] No invented numbers on any slide

### Demo
- [ ] Transition demo works
- [ ] Downscaling demo works
- [ ] Validation gate result works
- [ ] Fallback prepared and tested
