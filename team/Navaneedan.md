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

| Order | Artifact | Path | Status |
|---|---|---|---|
| 1 | Threat Object schema | `src/shared/contracts.py` | **done** — `contract_drift()` returns `[]` |
| 2 | Synthetic field generator | `src/shared/synthetic.py` | **done** |
| 3 | Interpolation baseline | `src/downscaling/baseline.py` | **done** |
| 4 | Extreme-preservation metrics | `src/downscaling/metrics.py` | **done** |
| 5 | Gate + lifecycle | `src/validation/evaluation.py`, `src/transition/lifecycle.py` | **done** |
| 6 | Transition model | `src/models/transition/transition_model.py` | **done** — trained and scored on 2 real events; no usable skill yet |
| 7 | Deterministic downscaling entry point | `src/downscaling/downscaling.py` | **done** — adapter over `baseline.py`; no hard-coded factor, no noise |
| 8 | Detection + tracking metrics | `src/validation/detection_metrics.py`, `src/validation/tracking_metrics.py` | **done** — implemented and tested; no reference labels yet, so nothing measured |
| 9 | Evidence plots | `src/shared/visualization.py` | **done** — coarse-vs-refined + lifecycle figures generated |
| 10 | Real coarse/fine pair + experiment | `src/data/land_fetch.py`, `src/downscaling/real_pair.py` | **done** — ERA5 0.25° → ERA5-Land 0.10°, scored (D1); resolutions confirmed in `configs/data.yaml` |

All ten are implemented, tested (438 tests, `438 passed` in `.venv`, ruff clean on these
files) and documented. **What is left is blocked on someone else, or on a decision that
is yours:**

| Remaining work | Blocked on |
|---|---|
| Learned downscaler (`src/models/downscaling/`) | **unblocked** — resolutions confirmed (0.25 → 0.10); it now needs building and scoring against the interpolation baseline on `src/downscaling/metrics.py`, not on visual appeal |
| Validation gate verdicts | thresholds must be justified from measured metric distributions. **Do not invent them** — a null threshold correctly yields "undecided" |
| Detection / tracking metric **values** | need reference masks/tracks (Pushpa / Sachin); the modules themselves are done |
| Slides 1–6 and the three demo segments | nothing — this is the remaining unblocked work |

**Resolved in the last pass:** `detection_metrics.py` and `tracking_metrics.py` are
implemented and tested; `/transition` reads the stored report and serves the shortest
publishable horizon (withheld horizons stay `null`); and `src/downscaling/downscaling.py`
is now a deterministic adapter over `baseline.py` with the hard-coded 2.4 factor and the
random noise removed. The real coarse/fine pair (ERA5 0.25° → ERA5-Land 0.10°) is
fetched, measured and scored (D1), and `configs/data.yaml` carries the confirmed
resolutions — which unblocks the learned downscaler.

---

## Evidence you must save

- [x] Downscaled field — `data/processed/validation/downscaling/heatwave_t2m.json` (real ERA5 0.25° → ERA5-Land 0.10°), plus the explicitly refused `amphan_tp.json`
- [x] Coarse vs refined visual comparison — `python -m src.shared.visualization` (synthetic pair, shared colour scale)
- [x] Extreme-preservation metric table — `tests/test_downscaling.py` + `docs/experiments.md` (peak preserved 1.00 at 2x, lost 0.62 at 10x)
- [x] Transition model output with a real probability — `data/processed/validation/<threat_id>/transition_report.{json,md}`, tables in `docs/experiments.md`
- [x] Validation metric table + gate verdict — the table is real; the **verdict is deliberately `undecided`** while every threshold is `null` (`docs/experiments.md`)
- [x] Threat lifecycle visual — `python -m src.shared.visualization` (state sequence + intensity, deterministic state machine)

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
- [x] Coarse input pipeline works
- [x] Interpolation baseline works (bilinear + nearest)
- [x] Target grid is a parameter, not a hard-coded factor (`upscale_factor` is derived; 12 → 5 gives 2.4)
- [x] Coarse-vs-refined plot created (real ERA5 → ERA5-Land pair, plus the synthetic interface check)
- [x] Real coarse/fine resolutions confirmed from measured grids and written to `configs/data.yaml` (0.25 / 0.10)
- [x] Extreme-preservation metric calculated
- [x] Metric tested against a known-failure case (a smoothed field is flagged as a failure)
- [ ] Learned baseline tested
- [ ] Diffusion investigated (only if feasible)
- [x] Results saved — plots + reports under `data/processed/` (gitignored) and the code that regenerates them

### Threat Transition Intelligence
- [x] Threat Object schema created (with Sachin) — `src/shared/contracts.py`
- [x] Lifecycle defined — `src/transition/lifecycle.py`, threshold-free rules
- [x] Transition target defined explicitly — `docs/transition.md`, written before training
- [x] Transition dataset prepared — one per event; pooling events is refused
- [x] Transition model implemented
- [x] Real probability output generated
- [x] Model evaluated (Brier + calibration + base rate) — out-of-fold, both events
- [x] Expected transition window implemented
- [x] Horizons 6/12/18/24h handled where data supports (6 h Amphan auto-refused: 3 positives)
- [x] Publishing gate — a fitted horizon is withheld unless it beats both the base rate and chance (`docs/transition.md` §7)
- [x] API serves published probabilities — `transition_service` serves the shortest publishable horizon; withheld horizons stay `null`

### Validation & Integration
- [x] Validation metrics implemented for transition + downscaling
- [x] `detection_metrics.py` / `tracking_metrics.py` written (with Pushpa / Sachin — the modules exist; values need reference labels)
- [x] PASS/DEGRADE/SUPPRESS rule defined — `src/validation/evaluation.py`
- [ ] Gate thresholds justified in `docs/experiments.md` (not guessed) — all 8 still `null`, so every verdict is correctly `undecided`
- [ ] Historical validation completed
- [x] Threat Object contract frozen and in sync with `configs/tracking.yaml`
- [ ] Threat Object connected to all modules — blocked while `src/tracking/` produces nothing
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
