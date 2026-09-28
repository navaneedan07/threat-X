# Experiments

The experiment log, and the only place validation gate thresholds may be
justified.

> **STATUS: experiments have been run on real ERA5 data, and the transition
> result is negative.** S1 below is a synthetic interface check; T1, T2 and D1 are
> real-data runs. Do not fill any remaining row with an estimate.

---

## Why this file exists

`configs/validation.yaml` defines PASS / DEGRADE / SUPPRESS gates but leaves
`pass`, `degrade` and `suppress` as `null`. They stay `null` until a real metric
distribution has been measured on historical events and the threshold is written
up in [Gate thresholds](#gate-thresholds).

A gate with an invented threshold is worse than no gate: it launders a guess into
an apparent pass.

---

## Rules

1. One row per experiment. Never overwrite a previous result.
2. Record the config, the data version and the seed — a number without its config
   is not reproducible.
3. Report the metric you actually measured, including a metric that came out badly.
4. Never report a metric from a component that did not run.
5. Never tune a threshold on the same events used to report the score.

---

## Experiment log

| # | Date | Stage | Config | Data version | Seed | Metric | Result | Notes |
|---|---|---|---|---|---|---|---|---|
| S1 | 2026-09-25 | Downscaling | `model.yaml` `interpolation.order: 1` | **SYNTHETIC only** — `src/shared/synthetic.py::sharp_peak_field`, no real data | deterministic | peak preservation | 1.00 at 2× coarsening · 0.97 at 4× · 0.62 at 10× | **Interface check, not a result.** Establishes that the metric responds to resolution loss and gives the interpolation baseline's operating range before any learned model is compared against it. The metric also flags a deliberately smoothed field (`smoothed_field`) at 0.62 against a 0.95 bar. Replace with a real-data row before quoting any of this. |
| T1 | 2026-09-25 | Transition | `escalation_quantile: 0.75`, horizons 6/12/18/24 h | ERA5 `era5_amphan` (2020-05-16→21), 48 steps @ 3 h, 0.25° | deterministic | Brier · skill · AUC · base rate | **skill −0.47 … +0.36** across horizons | Severity is a **proxy** (precipitation rate at centroid), not tracked-threat escalation. 27–33 rows, 3–9 positives. 6 h horizon refused (3 positives). **At 12 h the model is worse than always predicting the base rate.** |
| T2 | 2026-09-25 | Transition | `escalation_quantile: 0.75`, horizons 6/12/18/24 h | ERA5 `era5_heatwave` (2022-05-01→10), 80 steps @ 3 h, 0.25° | deterministic | Brier · skill · AUC · base rate | **skill +0.01 … +0.09** across horizons | Severity **proxy** (2 m temperature anomaly). 53–57 rows. Base rate reaches **86.8 % at 24 h**, so the label is nearly degenerate — a model answering "always escalates" scores well while discriminating nothing. **No useful skill.** |

| D1 | 2026-09-28 | Downscaling | `model.yaml` `interpolation.order: 1` | ERA5 `era5_heatwave` (0.25°) + ERA5-Land `era5_land_heatwave` (0.10°), frame 2022-05-01T09:00 | deterministic | extreme-preservation table | **peak preserved 1.007 · RMSE 2.36 K · IoU 0.049 · Dice 0.094** | **First real coarse/fine pair** (factor 2.5). ERA5 → ERA5-Land is a *different run*, not truth. Coarse is warmer at the peak (+2.16 K), so a peak ratio of 1.0 does **not** mean the baseline is good — the footprint overlap at p99 is 0.049 and exceedance is ~20× the reference. Amphan is **not scoreable**: ERA5-Land is land-only and 70 % of the cyclone domain is ocean, so the reference misses the extreme. |

---

## Detection

Chain: weather field -> climatology difference -> threshold mask -> candidate region.

| Metric | Definition | Target | Measured |
|---|---|---|---|
| Precision | correct candidates / all candidates | *unset* | — |
| Recall | correct candidates / all true events | *unset* | — |
| F1 | harmonic mean of precision and recall | *unset* | — |
| False alarm rate | false candidates / all candidates | *unset* | — |
| Miss rate | missed events / all true events | *unset* | — |

The metric module is implemented (`src/validation/detection_metrics.py`): cell-level
precision/recall/F1 plus false-alarm and miss rates, with an undefined denominator
reported as `null` rather than a misleading `1.0`. No reference masks exist yet, so
**no value is measured** and the columns above stay empty until they do.

Also required: a threshold-sensitivity sweep. Record how precision and recall move
as the z-score or percentile threshold changes, so the final choice is defensible.

---

## Tracking

| Metric | Definition | Target | Measured |
|---|---|---|---|
| Centroid error (km) | mean distance between tracked and reference centroid | *unset* | — |
| Trajectory error (km) | mean position error over the whole track | *unset* | — |
| Track continuity | fraction of timesteps with a maintained ID | *unset* | — |
| ID consistency | fraction of tracks with no spurious ID switch | *unset* | — |
| Duration error (h) | error in total threat lifetime | *unset* | — |

The metric module is implemented (`src/validation/tracking_metrics.py`): it matches
predicted to reference tracks within a required `match_radius_km` (there is no
default — a defaulted tolerance would be a guessed threshold in every score) and
reports centroid/trajectory error, track continuity, ID consistency and duration
error. No paired reference tracks exist yet, so **no value is measured**.

---

## Transition intelligence

**Target is a severity proxy — see [transition.md](transition.md) §2.** No
tracked-threat severity series exists in the repo yet, so "escalation" means the
event's own severity field crossing its p75. These numbers describe the proxy.

### Amphan / `THR-2020-0001` — precipitation proxy

| H | Rows | Positives | Base rate | Brier | **Skill** | AUC | Calibration |
|---|---|---|---|---|---|---|---|
| 6 h | 33 | 3 | 9.1 % | *untrained* | *untrained* | *untrained* | *untrained* |
| 12 h | 31 | 5 | 16.1 % | 0.1986 | **−0.4683** | 0.7769 | 0.2211 |
| 18 h | 29 | 7 | 24.1 % | 0.1878 | **−0.0253** | 0.7987 | 0.2096 |
| 24 h | 27 | 9 | 33.3 % | 0.1423 | **+0.3596** | 0.8827 | 0.1496 |

### Heatwave / `THR-2022-0001` — 2 m temperature anomaly proxy

| H | Rows | Positives | Base rate | Brier | **Skill** | AUC | Calibration |
|---|---|---|---|---|---|---|---|
| 6 h | 57 | 17 | 29.8 % | 0.1940 | **+0.0730** | 0.7235 | 0.1030 |
| 12 h | 56 | 33 | 58.9 % | 0.2395 | **+0.0106** | 0.6285 | 0.2427 |
| 18 h | 55 | 46 | 83.6 % | 0.1333 | **+0.0261** | 0.7971 | 0.1079 |
| 24 h | 53 | 46 | 86.8 % | 0.1042 | **+0.0910** | 0.7640 | 0.0585 |

**Publishing gate (`transition.md` §7).** A fitted horizon is served only if it
beats both the base-rate forecast (`skill > 0`) and chance (`AUC > 0.5`). The gate
is a floor, not a validation — it cannot tell whether the proxy target is the
right target:

| Event | Publishable | Withheld, and why |
|---|---|---|
| Amphan | 24 h only | 6 h (3 positives, minimum 5); 12 h and 18 h (skill ≤ 0) |
| Heatwave | 6, 12, 18, 24 h | — |

**Reading these honestly:**

1. **The model does not work.** Brier skill is within ±0.09 of the constant
   base-rate forecast everywhere except Amphan at 24 h (+0.36 on 27 rows, which is
   not enough to trust).
2. **Amphan at 12 h is actively harmful** — skill −0.47. Its Brier score of 0.199
   looks respectable in isolation, and is worse than predicting "no escalation"
   for every row. This is the exact trap the base-rate rule exists to catch.
3. **The heatwave label is nearly degenerate** — 86.8 % of rows escalate within
   24 h, so "always yes" scores well while discriminating nothing. AUC stays in a
   plausible-looking 0.63–0.80 range, which is why AUC alone must never be quoted.
4. **Metrics are out-of-fold** (5-fold `cross_val_predict`), so these are not the
   model grading its own homework.
5. **Features and label come from the same series**, so part of any apparent skill
   is autocorrelation rather than a physical precursor relationship.

**Conclusion:** with two events and ~27–57 rows per horizon, no credible
transition model can be trained. This matches the assessment in
[transition.md](transition.md) §6 and is now demonstrated with real numbers
rather than predicted. The machinery is correct; the data is not sufficient.

Artefacts: `data/processed/validation/<threat_id>/transition_report.{json,md}` —
**local only**, because `data/processed/**` is gitignored. These tables are the
committed record. Reproduce with `python -m src.models.transition.transition_model`,
which trains **one dataset per event**: the escalation boundary is a quantile of
each event's own severity, so pooling events would label one against another's
boundary and the dataset builder refuses it.

Extreme events are rare, so a good score can come from predicting "no transition"
everywhere. Always report the base rate alongside Brier and AUC.

Lifecycle artefact: `python -m src.shared.visualization` also writes the lifecycle
timeline (`data/processed/plots/lifecycle/threat_lifecycle.png`) from the
deterministic state machine (`src/transition/lifecycle.py`). It visualises the
state sequence and the intensity series the rules read — no trained model is
involved, and unevaluated events are named in the figure.

---

## Downscaling

Generic image metrics are not sufficient — **extreme preservation is the point**.

| Metric | Definition | Target | Measured |
|---|---|---|---|
| RMSE | root mean squared error vs reference | *unset* | — |
| MAE | mean absolute error | *unset* | — |
| Peak preservation | fine-grid peak / reference peak | *unset* | — |
| Extreme bias | predicted extreme − reference extreme | *unset* | — |
| IoU | threat-footprint overlap | *unset* | — |
| Dice | footprint overlap, Dice coefficient | *unset* | — |
| Percentile error | error in the upper-tail percentiles | *unset* | — |

A plain bilinear baseline that preserves the peak can legitimately beat a learned
model that smooths it away. Retain whichever wins on these metrics, and report
both.

### Real pair (D1)

`python -m src.downscaling.real_pair --all-events` scores the baseline on the
ERA5 (0.25°) → ERA5-Land (0.10°) pair, one artefact per event under
`data/processed/validation/downscaling/`. The heatwave result, masked to the
reference's footprint so both fields peak over the same area:

| Metric | Value |
|---|---|
| Peak preservation | 1.007 |
| Extreme bias | +2.16 K |
| RMSE | 2.36 K |
| MAE | 2.01 K |
| IoU (p99) | 0.049 |
| Dice (p99) | 0.094 |
| Exceedance change | +0.187 (predicted 19.7 % vs reference 1.0 %) |
| Percentile error (p95/p99) | 2.94 K / 2.68 K |

Within the same real field, decimating the reference by 2/4/8 and refining back
retains the peak (0.999 / 0.999 / 0.998) — the heatwave peak is broad enough that
2.5× coarsening does not resolve it. **Read together:** the peak ratio alone would
look like a pass while the spatial overlap says otherwise, which is why both are
reported.

`amphan_tp.json` records the **refusal**, not a number: ERA5-Land is land-only and
the Bay of Bengal domain is ~70 % ocean, so the coarse peak sits outside the
reference's footprint. That is the coverage guard working.

Coarse-vs-refined figures: `data/processed/plots/downscaling/real_heatwave_coarse_vs_refined.png`
(real) and `.../coarse_vs_refined.png` (synthetic interface check, generated by
`python -m src.shared.visualization`).

---

## Gate thresholds

**Fill this in only after the metrics above have real values.**

| Stage | Primary metric | PASS | DEGRADE | SUPPRESS | Justification |
|---|---|---|---|---|---|
| Detection | F1 | *unset* | *unset* | *unset* | — |
| Tracking | track continuity | *unset* | *unset* | *unset* | — |
| Transition | Brier score | *unset* | *unset* | *unset* | — |
| Downscaling | peak preservation | *unset* | *unset* | *unset* | Not settable from a single real pair (D1). Peak preservation (1.007) and footprint overlap (IoU 0.049) disagree, so any one threshold would encode a choice we cannot yet justify. |

The chosen values must be copied into `configs/validation.yaml`. The
justification column must explain **why** that number, referencing the measured
distribution — e.g. "baseline interpolation reaches 0.91, so PASS at 0.90 is the
level the baseline already meets".

---

## Reporting

The final validation run must emit a metric table as an artefact, not a claim in
prose. Save to the path in `configs/validation.yaml`
(`reporting.output_dir`), and reference the file when the result is quoted in the
README, the PPT or the demo.
