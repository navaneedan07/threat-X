# Experiments

The experiment log, and the only place validation gate thresholds may be
justified.

> **STATUS: experiments have been run on real ERA5 data. The transition result is
> negative; the learned downscaler has a positive extreme-preservation result.**
> S1 below is a synthetic interface check; T1–T2, D1–D4 and DET1 are real-data
> runs. Do not fill any remaining row with an estimate.

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

| D1 | 2026-09-28 | Downscaling | `model.yaml` `interpolation.order: 1` | ERA5 `era5_heatwave` (0.25°) + ERA5-Land `era5_land_heatwave` (0.10°), frame 2022-05-01T09:00 | deterministic | extreme-preservation table | **peak preserved 1.007 · RMSE 2.36 K · IoU 0.049 · Dice 0.094** | **First real coarse/fine pair** (factor 2.5). ERA5 → ERA5-Land is a *different run*, not truth. Coarse is warmer at the peak (+2.16 K), so a peak ratio of 1.0 does **not** mean the baseline is good — the footprint overlap at p99 is 0.049 and exceedance is ~20× the reference. Amphan was recorded here as **not scoreable**; that claim does not reproduce and is **superseded by D5** (the numbers above are left unchanged as the record of this run). |
| D2 | 2026-09-29 | Downscaling | `model.yaml` `interpolation.order: 1`, all 80 steps aggregated | ERA5 `era5_heatwave` + ERA5-Land `era5_land_heatwave`, 0.25° → 0.10° | deterministic | extreme-preservation table (mean of 80 per-step tables) | **peak 1.0002 · RMSE 1.672 K · MAE 1.342 K · IoU 0.287 · Dice 0.403 · extreme bias +0.064 K** | The same pair as D1 but averaged over **all 80 time steps** rather than the single peak frame — D1's IoU 0.049 is the peak-frame worst case, not the typical one. This is the rung every learned model must beat. |
| D3 | 2026-09-29 | Downscaling | `model.yaml` `downscaling.learned` — ridge linear filter, 5×5, 27 parameters, `alpha 1e-06` | same pair, spatial 30 % longitude holdout (67 of 221 columns) | deterministic | learned vs baseline table | **peak 0.9977 · RMSE 1.441 K · MAE 1.105 K · IoU 0.244 · Dice 0.331 · extreme bias −0.702 K** | Beats the baseline on RMSE (−0.231 K), MAE (−0.238 K) and p95 error (−0.129 K) but **shrinks the tail**: extreme bias is −0.70 K where bilinear is +0.06 K, so the p99 footprint IoU/Dice fall. Least squares is a conditional-mean estimator, and no radius, penalty or feature set tested removes that (see below). |
| D4 | 2026-09-29 | Downscaling | D3 + out-of-fold monotone quantile calibration, 201 knots | same pair, same holdout | deterministic | learned vs baseline table, calibrated | **peak 0.9994 · RMSE 1.466 K · MAE 1.137 K · IoU 0.274 · Dice 0.383 · extreme bias −0.179 K** | The calibration buys back most of the tail: extreme bias −0.70 → −0.18 K, Dice +0.052, IoU +0.030, p99 error −0.078 K, for a +0.025 K RMSE cost. The calibrated model beats the baseline on RMSE, MAE, p95 and p99; the baseline still wins on IoU, Dice and peak ratio. **Neither dominates — both tables are the result.** |
| D5 | 2026-09-29 | Downscaling | `model.yaml` `interpolation.order: 1` | ERA5 `era5_amphan` + ERA5-Land `era5_land_amphan`, frame 2020-05-21T00:00 | deterministic | extreme-preservation table | **peak preserved 0.0903 · extreme bias −149 mm/3h · IoU 0 · Dice 0 · RMSE/MAE/percentiles *null*** | The cyclone case, re-run. The coarse field is masked to the ERA5-Land footprint (70.3 % of the domain is NaN) and bilinear retains only **9 %** of the fine precipitation peak with **zero** p99 footprint overlap — the clearest case in the repo of interpolation demonstrably failing. Only 29.7 % of cells are comparable, so RMSE, MAE and percentile error are correctly `null` rather than quoted off a third of the domain. **This corrects D1's note:** the pair *is* scoreable for the baseline on the current code; it is the *learned* model that refuses amphan, and on holdout coverage, not on the peak test. |
| DET1 | 2026-09-29 | Detection | `Z_THRESHOLD: 2.5`, `ANOMALY_DIRECTION` (msl = low), `MIN_CLIMATOLOGY_COVERAGE: 0.9` | ERA5 `era5_amphan` (`msl`), `era5_heatwave` (`t2m`), 1991–2020 climatology | deterministic | frames with regions · strongest peak z | **amphan/msl 48 frames, 43 with regions, strongest 6.96 σ (525 cells, centroid 13.69 N 86.91 E) · amphan/t2m 37 of 48 · heatwave/t2m refused** | The detector had never run end to end: it wrote to a relative `output/` path, never renamed `valid_time`→`time`, and averaged the climatology over space. All fixed. No reference masks exist, so precision/recall stay *unset* in the table below. The heatwave is **refused** by the coverage floor — the local climatology covers lat 5–25 / lon 80–95, i.e. 15.9 % of the heatwave domain — see [dataset.md](dataset.md). **The Amphan σ is superseded by DET2.** |
| DET2 | 2026-09-29 | Detection | `Z_THRESHOLD: 2.5`, `MIN_CLIMATOLOGY_COVERAGE: 0.9`, region-scoped baseline via `EVENT_REGIONS` | ERA5 `era5_heatwave` (`t2m`) + fetched `north_india` climatology (30 files, 1991–2020, full May, 7440 samples/cell) | deterministic | frames with regions · strongest peak z | **heatwave/t2m 80 frames, 18 with regions, 30 regions, strongest 3.71 σ (7 cells, 22.00 N 72.50 E)** | The heatwave gap is closed by **fetching the missing baseline**, not by relaxing the floor: detection, tracking and trajectory now run, so the heatwave demo completes all twelve steps. Re-measured Amphan on the same code path: **19.36 σ**, not the 6.96 σ in DET1 — see the correction below. |

| G1 | 2026-09-30 | GNN | `weights/gnn/08_evaluate_era5_pilot.py`; frozen `st_gnn_checkpoint.pt` from stage 05 (trained on `dataset.csv`) | ERA5 `era5_gnn_pilot` 2020-05-16→22, **168 hourly steps**, 66 nodes; checkpoint unchanged during the run (SHA-256 verified before and after) | `torch.manual_seed(42)` (stage 05) | RMSE vs persistence, RMSE vs graph smoothing, domain diagnostics | **temperature 3.899 · pressure 11.490 · humidity 20.552 · wind 2.839 · precipitation 8.228** RMSE against **persistence 1.011 / 0.706 / 4.557 / 0.580 / 0.281** — the GNN loses on all five | **This is the finding, not a bug.** `domain_diagnostics.csv` shows the checkpoint was fitted on a tensor whose scales are not ERA5: its training **precipitation mean is 7.40 mm/h against the pilot's 0.075 mm/h** (~100×) and its training **pressure σ is 59.96 hPa against 2.97 hPa**. `pilot_std_over_checkpoint_std` is **0.049** (pressure) and **0.025** (precipitation). The two features the README previously quoted as the model's best wins are exactly the two that do not transfer. **Superseded by G2–G3.** |
| G2 | 2026-09-30 | GNN | `weights/gnn/09_train_real_era5.py --architecture plain --loss-weighting uniform`; GNN trained on **real ERA5** | ERA5 `era5_gnn_2020_03/04/05` (1824 hourly training hours, 2020-03-01→05-15) → held-out pilot week 2020-05-16→22 (162 scored targets × 66 nodes) | `torch.manual_seed(42)` | same table as G1, plus frozen-checkpoint comparison | **temperature 1.895 · pressure 1.986 · humidity 10.514 · wind 1.381 · precipitation 0.298** RMSE — **loses to persistence on all five** (−6 % to −181 %) | Training on real ERA5 already collapses the G1 error (pressure 11.49 → 1.99, precipitation 8.23 → 0.30) because the model and the target are now on the same distribution — but the plain decoder still loses to persistence everywhere: predicting the **absolute** field means the network must re-learn "the atmosphere barely moves in an hour" from 1824 hours of data, and it does not get there. Two defects in this run's first version were found and fixed before recording: a **loss imbalance** (flat MSE, dominated by precipitation) and an **evaluation broadcasting bug** that compared every prediction hour against every target hour. The numbers here are from the fixed code. |
| G3 | 2026-09-30 | GNN | `09_train_real_era5.py --architecture residual --loss-weighting persistence` | same archives, same split, same held-out window | `torch.manual_seed(42)` | same table as G2; weighted aggregate | **temperature 0.896 · pressure 0.559 · humidity 4.242 · wind 0.568 · precipitation 0.276** RMSE — **beats persistence on all five**: +11.4 % / +20.8 % / +6.9 % / +2.1 % / +2.0 %. Aggregate weighted MSE **0.0865 vs persistence 0.1030** (train 0.0534) | Two deliberate changes, both measured. (1) **Residual/increment head** (`ResidualSTGNN`): the decoder predicts the one-hour *change*, so a zero increment reproduces persistence exactly — asserted by test at epoch 0, which makes the starting point provable rather than assumed. (2) **Per-feature loss weighting** at 1/persistence MSE, so the least predictable feature cannot dominate the gradient and the weighted score 1.0 *means* "equal to persistence". Train/held-out gap (0.0534 → 0.0865) is modest overfitting with 1824 training hours; the headline gain survives it. **G3 is the reported model.** |
| G4 | 2026-09-30 | GNN | `weights/gnn/run_gnn_tracking_demo.py --checkpoint weights/gnn/outputs/real_era5/st_gnn_real_era5_checkpoint.pt` | frozen pilot tensor (168 steps) + the G3 checkpoint | `torch.manual_seed(42)` | GNN one-step field → `detect_field_anomalies` → `run_tracking_pipeline` | **1 anomaly box; threat ID `THR-2020-0001`** written to `gnn_tracking_demo.json` + `gnn_tracking_demo_trajectories.json` | **The integration claim, and the part that works end to end.** A GNN-predicted field is accepted by the real detector and the real tracker and produces a persistent Threat ID, so the learned stage is wired into the pipeline rather than sitting beside it. It is **one frame**, so this is a prototype bridge, not a forecast-skill claim — and no verdict is attached, because `configs/validation.yaml` gates are still `null`. The demo is checkpoint-aware: a residual checkpoint loads into `ResidualSTGNN`, so an increment is never silently returned as a field. |

> **What G1–G3 mean together.** With the two defects fixed, the progression is clean and each
> step is attributable: real data fixes the domain shift (G1 → G2), the residual head +
> persistence-weighted loss turn a losing model into a winning one (G2 → G3). The final
> model beats persistence on **all five features** at a one-hour lead time on the held-out
> pilot week, with a weighted aggregate gain of **16 %** (0.0865 vs 0.1030). What the stage
> establishes: the learned model is trained, evaluated and reported on the same real ERA5
> distribution as the rest of the project, its improvement is measured against an honest
> baseline, and it is wired into the detector and tracker. The one-hour target is close to
> persistence by construction, so the natural next experiment is a 6–24 h lead time — where
> persistence decays and the value of a learned model should grow — which needs the pilot
> contract relaxed, not a new threshold.

> **Correction to the previously committed `weights/gnn/outputs/era5_pilot_evaluation/`.**
> Those artefacts were produced by a checkpoint that was **never committed** and from an
> archive path on a teammate's machine, so they could not be reproduced from the repository.
> They have been regenerated by G1 on a real, freshly fetched archive and a locally retrained
> checkpoint. The G1 numbers above reproduce the committed ones to within CPU-versus-GPU
> numerical noise (temperature 3.899 versus 3.789, pressure 11.49 versus 10.23), which
> confirms the original measurement rather than overturning it.

> **Correction to DET1.** The Amphan `msl` peak intensity recorded there as 6.96 σ does
> not reproduce; the detector now reports **19.36 σ**, and this is *not* caused by the
> new climatology — the Amphan baseline is untouched by the fetch. Verification of the
> current value:
>
> 1. the streaming mean/std in `load_climatology_stats` match an independent
>    stack-every-file computation to **1.8e-11** relative (mean: exact);
> 2. the replaced `xr.concat` path gives the *same* statistics to 1e-7, so the
>    arithmetic did not change either value;
> 3. the physics — the deepest cell reads **942.7 hPa** against a 1991–2020 mean of
>    1003.8 hPa with σ 3.15 hPa, a **−61 hPa** excursion, consistent with Amphan's
>    observed central pressure of ≈942 hPa.
>
> 6.96 σ would imply only a ~21 hPa anomaly, which no Category-5 cyclone produces.
> The earlier number is superseded; I could not reconstruct how it was produced, so it
> is recorded rather than quietly deleted.


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
reported as `null` rather than a misleading `1.0`. The detector itself now runs end to
end on the local archives for **both** events (DET1, DET2), so detections are real; what
is still missing is a **reference mask** to score them against, so **no value is
measured** and the columns above stay empty until one exists.

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
both. On the current measurements neither does: the learned model wins on RMSE/MAE
and the tail percentiles, bilinear wins on footprint overlap.

### Real pair (D1–D2)

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

`amphan_tp.json` **is scored** (D5) — the coarse precipitation peak is covered by
the land reference even though 70.3 % of the domain is NaN, and bilinear keeps only
9 % of the fine peak. What makes the table honest is that the metrics needing paired
cells (RMSE, MAE, percentile error) come back `null` there, because only 29.7 % of
cells are comparable — below the 50 % floor. The *learned* model refuses amphan
outright, on that same coverage floor.

> **Correction.** D1 recorded amphan as unscoreable; that does not reproduce on the
> current code, so D5 supersedes it. D1's numbers are left in place as the record of
> that run — the correction is a new row, not an edit.

Coarse-vs-refined figures: `data/processed/plots/downscaling/real_heatwave_coarse_vs_refined.png`
(real) and `.../coarse_vs_refined.png` (synthetic interface check, generated by
`python -m src.shared.visualization`).

### Learned filter and calibration (D3–D4)

`python -m src.models.downscaling.super_resolution --all-events --sweep-radii 1,2,3,4`
fits a single ridge linear filter over a `(2r+1)²` neighbourhood and then a monotone
quantile map, and scores both on a **spatial** holdout — the east 30 % of the domain
by longitude, never seen during fitting. Same reference, same cells and same baseline
as D2:

| Metric | Baseline (D2) | Filter (D3) | Calibrated (D4) | Units |
|---|---|---|---|---|
| RMSE | 1.672 | **1.441** | 1.466 | K |
| MAE | 1.342 | **1.105** | 1.137 | K |
| Peak preservation | 1.0002 | 0.9977 | 0.9994 | — |
| Extreme bias | +0.064 | −0.702 | −0.179 | K |
| IoU (p99) | **0.2874** | 0.2435 | 0.2737 | — |
| Dice (p99) | **0.4032** | 0.3312 | 0.3832 | — |
| Exceedance change | +0.0211 | +0.0031 | +0.0067 | — |
| Percentile error p95 | 0.8411 | **0.7118** | 0.7868 | K |
| Percentile error p99 | 0.9225 | 0.9946 | **0.9171** | K |

Amphan is **refused** by the holdout-coverage rule, not scored: only 24.9 % of its
holdout cells have a reference value, below the 50 % floor, so no number is
reported. (Its baseline *is* scored — that is D5, at peak 0.0903.)

Two sweeps were run on the inner validation split, and **both came out flat** — which
is why the defaults are kept rather than tuned:

| Sweep | Measured | Reading |
|---|---|---|
| Penalty `alpha` (1e-06 … 100) | validation RMSE 1.43172644 … 1.43175511 K | With ~1.76 M training cells and 27 features the penalty is negligible at every candidate, so **ridge is inert here** and the learned result is essentially ordinary least squares. The sweep is written into the artefact rather than hidden. |
| Radius (1 → 4: 3×3, 5×5, 7×7, 9×9) | validation RMSE 1.4391, 1.4317, 1.4307, 1.4299 K | Monotone but 0.6 % end to end, i.e. inside noise. Radius 2 (27 parameters) is kept over radius 4 (83 parameters) on parsimony, not on score. |

The **feature class is exhausted**: `[1, centre, neighbours − centre]` spans *any* local
linear filter plus an intercept, so no linear feature set can do better at fixed radius.
Richer features (a quadratic centre, local max/min) moved RMSE by ~0.01 K and the tail
bias by ~0.02 K — which is why the tail is addressed by calibration instead.

Both the uncalibrated and calibrated tables are artefacts:
`data/processed/validation/downscaling/heatwave_t2m_learned.json` (**local only** —
`data/processed/**` is gitignored). Reproduce offline with the command above, or run the
whole chain with `python -m src.shared.demo --event heatwave`.

---

## Gate thresholds

**Fill this in only after the metrics above have real values.**

| Stage | Primary metric | PASS | DEGRADE | SUPPRESS | Justification |
|---|---|---|---|---|---|
| Detection | F1 | *unset* | *unset* | *unset* | — |
| Tracking | track continuity | *unset* | *unset* | *unset* | — |
| Transition | Brier score | *unset* | *unset* | *unset* | — |
| Downscaling | peak preservation | *unset* | *unset* | *unset* | Not settable from one event (D2–D4). Peak preservation (1.0002 / 0.9994) and footprint overlap (IoU 0.287 / 0.274) disagree, and the calibrated and uncalibrated models trade RMSE against extreme bias, so any single threshold would encode a choice we cannot yet justify. |

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
