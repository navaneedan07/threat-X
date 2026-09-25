# Experiments

The experiment log, and the only place validation gate thresholds may be
justified.

> **STATUS: no experiment on real data has been run.**
> Every table below is still a template. Do not fill a row with an estimate.
> One row in the log below is a **synthetic** interface check, labelled as such;
> it is not evidence about any real event.

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

---

## Transition intelligence

| Metric | Definition | Target | Measured |
|---|---|---|---|
| Brier score | mean squared error of the probability | *unset* | — |
| ROC-AUC | discrimination, where class balance allows | *unset* | — |
| Calibration | predicted vs observed frequency | *unset* | — |
| Lead-time error (h) | error in the predicted transition window | *unset* | — |

Extreme events are rare, so a good score can come from predicting "no transition"
everywhere. Always report the base rate alongside Brier and AUC.

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

---

## Gate thresholds

**Fill this in only after the metrics above have real values.**

| Stage | Primary metric | PASS | DEGRADE | SUPPRESS | Justification |
|---|---|---|---|---|---|
| Detection | F1 | *unset* | *unset* | *unset* | — |
| Tracking | track continuity | *unset* | *unset* | *unset* | — |
| Transition | Brier score | *unset* | *unset* | *unset* | — |
| Downscaling | peak preservation | *unset* | *unset* | *unset* | — |

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
