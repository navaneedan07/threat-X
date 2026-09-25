# Threat Transition Intelligence — target definition

**Status:** target defined and implemented (`src/models/transition/transition_model.py`).
Trained on a **severity proxy**, not on real tracked-threat escalation. A fitted
horizon is served only if it clears the publishing gate (§7); today most do not.
**Last updated:** 2026-09-25

> `team/Navaneedan.md`: *"Target definition must be explicit and written down
> before training."* This is that document. The model does not exist until this
> does.

---

## 1. What the model predicts

For a time step `t` and a horizon `H`:

```
P( severity band reaches TARGET within (t, t + H] | atmospheric precursor state at t )
```

* Rows are restricted to time steps where the **current band is already below
  `TARGET`**. A step that is already at the target has nothing to transition to,
  and including those rows would inflate the apparent base rate.
* `TARGET` is the next band up from the current one, so the question is always
  "does this escalate one step", never "does it eventually get bad".
* Features come from time `t` **only**. The label comes strictly after `t`. No
  row is built from any future value, and any row whose `t + H` runs past the end
  of the series is **dropped** rather than labelled optimistically.

## 2. The severity proxy — and why it is not the real thing

**The repo contains no threat-severity time series.** Nothing records "the threat
intensified at time t":

| Would-be source | State |
|---|---|
| `src/tracking/` (Sachin) | empty package — no Threat Object is produced by anything |
| `src/detection/anomaly_detection.py` | a standalone script; writes to `output/`, which does not exist, and nothing runs it |
| `src/transition/lifecycle.py` | assigns a *phase* from intensity, but needs an intensity series to exist first |

So severity is derived from the precursor series' own **event-defining field**,
measured at the threat centroid:

| Event | Severity field | Units |
|---|---|---|
| `cyclone`, `extreme_rainfall` | `precipitation_rate_mm3h` | mm/3h |
| `heatwave` | `t2m_anomaly_k` | K |

Band boundaries are **quantiles of that event's own distribution**, never absolute
cut-offs. `configs/tracking.yaml` → `severity_bands` is `null` for exactly this
reason: a fixed number would be a guess, and a guessed band boundary makes every
downstream probability meaningless. Deriving bands per event is defensible;
inventing a global threshold is not.

> **This is a proxy.** It measures *"did the surface signal at the centroid
> intensify"*, not *"did the tracked threat object escalate"*. Any probability it
> produces describes the proxy. Replace the proxy with real detector/tracker
> output the moment `src/tracking/` produces one, and retrain — do not relabel
> these results as threat transitions.

## 3. Horizons

`6 / 12 / 18 / 24` hours, "where the data supports it" — which is a real
constraint, not a formality:

* The precursor series is **3-hourly**, so `H = 6` is only 2 steps ahead.
* Rows whose `t + H` exceeds the series are dropped, so **each horizon has a
  different sample count and they are not comparable to each other**.
* Every horizon reports its own row count. A horizon with too few rows to fit is
  reported as untrained (probability stays `null`) rather than fitted anyway.

## 4. Features

Only precursor fields that are actually computed from the available surface data:

```
pressure_tendency_3h_hpa     wind_divergence_s1
wind_speed_ms                precipitation_rate_mm3h
t2m_anomaly_k
```

The three pressure-level fields (`moisture_flux_convergence_g_kg_s`,
`vorticity_850_s1`, `theta_e_gradient_k_100km`) are `null` throughout because the
dataset has no pressure-level variables. They are **never** used as features and
**never** zero-filled — silently imputing them would invent the physics the
project claims to explain.

## 5. Reporting rules

* **The base rate is always reported.** Escalation is rare. A model predicting
  "no transition" everywhere for a 3% base rate earns a Brier score of 0.03,
  which reads as excellent and means nothing.
* **The Brier skill score against the constant base-rate forecast is the number
  that decides usefulness.** `skill_score <= 0` means the model is no better than
  always predicting the base rate, and must be described that way.
* Calibration is reported alongside, never instead.
* **Known methodological limitation to state up front:** the features and the
  proxy label are derived from the *same* series, so the model can partly learn
  autocorrelation rather than a physical precursor relationship. A genuinely
  predictive precursor model needs features that lead the label by more than the
  series' own persistence — which requires either longer series or real forecast
  data. This caveat belongs on the slide, not buried.

## 6. Data availability (honest assessment)

| Quantity | Value |
|---|---|
| Events with data | 2 (`era5_amphan`, `era5_heatwave`) |
| Time step | 3 hours |
| Amphan span | 2020-05-16 → 2020-05-21 (6 days ≈ 48 steps) |
| Heatwave span | 2022-05-01 → 2022-05-10 (10 days ≈ 80 steps) |
| Default pipeline window | **48 h ≈ 17 steps per event** |
| Maximum rows if the full span is used | 128 (measured: 48 + 80) |

The run below used the **full span** (`window_hours=400`) rather than the 48 h
default, so these are the largest numbers the current data can produce.

### Measured outcome (2026-09-25)

Both events were run end to end on real ERA5 data, **one dataset per event**, over
the full span (`window_hours=400`). Per-horizon detail lives in
[`docs/experiments.md`](experiments.md) → *Transition intelligence*:

| Event | Horizons trained | Brier skill range | Publishing gate (§7) |
|---|---|---|---|
| `THR-2020-0001` Amphan | 12, 18, 24 h (6 h refused) | −0.47 … +0.36 | **only 24 h** — 12 h and 18 h withheld, both at skill ≤ 0 |
| `THR-2022-0001` heatwave | 6, 12, 18, 24 h | +0.01 … +0.09 | all four clear the floor |

The 6 h Amphan horizon was **refused automatically** (3 positives against a
minimum of 5). That is the intended behaviour: the probability is withheld rather
than fitted on three examples.

Two things are true at the same time, and both belong on the slide:

* **The gate is a floor, not a validation.** Clearing it means "beat the constant
  base-rate forecast and beat chance on these rows". It says nothing about whether
  the target is the right target — and here the target is a proxy (§2).
* **The sample still cannot support the claim.** Amphan 24 h clears the gate on
  **27 rows**, and the heatwave's 24 h base rate is **86.8 %**, so its label is
  near-degenerate: "always yes" scores well while discriminating almost nothing.

That is **not enough to train a credible model**, and no amount of model choice
fixes it. What the current implementation demonstrates is that the *machinery*
works end to end — dataset construction, horizon handling, fitting, calibrated
probability output, honest metric reporting, and refusing to serve what it cannot
back up — on a sample too small to support a scientific claim.

**To make this real, the team needs at least one of:**

1. More events (each new ERA5 case adds ~50–80 rows per event type).
2. Real forecast data rather than reanalysis, so lead time means something.
3. A threat-severity series from `src/detection/` + `src/tracking/`, replacing
   the proxy in §2.

Until then every transition probability stays labelled **prototype, proxy
target**, both in the API (`provenance.model_id`) and on the slides.

## 7. The publishing gate — fitted is not the same as publishable

`TransitionReport.can_publish()` decides whether any probability from this module
may reach the API, and `backend/main.py` → `transition_intelligence` follows it.
A horizon is published only if **both** conditions hold on the out-of-fold metrics:

| Condition | Reference value | Why this is not an invented threshold |
|---|---|---|
| `skill_score > 0` | `0.0` | the score the constant base-rate forecast itself earns |
| `roc_auc > 0.5` | `0.5` | the score earned by zero information |

Those two constants are the **only** numbers in the gate, and `tests/test_transition_model.py`
asserts their values so a tuned boundary cannot be slipped in later. A stricter
bound would be a guessed threshold, which is what `severity_bands: null` exists to
prevent.

A withheld horizon is **still reported**, with the metric that failed it. The API
keeps its probability `null` and the report keeps the number, so withholding is a
result rather than a gap. The run above contains both failure modes, which is why
the gate needs both conditions:

* **Amphan 18 h — ranks well, forecasts badly.** AUC 0.7987 but skill −0.0253. It
  orders cases better than chance and is still worse than always predicting the
  base rate. AUC alone would have shipped this.
* **Amphan 12 h — the trap the base-rate rule exists to catch.** Brier 0.1986 sits
  comfortably next to 18 h's 0.1878; `skill_score = −0.4683` shows it is worse
  than predicting "no escalation" for every single row.

`TransitionEstimate.calibrated` stays `false` regardless of the gate: these models
are not calibrated against observed threat escalation, and that flag is not the
gate. `docs/api.md` records what the endpoint does while the gate withholds.
