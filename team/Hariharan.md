# HARIHARAN — THREAT-X

> **Primary:** Atmospheric Precursor Analysis
> **Secondary:** Backend & REST API
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 6 PRECURSORS, 10 API + GIS (with Sachin) |
| **Milestones** | M5, M6 (with Navaneedan), M9 (with Sachin) |
| **Main artifact** | Precursor feature module · feature plots · FastAPI endpoints · Pydantic/JSON schema · integration tests |
| **PPT** | Slides 2–6: explainability / API / deployment (Slide 4 **lead** with Navaneedan) |
| **Demo segments** | 95–125s explain evolution · 175–195s validate + alert |
| **Stack** | Python, Xarray, NumPy, Pandas, MetPy, scikit-learn, Matplotlib, FastAPI, Pydantic, Uvicorn |

You own the **"WHY is it evolving?"** answer. A system that only says "it got stronger" is a
dashboard; a system that says "it got stronger *because* moisture converged and pressure
fell" is intelligence. Then you make that readable to a machine through the API.

---

## 1. Precursor analysis — what to actually build (`src/precursors/`)

### Step 1 — Study the signals

Look at what changes **before** an anomaly intensifies, not what happens after:

| Group | Variables |
|---|---|
| Moisture | relative humidity, specific humidity, moisture convergence, precipitable water |
| Pressure | sea-level pressure, **pressure tendency** (the change, not the level), geopotential height change |
| Temperature | anomaly, gradient, vertical structure |
| Wind | speed, direction, convergence, vertical shear, **vorticity** |

### Step 2 — Compact feature set (`src/precursors/feature_engineering.py`)

- Engineer a **compact** feature set — compact matters. A 200-column table that nobody can
  explain defeats the purpose.
- Document every feature: definition, units, source variable, level, and **temporal window**.
- Normalize where needed and record the normalization.
- **Exit condition (M5):** feature table + explanatory plots saved.

### Step 3 — Explanatory plots (`src/precursors/analysis.py`)

Time-series and spatial plots that let a human see the precursor move before the threat
does. These are your slide 3 and demo assets.

### Step 4 — Hand off to transition

> **Critical:** feed precursor features **into the transition stage**, not into a
> disconnected forecasting model. Navaneedan's transition model consumes your feature table
> (M6). Agree the table schema with them **before** you build it.

---

## 2. Backend / REST API — what to actually build (`backend/`)

Currently `backend/main.py` serves only `GET /health`. The threat endpoints in
`docs/api.md` are marked **Planned** — add each one only when it can serve **real** pipeline
output.

### Endpoints

| Method | Path | Serves |
|---|---|---|
| GET | `/health` | **Implemented** — service status + which stages are wired |
| GET | `/api/v1/threats` | all active threat objects |
| GET | `/api/v1/threats/{id}` | one threat object |
| GET | `/api/v1/threats/{id}/trajectory` | T0..Tn positions |
| GET | `/api/v1/threats/{id}/precursors` | your feature series |
| GET | `/api/v1/threats/{id}/transition` | probability + window |
| GET | `/api/v1/threats/{id}/footprint` | GeoJSON footprint |
| GET | `/api/v1/alerts` | alert-shaped summary |

### The JSON schema (`backend/schemas/`)

Must contain, at minimum:

```text
event ID · timestamp · centroid · footprint/radius · severity
confidence · supporting variables · provenance
```

**Provenance** is the one people forget — record which model/version/config produced the
number. Without it the output cannot be defended.

### Non-negotiable conventions

- **`null` means "not computed yet."** Never substitute a placeholder number.
- **No invented probabilities.** A probability ships only from a trained and validated model.
- Distances km, speeds km/h, angles degrees clockwise from north, times ISO 8601 UTC.
- `centroid` is `[latitude, longitude]`.

### Degraded mode

The demo must continue if an expensive ML component is unavailable. If detection or tracking
has not run, the service still starts and reports `pipeline_stages` honestly in `/health`.
A missing artifact returns `503` naming the stage that did not run — **never** fabricated
threat data.

### Tests

Integration tests in `tests/` that exercise a documented JSON response end-to-end from model
output to endpoint. **Exit condition (M9):** JSON alert + map + trajectory + localized footprint.

---

## Evidence you must save

- [ ] Precursor feature module (committed)
- [ ] Feature table (saved artifact)
- [ ] Pressure / moisture / temperature / wind plots
- [ ] Threat-vs-precursor plot
- [ ] FastAPI service running
- [ ] Pydantic schemas
- [ ] A real JSON alert response captured
- [ ] Integration tests passing
- [ ] Degraded-mode behaviour demonstrated

---

## PPT contribution

| Slide | Your input |
|---|---|
| 2 IDEA | Explainable threat evolution — the threat *and* the reason |
| 3 TECHNICAL | Precursor block + API block |
| 4 FEASIBILITY | **Lead with Navaneedan.** Deployment architecture, degraded mode, mitigation and fallbacks |
| 5 IMPACT | Operational usefulness — what a forecaster does with this |
| 6 REFERENCES | Atmospheric / API references — add rows to `docs/references.md` first |

**Required evidence (from the team checklist):** precursor plot · API schema/JSON ·
deployment/data-flow block · degraded mode.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 95–125s | Explain evolution | Precursor feature plot + change over time |
| 175–195s | Validate + alert (with Navaneedan, Sachin) | PASS/DEGRADE + JSON alert + map |

**The moment that matters:** open the API docs, hit a real endpoint, and show the JSON alert
returning actual pipeline output. Not a screenshot of one.

---

## Fallback

| If this fails | Fall back to |
|---|---|
| FastAPI not ready | Pre-computed JSON in `data/samples/` served by the same service |
| Precursor features incomplete | Plot the raw variables (pressure/moisture/wind) directly — still explains evolution |
| Live inference unavailable at the venue | Degraded mode + prepared response files |
| Transition model unavailable | Serve `null` for probability and window; the dashboard renders "—" |

---

## Definition of done

- **PRECURSORS:** feature table and at least one explanatory plot are saved
- **API:** one documented JSON response works from model output to endpoint
- The API must expose the **same threat state** the dashboard renders — no divergence
- Precursor features must explain an **actual** event, not a synthetic one
- Plus the team-wide rule: *runs from a reproducible input, produces a saved output, has at
  least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Precursors
- [ ] Variables selected
- [ ] Feature definitions documented (units + temporal window)
- [ ] Feature extraction implemented
- [ ] Normalization implemented
- [ ] Feature table generated and saved
- [ ] Pressure plot generated
- [ ] Moisture plot generated
- [ ] Temperature plot generated
- [ ] Wind / vorticity / convergence plot generated
- [ ] Threat-vs-precursor plot generated
- [ ] Feature table schema agreed with Navaneedan (M6 input)

### API
- [ ] `backend/` service runs via `uvicorn backend.main:app --reload`
- [ ] `/health` reports real stage status
- [ ] Threat endpoint works
- [ ] Trajectory endpoint works
- [ ] Precursor endpoint works
- [ ] Transition endpoint works
- [ ] Footprint endpoint works
- [ ] Alert endpoint works
- [ ] Pydantic schemas created
- [ ] JSON response validated against the schema
- [ ] Provenance included in responses
- [ ] Error handling implemented (404 / 503 / 422)
- [ ] Degraded mode implemented and tested
- [ ] Integration tests written and passing

### PPT
- [ ] Slide 2 explainability content supplied
- [ ] Slide 3 precursor + API blocks supplied
- [ ] Slide 4 deployment architecture + fallback supplied
- [ ] Slide 5 operational evidence supplied
- [ ] Slide 6 references supplied
- [ ] No invented numbers

### Demo
- [ ] Precursor plot works offline
- [ ] Live API request works
- [ ] JSON alert returns real data
- [ ] Dashboard receives the response
