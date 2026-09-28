# API

The REST contract between the pipeline and the dashboard.

> **STATUS: all eight endpoints are implemented and routed** (`backend/routers/`,
> 47 tests in `tests/test_api.py`). What they serve today is mostly **fixture-backed**,
> because `src/detection/` and `src/tracking/` do not produce threat objects yet.
> `/health` says which stages are live; every response says which stage it came from.
> An endpoint may only be wired to real pipeline output once that output exists —
> a stub returning invented numbers is worse than a 404.

---

## Running

```bash
uvicorn backend.main:app --reload
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/health` | liveness |
| `http://localhost:8000/docs` | interactive OpenAPI docs |

---

## Endpoints

| Method | Path | Status | Returns |
|---|---|---|---|
| GET | `/health` | **Implemented** | service status + which stages are wired |
| GET | `/api/v1/threats` | **Implemented** (fixtures) | all active threat objects |
| GET | `/api/v1/threats/{threat_id}` | **Implemented** (fixtures) | one threat object |
| GET | `/api/v1/threats/{threat_id}/trajectory` | **Implemented** (fixtures) | T0..Tn positions |
| GET | `/api/v1/threats/{threat_id}/precursors` | **Implemented** (fixtures) | precursor feature series |
| GET | `/api/v1/threats/{threat_id}/transition` | **Implemented** (report-backed when a report exists) | transition probability + window |
| GET | `/api/v1/threats/{threat_id}/footprint` | **Implemented** (fixtures) | footprint geometry (GeoJSON) |
| GET | `/api/v1/alerts` | **Implemented** (fixtures) | alert-shaped summary |

"Fixtures" means the response comes from `data/samples/`, not from the pipeline.
"Null-shaped" means the endpoint works and its model fields are `null` — see the
next section. `GET /health` reports a `pipeline_stages` map of booleans, and a
stage flips to `true` only when it can serve real output — the same rule as the
endpoints.

---

## Conventions

### `null` means "not computed yet"

Never substitute a placeholder number. If a model output does not exist, the
field is `null` and the consumer renders "—".

```json
{
  "transition": {
    "target_severity": "severe",
    "probability": null,
    "expected_window": null
  }
}
```

### No invented probabilities

A probability is only published when it comes from a trained model that beats the
no-skill references. `src/models/transition/transition_model.py` produces
probabilities for real events and then **withholds most of them**:
`TransitionReport.can_publish()` publishes a horizon only if it beats both the
base-rate forecast and chance (`docs/transition.md` §7).

`transition_service` reads the stored report.
`train_all_horizons` writes one report per event to
`<DATA_ROOT>/processed/validation/<threat_id>/transition_report.json`, and each
horizon records a `serving` block with the probability for the threat's most
recent fully-observed state — **but only when the gate allows it**. The service
serves the shortest publishable horizon and leaves every withheld horizon `null`.

For a threat with no stored report (every fixture today), `/transition` returns:

```json
{
  "transition_evaluation": {
    "target_state": "extreme",
    "probability": null,
    "calibrated": false,
    "brier_score_baseline": null,
    "expected_window_hours": null
  },
  "provenance": { "model_id": null, "training_run": null }
}
```

That is the contract working, not a stub: the gate decides per horizon what may be
served, and a withheld horizon stays null with its reason recorded in the report.
Served probabilities are prototype, severity-**proxy** targets and stay labelled as
such in `provenance.training_run`. See `experiments.md` for the numbers,
`tests/test_transition_serving.py` for the gate behaviour, and
`tests/test_api.py::test_probability_null_when_uncalibrated` for the null contract.

### Units are explicit

Distances in km, speeds in km/h, angles in degrees clockwise from north, times in
ISO 8601 UTC. Do not ship a bare number whose unit lives only in a docstring.

---

## Response: threat object

The example shape from the README. Values marked *illustrative* are format
examples, **not** model output.

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
    "probability": null,
    "window_hours": null
  },
  "uncertainty": {
    "ensemble_agreement": null
  }
}
```

`centroid` is `[latitude, longitude]` (GeoJSON ordering). Footprint geometry is
returned as GeoJSON separately to keep the list response small.

---

## Schemas

Pydantic models live in `backend/schemas/`. They are the single source of truth for
the JSON contract — the dashboard should be able to be built from the generated
OpenAPI spec alone.

The shared Python-side type is `src/shared/contracts.py`
(`THREAT_OBJECT_FIELDS`), and its field list is kept identical to
`configs/tracking.yaml` → `threat_object_fields`; run `contract_drift()` to check.
`tests/test_contracts.py` asserts every field these Pydantic models promise is
present on the contract.

The API deliberately serves a **subset** as a summary. Three pipeline-side fields
are not on `ThreatObject`:

| Field | Where it surfaces instead |
|---|---|
| `footprint` | `GET /threats/{id}/footprint`, as GeoJSON, to keep list responses small |
| `lifecycle_phase` | `GET /threats/{id}/transition`, which needs the threat history to derive it |
| `ensemble_agreement` | not yet exposed — stays `null` until ensemble data is available |

---

## Error handling

| Situation | Behaviour |
|---|---|
| Unknown `threat_id` | `404` with a machine-readable `detail` |
| Pipeline artefact missing | `503`, `detail` names the stage that did not run |
| Model output not yet available | `200` with `null` fields — **not** `500` |
| Malformed query parameters | `422` (FastAPI default) |

Degraded mode: if detection or tracking has not run, the service still starts and
reports `pipeline_stages` honestly in `/health`. It never fabricates threat data.

---

## Integration checklist

"Serves real output" means from the pipeline, not from `data/samples/`.

- [x] Pydantic schemas written
- [x] `/health` reports real stage status
- [x] Degraded mode tested (`tests/test_api.py`, including the null-probability case)
- [ ] Threat endpoint serves real output — needs `src/tracking/` (Sachin)
- [ ] Trajectory endpoint serves real output — needs `src/tracking/` (Sachin)
- [ ] Precursor endpoint serves real output — needs `precursors.json` (exists locally; `data/processed/**` is gitignored)
- [x] Transition endpoint serves real output — reads the stored report and serves the shortest publishable horizon; withheld horizons stay `null`
- [ ] Alert endpoint serves real output
- [ ] Dashboard consumes the API
- [ ] Response latency measured
