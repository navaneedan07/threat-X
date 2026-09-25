# API

The REST contract between the pipeline and the dashboard.

> **STATUS: only `GET /health` is implemented.**
> Every threat endpoint below is **Planned**. Do not add an endpoint until it can
> serve real pipeline output — a stub returning invented probabilities is worse
> than a 404.

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
| GET | `/api/v1/threats` | Planned | all active threat objects |
| GET | `/api/v1/threats/{threat_id}` | Planned | one threat object |
| GET | `/api/v1/threats/{threat_id}/trajectory` | Planned | T0..Tn positions |
| GET | `/api/v1/threats/{threat_id}/precursors` | Planned | precursor feature series |
| GET | `/api/v1/threats/{threat_id}/transition` | Planned | transition probability + window |
| GET | `/api/v1/threats/{threat_id}/footprint` | Planned | footprint geometry (GeoJSON) |
| GET | `/api/v1/alerts` | Planned | alert-shaped summary |

`GET /health` reports a `pipeline_stages` map of booleans. A stage flips to `true`
only when it can serve real output — the same rule as the endpoints.

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

A probability is only published when it comes from a trained and validated model.
See `experiments.md` for where that evidence lives.

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

- [ ] Pydantic schemas written
- [ ] `/health` reports real stage status
- [ ] Threat endpoint serves real output
- [ ] Trajectory endpoint serves real output
- [ ] Precursor endpoint serves real output
- [ ] Transition endpoint serves real output
- [ ] Alert endpoint serves real output
- [ ] Dashboard consumes the API
- [ ] Degraded mode tested
- [ ] Response latency measured
