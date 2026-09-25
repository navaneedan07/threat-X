# Architecture

How Threat-X is divided up, what each part owns, and what actually exists.

> **Keep the status columns honest.** This file is the single place the team
> checks "is that built yet?". See [Status vocabulary](#status-vocabulary).

---

## Status vocabulary

Every component in this document is labelled with exactly one of:

| Label | Meaning |
|---|---|
| **Implemented** | Exists in the codebase and has been tested |
| **Experimental** | Under development or being evaluated |
| **Planned** | Described by the architecture, not yet written |
| **Target** | A requirement from the SIH problem statement we intend to address |

A component that is **Target** but not **Implemented** must never be described as
working — in the README, the PPT, the demo, or a commit message.

---

## Layer map

The repo separates the deployable layers at the top level.

| Layer | Directory | Language | Owner |
|---|---|---|---|
| Frontend / UI | `frontend/` | TypeScript (React) | Sachin |
| Backend / service | `backend/` | Python (FastAPI) | Hariharan |
| Pipeline logic | `src/` | Python | per stage |
| Learned models (ML + DL) | `src/models/` | Python | Varnika, Navaneedan |
| Trained weights | `weights/` | — | per model |

Supporting trees: `configs/` (parameters), `data/` (data, mostly gitignored),
`notebooks/`, `tests/`, `docs/`.

### Why `src/` and `src/models/` are separate

`src/` holds **deterministic** processing — interpolation, thresholding,
clustering, geospatial math. `src/models/` holds only things that are **fitted or
trained**.

This is not cosmetic. The README requires the interpolation baseline to remain
available and to be compared against any learned downscaler. For that comparison
to be fair the baseline must not live inside the package it is being benchmarked
against — so `src/downscaling/baseline.py` (deterministic) and
`src/models/downscaling/` (learned) are deliberately in different trees.

Code is source and lives in `src/`. Trained artefacts are data and live in
`weights/`.

**Rule:** a new file goes in `src/models/` if it is trained, and in the relevant
`src/<stage>/` package if it is not. If it is a threshold, a formula, or a
geometry calculation, it is not a model.

---

## Pipeline stages

| Stage | Directory | Owner | Status |
|---|---|---|---|
| Data ingestion & preprocessing | `src/data/` | Aravinth, Pushpa | Planned |
| Climatological baseline | `src/data/` | Pushpa | Planned |
| Extreme anomaly detection | `src/detection/` | Pushpa | Planned |
| Threat Object + tracking | `src/tracking/` | Sachin | Planned |
| Atmospheric precursors | `src/precursors/` | Hariharan | **Implemented** |
| Threat lifecycle (deterministic) | `src/transition/` | Navaneedan | **Implemented** |
| Downscaling baseline (interpolation) | `src/downscaling/` | Navaneedan, Aravinth | **Implemented** (baseline) |
| Validation & gates | `src/validation/` | Navaneedan | **Implemented** (thresholds unresolved) |
| Shared contracts & helpers | `src/shared/` | Navaneedan | **Implemented** |
| GNN / mesh | `src/models/gnn/` | Varnika | Planned |
| Learned downscaling (CNN / diffusion) | `src/models/downscaling/` | Navaneedan, Aravinth | Planned |
| Learned transition model | `src/models/transition/` | Navaneedan | **Implemented** (prototype target; publishing gate withholds 2 of 7 fitted horizons) |
| REST API | `backend/` | Hariharan | **Implemented** (all 8 endpoints) |
| Dashboard | `frontend/` | Sachin | Planned |

---

## Data flow

```text
NWP / Ensemble Forecast
        |
        v
src/data/          ingest, standardize coords/time, align grids
        |
        v
src/data/          climatological baseline -> anomaly / percentile / EFI
        |
        v
src/detection/     threshold -> spatial filter -> connected components
        |
        v
src/tracking/      persistent Threat Object (stable threat_id)
        |
        +----------------------+
        v                      v
src/models/gnn/        src/precursors/
GNN trajectory         P / T / RH / wind
        |                      |
        +----------+-----------+
                   v
        src/transition/        lifecycle + transition probability + window
                   |
                   v
        src/downscaling/       12 km -> ~5 km localization
        (or src/models/downscaling/ when a learned model has earned its place)
                   |
                   v
        src/validation/        metrics + PASS / DEGRADE / SUPPRESS gate
                   |
                   v
             backend/          machine-readable threat JSON
                   |
                   v
            frontend/          map + trajectory + threat card + alert
```

---

## Interfaces between layers

These are the contracts. Changing one is a breaking change and needs all
affected owners to agree.

| Interface | Producer | Consumer | Status |
|---|---|---|---|
| Anomaly mask + candidate regions | `src/detection/` | `src/tracking/` | Planned |
| **Threat Object** | schema in `src/shared/contracts.py` (`Implemented`); producer `src/tracking/` | backend, frontend, GNN, TTIE | **Implemented** (schema), producer Planned |
| Trajectory series | `src/tracking/` | backend, frontend | Planned |
| Precursor feature table | `src/precursors/` | TTIE | **Implemented** (producer) |
| Transition output | `src/transition/`, `src/models/transition/` | backend, frontend | **Implemented** (lifecycle + model); probabilities withheld by the publishing gate (`transition.md` §7) and the service does not read the report yet |
| Refined local field | `src/downscaling/` | validation, frontend | **Implemented** (interpolation baseline) |
| Metric table + gate verdict | `src/validation/` | README, PPT, demo | **Implemented** (thresholds unresolved) |
| Threat JSON | `backend/` | `frontend/` | Planned |

The **Threat Object** is the most load-bearing of these — four components
serialize it, so its schema must be frozen before parallel work can safely
proceed. The implementation is `src/shared/contracts.py` (23 types, stdlib-only
so every stage can import it). `tests/test_contracts.py` asserts its field names
match `backend/schemas/threat.py`, so the two cannot drift apart silently.

`configs/tracking.yaml` → `threat_object_fields` declares the same 15 top-level
fields, and `contract_drift()` checks the agreement in **both** directions, so a
field added to one side and not the other fails `tests/test_contracts.py` rather
than surfacing months later as an unexplained mismatch:

```bash
python -c "from src.shared.contracts import contract_drift; print(contract_drift())"
# [] means the config and the code agree
```

---

## Configuration

Parameters are never hard-coded in modules. Each stage reads its own config:

| File | Covers |
|---|---|
| `configs/data.yaml` | sources, variables, grid, time, ensemble |
| `configs/model.yaml` | detector, graph/GNN, transition, downscaling |
| `configs/tracking.yaml` | association thresholds, lifecycle, severity bands |
| `configs/validation.yaml` | metric sets, PASS/DEGRADE/SUPPRESS gates |

Every unresolved value is `null` on purpose. If a value is unknown it stays
`null` and the stage reports "not available" rather than substituting a guess.

---

## Open questions

Track decisions that block more than one person here.

| # | Question | Blocks | Owner | Status |
|---|---|---|---|---|
| 1 | Which dataset, and exactly which version? | everything | Aravinth | open |
| 2 | What are the coarse and fine grid resolutions? | downscaling | Aravinth | open |
| 3 | Which extreme-event type is the primary demo case? | detection, replay | Pushpa | open |
| 4 | Threat Object schema frozen? | tracking, backend, GNN, TTIE | Navaneedan | **closed** — schema Implemented; `configs/tracking.yaml` aligned |
| 5 | Gate thresholds — where are they justified? | validation | Navaneedan | open |
| 6 | Is a learned downscaler actually beating the interpolation baseline? | downscaling | Navaneedan | open — baseline is now **measurable**: on synthetic data it preserves the peak at 2x coarsening (1.00) but loses it at 10x (0.62), so there is a number to beat |
| 7 | Does the API serve the transition probabilities at all, given the target is a proxy? | backend, slides, demo | Navaneedan | open — the model runs and the gate is implemented; serving them means publishing a proxy. Decide before the demo, and either way the caveat goes on the slide |

---

## Related docs

- `dataset.md` — dataset decision record
- `api.md` — REST contract
- `experiments.md` — experiment log + gate threshold justification
- `references.md` — citations
