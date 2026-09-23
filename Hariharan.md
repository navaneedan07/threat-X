# HARIHARAN --- SIH 26078

## Primary Work

**Atmospheric Precursor Analysis**

## Secondary Work

**Backend & REST API**

## Main Tasks

-   Analyze pressure, temperature, moisture/humidity, wind and relevant
    derived signals.
-   Engineer a compact precursor feature set.
-   Document feature definitions, units, source variables and temporal
    windows.
-   Produce explanatory time-series/spatial plots.
-   Feed precursor features into Threat Transition Intelligence.
-   Build FastAPI endpoints for threats, trajectories, precursors,
    forecasts and alerts.
-   Define Pydantic schemas and a stable JSON contract.
-   Connect API outputs to Sachin's dashboard.
-   Implement degraded-mode/error handling.

## Suggested API

-   `GET /health`
-   `GET /threats`
-   `GET /threats/{id}`
-   `GET /threats/{id}/trajectory`
-   `GET /threats/{id}/precursors`
-   `GET /threats/{id}/forecast`
-   `GET /alerts/{id}`

## Required Deliverables

-   Precursor feature module
-   Feature table
-   Pressure/moisture/temperature/wind plots
-   FastAPI service
-   Pydantic schemas
-   JSON alert response
-   Integration tests

## PPT Contribution

-   **Slide 2:** Explainable threat evolution.
-   **Slide 3:** Precursor + API blocks.
-   **Slide 4:** Deployment architecture and feasibility.
-   **Slide 5:** Operational usefulness.
-   **Slide 6:** Atmospheric/API references.

## Demo

Show: 1. Threat ID 2. Precursor plot 3. Threat evolution 4. API request
5. JSON alert 6. Dashboard consuming the response

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Precursors

-   [ ] Variables selected
-   [ ] Feature definitions documented
-   [ ] Feature extraction implemented
-   [ ] Normalization implemented
-   [ ] Feature table generated
-   [ ] Pressure plot generated
-   [ ] Moisture plot generated
-   [ ] Temperature plot generated
-   [ ] Wind/convergence plot generated
-   [ ] Threat-vs-precursor plot generated

### API

-   [ ] FastAPI project created
-   [ ] Health endpoint works
-   [ ] Threat endpoint works
-   [ ] Trajectory endpoint works
-   [ ] Precursor endpoint works
-   [ ] Alert endpoint works
-   [ ] Pydantic schemas created
-   [ ] JSON response validated
-   [ ] API documentation checked

### Integration

-   [ ] API receives real model output
-   [ ] Dashboard can consume API
-   [ ] Provenance included
-   [ ] Error handling implemented
-   [ ] Degraded mode tested

### PPT

-   [ ] Slide 2 content supplied
-   [ ] Slide 3 API/data-flow block supplied
-   [ ] Slide 4 deployment evidence supplied
-   [ ] Slide 5 operational evidence supplied
-   [ ] Slide 6 references supplied

### Demo

-   [ ] Precursor plot works
-   [ ] API request works
-   [ ] JSON alert works
-   [ ] Dashboard receives response

## Definition of Done

Precursor features must explain an actual event and the API must expose
the same threat state used by the dashboard.
