# Atmospheric Precursor Analysis

**Status: Implemented** — `src/precursors/` (Hariharan)
**Last updated: 2026-09-25**

---

## Overview

The precursor analysis module computes physically grounded atmospheric precursor signals from ERA5 surface-level data and exposes them through the REST API at `GET /api/v1/threats/{threat_id}/precursors`.

The purpose is to explain *why* a threat is intensifying by quantifying the atmospheric signals that accompany its evolution.

---

## Dataset Used

**Source:** ERA5 Reanalysis (Copernicus CDS) — surface single-level variables
**Format:** CDS ZIP archive containing two HDF5-format NetCDF4 inner files
**Grid:** 0.25° (~27.8 km), 3-hourly timesteps
**Engine:** `h5netcdf` (h5py backend — required because `netCDF4` does not support file-object reads)

### Available Files

| File | Event | Period | Region |
|------|-------|--------|--------|
| `data/raw/era5_amphan.nc` | Cyclone Amphan | 2020-05-16 to 2020-05-21 | Bay of Bengal (5°–25°N, 80°–95°E) |
| `data/raw/era5_heatwave.nc` | North India Heatwave | 2022-05-01 to 2022-05-10 | N. India (20°–35°N, 68°–90°E) |

### Available Variables

| ERA5 Name | Long name | Units | Step type |
|-----------|-----------|-------|-----------|
| `u10` | 10 m U wind component | m/s | Instantaneous |
| `v10` | 10 m V wind component | m/s | Instantaneous |
| `t2m` | 2 m temperature | K | Instantaneous |
| `msl` | Mean sea level pressure | Pa | Instantaneous |
| `tp` | Total precipitation | m | Accumulated/step |

### Variables NOT Available (no pressure-level data)

The dataset does **not** include pressure-level fields. The following precursors are therefore represented as `null` in all outputs:
- 850 hPa moisture flux convergence
- 850 hPa relative vorticity
- Deep-layer bulk wind shear
- Theta-e gradient

These nulls are accurate and intentional, not computation failures.

---

## Computed Precursor Signals

| Signal | Field name in output | Formula | Units |
|--------|---------------------|---------|-------|
| MSLP Tendency | `pressure_tendency_3h_hpa` | msl[t] − msl[t−1] / 100 | hPa/3h |
| Wind Divergence | `wind_divergence_s1` | ∂u10/∂x + ∂v10/∂y (finite diff.) | s⁻¹ |
| Wind Speed | `wind_speed_ms` | √(u10² + v10²) | m/s |
| Precipitation Rate | `precipitation_rate_mm3h` | tp × 1000 | mm/3h |
| Temperature Anomaly | `t2m_anomaly_k` | t2m − temporal_mean(t2m) | K |

All values are spatially averaged over a 1.5° × 1.5° neighbourhood (~167 km) around the threat centroid.

---

## Module Structure

```
src/precursors/
├── __init__.py       Public API exports
├── engine.py         Core computation (PrecursorResult, compute_precursors, compute_precursor_series, generate_explanatory_summary)
├── pipeline.py       Pipeline runner — processes datasets, writes data/processed/precursors.json, generates plots
└── plots.py          Diagnostic time-series plots (matplotlib, Agg backend)

src/data/
└── loader.py         ERA5 ZIP archive loader (h5netcdf engine)
```

---

## How to Run

### Run the full pipeline (processes both datasets):
```bash
cd threat-X/
python -m src.precursors.pipeline
```

**Outputs:**
- `data/processed/precursors.json` — real computed precursor series (API-ready)
- `data/processed/plots/era5_amphan/precursor_series.png`
- `data/processed/plots/era5_heatwave/precursor_series.png`

### Run from Python:
```python
from src.precursors.pipeline import run_precursor_pipeline

entry = run_precursor_pipeline(
    "era5_amphan",           # or "era5_heatwave"
    window_hours=48,
    output_dir="data/processed",
)
print(entry["explanatory_summary"])
```

### Compute for a custom centroid / dataset:
```python
from src.data.loader import load_dataset
from src.precursors.engine import compute_precursor_series, generate_explanatory_summary

ds = load_dataset("data/raw/era5_amphan.nc")
series = compute_precursor_series(ds, centroid_lat=14.0, centroid_lon=87.0, window_hours=24)
summary = generate_explanatory_summary(series)
```

---

## API Integration

The REST API serves precursor data at:
```
GET /api/v1/threats/{threat_id}/precursors?window_hours=24
```

**Real computed IDs** (from pipeline output):
- `THR-2020-0001` — Cyclone Amphan
- `THR-2022-0001` — North India Heatwave

**Demo fixture ID** (API always available, values from fixture):
- `THR-2026-0001` — Chennai rainfall scenario (fixture, not real computation)

**Service:** `backend/services/precursor_service.py` checks `data/processed/precursors.json` first, then falls back to `data/samples/precursors.json` for fixture IDs.

**Pipeline flag:** `precursor_analysis = True` in `backend/main.py` (flipped after implementation).

---

## Inputs Expected

```
data/raw/era5_amphan.nc        ERA5 CDS ZIP archive (Bay of Bengal, cyclone event)
data/raw/era5_heatwave.nc      ERA5 CDS ZIP archive (North India, heatwave event)
```

The loader handles the unusual format: each `.nc` file is a ZIP containing two inner HDF5-NetCDF4 files (instant + accumulated). This is the standard CDS download format. The `h5netcdf` engine is required (not `netCDF4`).

---

## Outputs Produced

```
data/processed/precursors.json     Real computed series, keyed by threat_id
data/processed/plots/              Diagnostic PNG plots
```

The `precursors.json` format:
```json
{
  "_meta": { "generated_at": "...", "source": "src.precursors.pipeline", "note": "..." },
  "precursors": {
    "THR-2020-0001": {
      "threat_id": "THR-2020-0001",
      "variables_analyzed": ["mslp_tendency", "wind_divergence_10m", ...],
      "unavailable_variables": ["moisture_flux_convergence_g_kg_s (no 850 hPa data)", ...],
      "series": [
        {
          "timestamp": "2020-05-20T21:00:00Z",
          "pressure_tendency_3h_hpa": -1.9,
          "wind_divergence_s1": -3.2e-05,
          "wind_speed_ms": 21.4,
          "precipitation_rate_mm3h": 4.1,
          "t2m_anomaly_k": -0.8,
          "moisture_flux_convergence_g_kg_s": null,
          "vorticity_850_s1": null,
          "theta_e_gradient_k_100km": null
        }
      ],
      "explanatory_summary": "Evolution driven by ..."
    }
  }
}
```

---

## Connection to Transition Intelligence (Navaneedan)

The precursor series is the primary input to the Threat Transition Intelligence Engine (TTIE).

**Interface:** When Navaneedan's TTIE is ready, load precursor series from `data/processed/precursors.json` keyed by `threat_id`. The `series` array contains per-timestep scalar features ready for ML ingestion.

---

## Tests

```bash
pytest tests/test_precursors.py -v   # 32 tests covering loader, engine, series, API
pytest tests/test_api.py -v          # 47 original API tests (all still pass)
```

---

## Limitations

1. **Surface-only data**: No 850 hPa wind/moisture data → vorticity, moisture flux convergence, theta-e gradient, bulk shear cannot be computed. Clearly documented as null.
2. **Fixed centroids**: Cyclone track used is an approximate centroid; real integration awaits Sachin's tracker output.
3. **No climatological baseline**: Temperature anomaly is relative to the dataset temporal mean, not a multi-year climatology. Anomaly interpretation is therefore event-relative, not historically calibrated.
4. **Finite-difference divergence**: Computed from 10m wind on a coarse 0.25° grid. This is an approximation; mass divergence at 850 hPa would be more physically meaningful.
