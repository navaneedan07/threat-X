# Dataset

The dataset decision record for Threat-X.

> **STATUS: NO DATASET CONFIRMED.**
> Nothing in this file may be treated as decided until the checkbox in
> [Decision](#decision) is ticked and the exact version is recorded.

Dataset availability, licensing, spatial/temporal resolution, ensemble
configuration and access restrictions **must be verified for the exact version
used** — not assumed from a paper, a tutorial, or a blog post.

---

## Decision

- [x] Primary dataset selected — ERA5 (coarse + climatology), ERA5-Land (fine reference)
- [ ] Exact version / release recorded — CDS does not version these products; record access date instead
- [x] Licence and access terms recorded — both CDS licences accepted (2026-09-28)
- [x] Variable table documented — see [Confirmed datasets](#confirmed-datasets)
- [x] Spatial and temporal resolution confirmed — 0.25° / 0.10°, 3-hourly
- [x] Ensemble configuration confirmed (explicitly N/A — deterministic reanalysis)
- [ ] `configs/data.yaml` filled in — grid resolutions are set; source/version/variables still `null`

Unticked boxes keep their `null` values and downstream stages run on synthetic or
prepared sample data.

---

## Candidates

Candidates from the problem statement. **None are verified yet.**

| Candidate | Intended role | Status | Notes |
|---|---|---|---|
| ERA5 (Copernicus) | Coarse reanalysis + climatological baseline | **Confirmed** | `reanalysis-era5-single-levels`; 0.25° verified from the local files. |
| ERA5-Land (Copernicus) | Fine reference for downscaling | **Confirmed** | `reanalysis-era5-land`; 0.10°; **land-only** (ocean cells are NaN). |
| IMDAA | Indian-region reanalysis | Unverified | Regional — verify coverage and resolution. |
| NWP forecast products | The medium-range forecast input | Unverified | Access path not yet established. |
| Ensemble (EPS) data | Uncertainty / ensemble agreement | Unverified | Optional; only if access is granted. |
| Historical extreme events | Reproducible case studies | Unverified | Must have enough reference data to score against. |

> Where each candidate lives, its direct download link and the exact access
> steps are documented in [`docs/dataset_sources.md`](dataset_sources.md).

> Do not write a resolution (e.g. "12 km" or "0.25°") anywhere until the chosen
> product's documentation has been read and the number recorded here.

---

## Variables

`configs/data.yaml` currently has an **empty** `variables:` list. It stays empty
until the confirmed dataset's variable table is documented here.

Likely needed, expressed as *roles* rather than names — the actual dataset short
name must be recorded per dataset:

| Role | Used by | Notes |
|---|---|---|
| Surface temperature | detection, precursors | |
| Precipitation | detection, downscaling | Usually accumulated — verify units and accumulation window |
| Mean sea-level pressure | detection, precursors | |
| Geopotential height (mid-level) | detection, precursors | Verify level |
| Relative humidity | precursors | Verify level |
| Specific humidity | precursors | Verify level |
| Wind components (u, v) | precursors, downscaling | Verify height |
| Vorticity | precursors | Derived or provided — record which |

**Rule:** record the exact variable name, level, units and any accumulation
semantics. A silently mismatched accumulation window is the most common source of
a wrong precipitation anomaly.

---

## Grid and time

| Property | Value | Verified? |
|---|---|---|
| Coarse input resolution | 0.25° (~28 km), ERA5 | **yes** |
| Fine (localization target) resolution | 0.10° (~11 km), ERA5-Land (factor 2.5, not the 12→5 km target) | **yes** |
| Latitude range | event-specific — `bay_of_bengal` 5–25, `north_india` 20–35 | **yes** |
| Longitude range | event-specific — `bay_of_bengal` 80–95, `north_india` 68–90 | **yes** |
| Temporal range | amphan 2020-05-16→21 (48×3 h); heatwave 2022-05-01→10 (80×3 h); climatology `north_india` 1991–2020 full May (248×30 = 7440 samples/cell); climatology `bay_of_bengal` 1991–2020 **18 May only** (8×30 = 240 samples/cell, legacy window — see below) | **yes** |
| Forecast step frequency | 3 h | **yes** |
| Ensemble members | N/A — deterministic reanalysis | **yes** |

## Confirmed datasets

| | ERA5 | ERA5-Land |
|---|---|---|
| CDS dataset | `reanalysis-era5-single-levels` | `reanalysis-era5-land` |
| Resolution | 0.25° (~28 km) | 0.10° (~11 km) |
| Variables | `u10`, `v10`, `t2m`, `msl`, `tp` | `t2m`, `u10`, `v10`, `tp` (no `msl` — carries `surface_pressure`) |
| Cadence | 3-hourly | 3-hourly |
| Local files | `data/raw/era5_<event>.nc` | `data/raw/era5_land_<event>.nc` |
| Fetcher | `python -m src.data.cds_fetch` | `python -m src.data.land_fetch` |
| Coverage | global | **land only** — ocean cells are NaN |
| Access date | 2026-09-25 | 2026-09-28 |

> **ERA5-Land is land-only.** The Amphan domain is ~70 % NaN in the fine field, so
> every metric that needs paired cells is uncomputable there and comes back `null`
> rather than quoted off a third of the domain. The peak-overlap metrics *are*
> computable: the interpolation baseline keeps only **9 %** of the fine precipitation
> peak with **zero** p99 footprint overlap (D5), and the *learned* model refuses the
> event outright on its holdout-coverage floor (D3–D4). See `docs/experiments.md`.

> **Climatological baseline — one region each, and they do not overlap.**
>
> | Region | Files | Domain | Window | Samples/cell |
> |---|---|---|---|---|
> | `bay_of_bengal` (Amphan) | `era5_clim_1991.nc` … `era5_clim_2020.nc` (legacy, un-suffixed names) | lat 5–25, lon 80–95 | **18 May only**, 8×3 h | 240 |
> | `north_india` (heatwave) | `era5_clim_north_india_1991.nc` … `_2020.nc` | lat 20–35, lon 68–90 | full May, 248×3 h | 7440 |
>
> Fetch with:
>
> ```bash
> python -m src.data.cds_fetch --climatology --all-regions --years 1991-2020
> ```
>
> **Each event is scored against its own region** (`EVENT_REGIONS` in
> `src/detection/anomaly_detection.py`). Pooling them is not an option: the two
> domains differ, so a shared baseline would measure one event against the other's
> climate — and `xr.concat` over files of different extents produces a union grid with
> NaN in each region's empty part. The old code globbed `era5_clim_*.nc` with no region
> filter, which is exactly what would have happened once a second region was fetched.
>
> `MIN_CLIMATOLOGY_COVERAGE = 0.9` still **refuses** an event whose domain the
> baseline does not cover (rather than reporting "no anomaly" where no comparison was
> possible); it now correctly passes for the heatwave, and remains armed as the guard.
>
> **Known gap: the Bay of Bengal baseline is the single-day window.** Those 30 files
> carry 8 time steps per year (one calendar day), the version the `cds_fetch` docstring
> itself warns against — 240 samples, all from 18 May, so the spread is day-to-day
> weather rather than a full climatological distribution. The heatwave baseline is the
> corrected full-May window (7440 samples). Re-fetch `bay_of_bengal` with
> `--all-regions --days 1-31` to lift this; it would change every Amphan z-score, so
> it is recorded rather than silently replaced.

---

## Reproducibility requirements

Every dataset use must record:

- dataset name and **exact version / release date**
- the access date
- the variable list actually downloaded
- the spatial and temporal subset requested
- the preprocessing applied (see `src/data/`)
- a random seed if any sampling was involved

The demo must run **offline** from `data/samples/`, so the chosen historical case
has to be extracted and committed rather than downloaded live.

---

## Rules

1. Never invent a resolution, variable name or time range.
2. Never claim a dataset is usable before the access terms have been checked.
3. Never commit raw archives or credentials — see `.gitignore` and `.env.example`.
4. Record the version, not just the name: "ERA5" alone is not reproducible.
