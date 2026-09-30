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

### GNN mesh datasets (66-node box)

The GNN stages do not use the event domains above. They use their own bounded box, in
their own request, so their cadence and feature definitions cannot be changed by the
event/climatology workflows:

| | Value |
|---|---|
| CDS dataset | `reanalysis-era5-single-levels` |
| Area | `[15.25, 75.75, 9.75, 81.25]` (N, W, S, E) — 23 × 23 cells at 0.25° |
| Cadence | **hourly**, 24 steps/day (not 3-hourly like the event archives) |
| Variables | `2m_temperature`, `2m_dewpoint_temperature`, `mean_sea_level_pressure`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_precipitation` |
| Nodes | the 66 exact ERA5 cells in `weights/gnn/outputs/node_table.csv` — lat 10.0–15.0 at 0.5°, lon 76.0–81.0 at 1.0° |
| Selection | exact coordinate lookup, `interpolation = false` |

| Purpose | Local file | Command |
|---|---|---|
| Frozen pilot / held-out week (2020-05-16 … 2020-05-22, **168 hourly steps**) | `data/raw/era5_gnn_pilot_20200516_20200522.zip` | `python -m src.data.cds_fetch --gnn-pilot` |
| Training months (stage 09) | `data/raw/era5_gnn_2020_03.zip` … `_2020_05.zip` | `python -m src.data.cds_fetch --gnn-month 2020-03 --gnn-month 2020-04 --gnn-month 2020-05` |
| Any other month / day | `data/raw/era5_gnn_<YYYY>_<MM>.zip` | `python -m src.data.cds_fetch --gnn-month 2020-06` / `--gnn-probe-date 2020-07-01` |

**Stage 09 split.** Train `2020-03-01T00:00Z … 2020-05-15T23:00Z` (**1824 hours**, 76 days);
test the pilot week `2020-05-16T00:00Z … 2020-05-22T23:00Z` (**168 hours**). The split is
enforced by `split_training_window`, which refuses to run unless the archives are
contiguous with the pilot and the training window ends on 2020-05-15T23:00Z — so the
pilot cannot drift into the training set.

**What is committed from these fetches.** The archives are gitignored (13 MB). The *derived*
pilot tensor, its timestamps and its metadata (`weights/gnn/outputs/era5_pilot/`, 230 KB) and
the two trained checkpoints (150 KB) **are** committed, so `08_evaluate_era5_pilot.py`, the
GNN tracking demo and both GNN test files run on a clone with no network and no CDS
account. Re-fetch and re-derive only if you want to reproduce them from source.

The pilot and the training months overlap in space and time deliberately: the pilot week is
a real week of the same box, held out of training, so the evaluation measures transfer
rather than interpolation of seen timestamps.

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
