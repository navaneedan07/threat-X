# data/ — what is actually on disk

> **Rule:** nothing in this tree may be synthetic unless a file is explicitly
> labelled as a fixture. Every `.nc` / `.grib2` / `.csv` below is real data from
> the source named beside it. See `docs/dataset.md` and `docs/dataset_sources.md`.

`data/raw/`, `data/processed/` and `data/climatology/` are **gitignored** — raw
archives are not committed. `data/samples/` **is** committed so the offline demo
and the API fixtures work from a fresh clone.

---

## Confirmed real data

| Path | Source | Version / run | License | Access date |
|---|---|---|---|---|
| `raw/era5_amphan.nc` | ERA5 single levels (Copernicus CDS) | 2020-05-16 → 05-21, 0.25°, 3-hourly | Copernicus licence | pre-existing |
| `raw/era5_heatwave.nc` | ERA5 single levels (Copernicus CDS) | 2022-05-01 → 05-10, 0.25°, 3-hourly | Copernicus licence | pre-existing |
| `raw/ibtracs.NI.list.v04r01.csv` | IBTrACS v04r01, NOAA NCEI (North Indian Ocean) | v04r01 | Public domain | 2026-09-26 |
| `raw/ecmwf_oper_2026092506_sfc.grib2` | ECMWF Open Data — IFS HRES 0.25° | run 2026-09-25 06Z | CC BY 4.0 — attribute ECMWF | 2026-09-26 |
| `raw/ecmwf_oper_2026092506_pl.grib2` | ECMWF Open Data — IFS HRES 0.25° pressure levels | run 2026-09-25 06Z, 500/850 hPa | CC BY 4.0 — attribute ECMWF | 2026-09-26 |

Direct links:

- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- IBTrACS NI CSV: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv
- ECMWF Open Data: https://www.ecmwf.int/en/forecasts/datasets/open-data

---

## What each file is for

- **ERA5 event archives** — the historical case (Cyclone Amphan, north-India
  heatwave). Real reanalysis; drives the precursor module. Load with
  `src/data/loader.py` (they are ZIP-wrapped CDS archives, despite the `.nc`
  extension).
- **IBTrACS CSV** — real cyclone best tracks (position + intensity). This is the
  ground truth the tracker must be scored against. Load with
  `src/data/ibtracs.py`.
- **ECMWF Open Data GRIB2** — the real **medium-range forecast input**. This is
  what makes "forecast" real: reanalysis has no lead time. Load with
  `src/data/open_data_fetch.load_open_data` (subsets the global field to the
  India domain).

---

## How it was fetched (reproducible)

```bash
# IBTrACS (no account)
curl -L -o data/raw/ibtracs.NI.list.v04r01.csv \
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv

# ECMWF open-data forecast (no account)
python -m src.data.open_data_fetch --source ecmwf --pressure-levels

# ERA5 event + climatology (REQUIRES CDS credentials — see below)
python -m src.data.cds_fetch --event amphan --pressure-levels
python -m src.data.cds_fetch --climatology --all-regions --days 1-31 --pressure-levels
```

### CDS credentials

ERA5 downloads use `cdsapi`, which reads credentials from `~/.cdsapirc` or the
`CDSAPI_URL` / `CDSAPI_KEY` environment variables (also read from a gitignored
`.env` via `src/data/cds_fetch.py`). **No credentials are committed.**

---

## Known issues

- `climatology/era5_clim_<year>.nc` are **stale**: they hold a single day
  (18 May) per year, and only the Bay-of-Bengal domain. A defensible baseline
  needs a seasonal window across both regions — run the `--all-regions`
  command above once CDS credentials are available. Do not build anomaly
  detection on the stale files.
- `data/samples/*.json` are **hand-written fixtures**, not model output. They
  exist so the API and dashboard can be developed before the pipeline produces
  real threat objects. Do not quote their numbers as results.
- ECMWF Open Data only mirrors recent runs (a rolling window of a few days), so
  the forecast files must be re-fetched for a live demo. The historical ERA5
  case remains the offline fallback.
