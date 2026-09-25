# Dataset Sources — where to get the data, and how

Companion to [`docs/dataset.md`](dataset.md). This file records **where each
candidate dataset lives, the direct link, and the access method**. It does
**not** confirm any dataset: `docs/dataset.md`'s Decision checkbox and
`configs/data.yaml` stay unticked/unfilled until the licence, version and
variable table of the chosen product have actually been read and recorded.

> **Rule (from `docs/dataset.md`):** resolutions and variable names below are
> *reported by the provider's documentation* — re-read that documentation and
> record the exact numbers in `docs/dataset.md` before treating them as
> project facts. Never record a version you did not download yourself.

**Status legend:** 🟢 free, no approval needed · 🟡 free, account/registration
required · 🟠 access must be requested · ⚪ licence/terms must be verified

---

## 1. ERA5 — global reanalysis (climatological baseline)

**Role in Threat-X:** historical reanalysis + climatological baseline for
anomaly detection. **Status: 🟡 (free, CDS account required).**

| What | Direct link |
|---|---|
| ERA5 hourly, single levels (1940–present) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels |
| ERA5 hourly, pressure levels (1940–present) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels |
| ERA5 monthly averaged, single levels | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means |
| ERA5-Land (land surface, finer grid) | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land |
| CDS homepage / catalogue | https://cds.climate.copernicus.eu/ |
| CDS API setup (token + `.cdsapirc`) | https://cds.climate.copernicus.eu/how-to-api |
| Licence text | https://cds.climate.copernicus.eu/licences |

### How — web (fastest for a one-off case study)

1. Create a free Copernicus account and log in at https://cds.climate.copernicus.eu/.
2. Open the dataset link above → **Download** tab → pick variables, region
   (draw a box over India, e.g. lon 65–100, lat 5–40), time range and format
   (`zip` of NetCDF is the default) → **Submit request** → **My requests** →
   download when finished.

### How — CDS API (for scripted / reproducible downloads)

```bash
pip install cdsapi
```

1. Copy your personal URL + key from https://cds.climate.copernicus.eu/how-to-api
   into `~/.cdsapirc` (never commit it — it is a credential).
2. Request, e.g.:

```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "total_precipitation",
                     "mean_sea_level_pressure"],
        "year": "2023", "month": "07", "day": ["23", "24", "25"],
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        "area": [40, 65, 5, 100],          # N, W, S, E
        "grid": [0.25, 0.25],
        "format": "netcdf.zip",
    },
    "era5_case_20230723.nc.zip",
).download()
```

**Notes**

- Alternative bulk mirrors (useful if CDS queues are slow): Google ARCO-ERA5
  https://github.com/google-research/arco-era5 and AWS
  https://registry.opendata.aws/arco-era5/.
- Record the dataset DOI/version, access date and requested subset in
  `docs/dataset.md` once downloaded.
- Credentials: keep in `~/.cdsapirc`, or as `CDSAPI_URL` / `CDSAPI_KEY` in
  `.env` (see `.env.example`). `.env` is gitignored.

---

## 2. IMDAA — Indian-region reanalysis

**Role in Threat-X:** high-resolution regional reanalysis / baseline for the
Indian domain. **Status: 🟡/🟠 (free for research; access path varies — verify
terms with the provider).**

| What | Direct link |
|---|---|
| IMDAA Regional Reanalysis portal (NCMRWF) | https://rds.ncmrwf.gov.in/ |
| IMD Pune — Climate Research & Services data download | https://rcc.imdpune.gov.in/download.php |
| IMDAA system description (paper, JCLI) | https://journals.ametsoc.org/view/journals/clim/34/12/JCLI-D-20-0412.1.xml |

### How

1. Open https://rds.ncmrwf.gov.in/ — browse the IMDAA catalogue, register /
   log in if the portal requires it, select variables (surface + pressure
   levels), domain (Indian region) and period, then submit the request.
2. If the portal does not serve the required period or variables, use the
   IMD Pune RCC download page (https://rcc.imdpune.gov.in/download.php) or
   mail the data-request desk listed there — Indian meteorological data is
   typically released for research use on request.
3. Files arrive as NetCDF/GRIB → place them under `data/raw/` and process with
   `src/data/`.

**Notes**

- Reported as 12 km, hourly, satellite era (≈1979 onward) — **verify against
  the portal's documentation before recording it in `docs/dataset.md`.**
- Licence/terms: not yet verified → tick the licence checkbox in
  `docs/dataset.md` only after the provider states them in writing.

---

## 3. NWP forecast products (medium-range forecast input)

The forecast input the pipeline actually tracks. Three tiers, from easiest to
most "official":

### 3a. ECMWF open data (IFS + AIFS, free real-time) — 🟢

| What | Direct link |
|---|---|
| ECMWF Open Data overview | https://www.ecmwf.int/en/forecasts/datasets/open-data |
| Access real-time open data (direct file links) | https://www.ecmwf.int/en/forecasts/access-forecasts/access-real-time-open-data |
| AWS mirror (S3, no account needed) | https://registry.opendata.aws/ecmwf-forecasts/ |
| Python client `ecmwf-opendata` | https://github.com/ecmwf/ecmwf-opendata |

```bash
pip install ecmwf-opendata
```

```python
from ecmwf.opendata import Client
client = Client(source="ecmwf")
client.retrieve(
    type="fc", stream="oper", param=["2t", "tp", "msl"],
    date="2026-09-23", time=0, step=[0, 24, 48, 72, 96, 120],
    target="ifs_fc.grib2",
).download()
```

Real-time only (rolling recent days) — historical forecasts need MARS/ECDS.

### 3b. NCEP GFS (NOAA) — 🟢

| What | Direct link |
|---|---|
| NOMADS homepage (all NCEP model data) | https://nomads.ncep.noaa.gov/ |
| GFS 0.25° GRIB filter (subset by box/level) | https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25 |
| GFS raw directory listing | https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/ |
| AWS mirror | https://registry.opendata.aws/noaa-gfs-bdp-pds/ |
| GFS historical archive (UCAR) | https://data.ucar.edu/dataset/ncep-gfs-0-25-degree-global-forecast-grids-historical-archive |

### 3c. Indian NWP (NCMRWF / IMD) — 🟠 (terms to verify)

| What | Direct link |
|---|---|
| NCMRWF weather products / forecast dashboard | https://nwp.ncmrwf.gov.in/forecast-dashboard |
| IMD (India Meteorological Department) | https://mausam.imd.gov.in/ |
| NCMRWF homepage | https://ncmrwf.gov.in/ |

**How:** the NCMRWF dashboard and IMD portals publish model charts and selected
gridded products; full GRIB archives for NCMRWF/IMD models usually require a
research request — start the request early. If access is denied or slow, use
3a/3b (ECMWF open data / GFS) as the working NWP source and keep the loader
pluggable (`src/data/loaders.py`).

---

## 4. Ensemble (EPS) data — uncertainty / ensemble agreement

**Status: 🟡 (free, registration/licence acceptance required).**

| What | Direct link |
|---|---|
| TIGGE forecasts (ECMWF Data Store) | https://ecds.ecmwf.int/datasets/tigge-forecasts |
| TIGGE legacy retrieval page | https://apps.ecmwf.int/datasets/data/tigge/ |
| TIGGE project overview | https://www.ecmwf.int/en/research/projects/tigge |
| TIGGE licence (accept before download) | https://cds.climate.copernicus.eu/licences/tigge-licence |
| NOAA GEFS via NOMADS (GRIB filter) | https://nomads.ncep.noaa.gov/gribfilter.php?ds=gefs_atmos_0p50a |
| NOAA GEFS on AWS | https://registry.opendata.aws/noaa-gefs/ |

### How — TIGGE

1. Register/log in at https://ecds.ecmwf.int/ and accept the TIGGE licence.
2. Open https://ecds.ecmwf.int/datasets/tigge-forecasts → **Download** tab →
   choose centre (e.g. ECMWF), forecast reference time, lead times and
   variables → submit.
3. Alternative: MARS script retrieval via the TIGGE keywords documented on the
   same page (useful for scripted re-downloads).

### How — GEFS (ensemble over India without TIGGE)

Use the NOMADS GRIB filter link above (subset by geographic box, levels and
members), or read from the AWS bucket with `herbie` / `s3fs`.

**Notes**

- TIGGE covers multiple centres from 2006 onward — pick one centre and record
  it; do not mix centres without saying so.
- If ensemble access is not granted in time, set `ensemble.enabled: false`
  (current value) and treat `ensemble_agreement` as `null` on the Threat Object.

---

## 5. Historical extreme-event cases (reproducible replay + verification)

Each replay case needs (a) the historical forecast, (b) a reference/observed
field to score against, and (c) documented event facts.

### 5a. IBTrACS — tropical-cyclone best tracks (ground truth for tracks) — 🟢

| What | Direct link |
|---|---|
| IBTrACS product page (v4r01) | https://www.ncei.noaa.gov/products/international-best-track-archive |
| **Direct CSV — all North Indian Ocean storms** | https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv |
| CSV directory (basins, last-3-years, since-1980 subsets) | https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ |
| DOI | https://doi.org/10.25921/82ty-9e16 |

```bash
# North Indian Ocean best tracks (verify checksum/date on first use)
curl -L -o data/raw/ibtracs.NI.list.v04r01.csv \
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv
```

Use `v04r01` (basin `NI` for Indian cyclones) and record the version + access
date — IBTrACS is updated ~3×/week, so the version string matters.

### 5b. IMD gridded rainfall (observed reference over India) — 🟡

| What | Direct link |
|---|---|
| IMD 0.25° daily rainfall, NetCDF (1901–present) | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html |
| IMD 0.25° daily rainfall, binary format | https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html |

**How:** open the page → follow the per-year download links → NetCDF files,
one per year, under `data/raw/imd_rain/`. Registration/acknowledgement terms
apply — read the citation note on the page.

### 5c. RSMC New Delhi — official cyclone advisories & best track (case facts) — 🟢

| What | Direct link |
|---|---|
| RSMC New Delhi (IMD) | https://rsmcnewdelhi.imd.gov.in/ |

Use for documented cyclone case studies (dates, tracks, intensity) to select
replay events.

### 5d. CHIRPS — high-resolution rainfall for verification (optional) — 🟢

| What | Direct link |
|---|---|
| CHIRPS-2.0 data directory (global, NetCDF/TIFF) | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| CHIRPS product page | https://www.chc.ucsb.edu/data/chirps |

Useful as an independent precipitation reference where IMD gridded data is not
available (e.g. neighbouring regions).

---

## Quick reference table

| # | Dataset | Role in Threat-X | Where | Direct link | How | Status |
|---|---|---|---|---|---|---|
| 1 | ERA5 | climatological baseline | Copernicus CDS | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels | Web form or `cdsapi` | 🟡 |
| 2 | IMDAA | Indian-region baseline | NCMRWF / IMD Pune | https://rds.ncmrwf.gov.in/ | Portal request / email | 🟡/🟠 |
| 3 | ECMWF open data | NWP forecast input | ECMWF + AWS | https://www.ecmwf.int/en/forecasts/datasets/open-data | `ecmwf-opendata` / S3 | 🟢 |
| 4 | NCEP GFS | NWP forecast input (backup) | NOMADS + AWS | https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25 | GRIB filter / AWS | 🟢 |
| 5 | NCMRWF / IMD NWP | Indian NWP (preferred if granted) | NCMRWF / IMD | https://nwp.ncmrwf.gov.in/forecast-dashboard | Dashboard / request | 🟠 |
| 6 | TIGGE | multi-centre EPS | ECMWF ECDS | https://ecds.ecmwf.int/datasets/tigge-forecasts | Web form / MARS | 🟡 |
| 7 | NOAA GEFS | EPS (backup) | NOMADS + AWS | https://nomads.ncep.noaa.gov/gribfilter.php?ds=gefs_atmos_0p50a | GRIB filter / AWS | 🟢 |
| 8 | IBTrACS v4r01 | cyclone ground truth | NOAA NCEI | https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv | `curl` / browser | 🟢 |
| 9 | IMD 0.25° rainfall | observed reference | IMD Pune | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html | Yearly NetCDF links | 🟡 |
| 10 | CHIRPS-2.0 | rainfall reference (optional) | UCSB CHC | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ | HTTP directory | 🟢 |

---

## Access checklist (per dataset actually downloaded)

Copy into `docs/dataset.md` when a dataset is confirmed:

- [ ] Source name + **exact version/release** (e.g. `ERA5 hourly v5`, `IBTrACS v04r01`)
- [ ] Direct link used, and access date
- [ ] Licence / access terms read and recorded
- [ ] Variables + levels + units downloaded (paste the variable list)
- [ ] Spatial and temporal subset requested
- [ ] Preprocessing applied (`src/data/`)
- [ ] Download script or API request saved (so it can be re-run)
- [ ] Sample case extracted into `data/samples/` for the offline demo
- [ ] Raw archives and credentials **not** committed (`.gitignore`)

**Credentials stay out of git:** `~/.cdsapirc` for CDS, `.env`
(`CDSAPI_URL` / `CDSAPI_KEY`) for anything else — never in `configs/`.
