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

- [ ] Primary dataset selected
- [ ] Exact version / release recorded
- [ ] Licence and access terms recorded
- [ ] Variable table documented
- [ ] Spatial and temporal resolution confirmed
- [ ] Ensemble configuration confirmed (or explicitly N/A)
- [ ] `configs/data.yaml` filled in

Until every box above is ticked, `configs/data.yaml` keeps its `null` values and
downstream stages run on synthetic or prepared sample data.

---

## Candidates

Candidates from the problem statement. **None are verified yet.**

| Candidate | Intended role | Status | Notes |
|---|---|---|---|
| ERA5 (Copernicus) | Historical reanalysis + climatological baseline | Unverified | Global, needs CDS account. Verify licence. |
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
| Coarse (NWP) resolution | *unset* | no |
| Fine (localization target) resolution | *unset* | no |
| Latitude range | *unset* | no |
| Longitude range | *unset* | no |
| Temporal range | *unset* | no |
| Forecast step frequency | *unset* | no |
| Ensemble members | *unset* | no |

Filling these in is what unblocks the downscaling stage — it cannot choose an
interpolation baseline or build a paired coarse/fine dataset without the two
resolutions.

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
