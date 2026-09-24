# ARAVINTH — THREAT-X

> **Primary:** 12 km → ~5 km Downscaling (co-owned with Navaneedan)
> **Secondary:** Weather Data & Preprocessing
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 1 DATA (with Pushpa), 8 DOWNSCALING (with Navaneedan) |
| **Milestones** | M1 (with Pushpa), M7 (with Navaneedan) |
| **Main artifact** | Paired coarse/fine data · preprocessing scripts · baseline downscaler · event samples · experiment configuration |
| **PPT** | Slides 2–6: data/downscaling evidence (Slide 4 co-owner with Hariharan + Pushpa) |
| **Demo segments** | 20–45s load data + climatology · 150–175s localize/downscale |
| **Stack** | Python, NumPy, Pandas, Xarray, Dask, NetCDF4/cfgrib, PyTorch, CUDA, Cartopy |

You are **first in the build order**. M1 gates everything — nobody can detect, track or
downscale anything until clean, reproducible inputs exist. Treat this as the highest
priority, and get a loader working on *something* before chasing the ideal dataset.

---

## 1. Data — what to actually build (`src/data/`)

### Step 1 — Loader (`src/data/loaders.py`)

- Read ERA5 / IMDAA / NWP products from **NetCDF or GRIB** (`xarray` + `dask` for chunked,
  lazy loading — never load a full 4D field into memory).
- Keep the loader **pluggable**: the dataset is not confirmed, so take the source and
  variable mapping as parameters rather than hard-coding an ERA5 shape.
- **Exit condition:** the same call returns the same array twice, on a machine with no
  internet.

### Step 2 — Preprocessing (`src/data/preprocessing.py`)

| Task | Requirement |
|---|---|
| Standardize coordinates | One lat/lon convention across every source. Record whether latitude is ascending. |
| Standardize timestamps | ISO 8601 **UTC** everywhere. Forecast steps and analysis times are different things — don't conflate them. |
| Align spatial grids | Coarse and reference on a defensible common grid before pairing |
| Align temporal grids | Match forecast lead times to reference valid times |
| Handle missing values | Document the policy; never silently `fillna(0)` on a precipitation field |
| Units | Record units per variable. **Precipitation accumulation windows differ between products** — this is the most common source of a wrong anomaly. |

### Step 3 — Climatological baseline (`src/data/climatology.py`)

- With Pushpa. Rolling seasonal climatology from the reanalysis.
- Output feeds `src/detection/` — a value that is extreme in one region or season is normal
  in another, so absolute thresholds are not acceptable.

### Step 4 — Event extraction and pairing

- Extract the selected historical case into `data/samples/` so the demo runs **offline**.
- Build **paired coarse/fine** arrays — this is the M1→M7 handoff and the input to every
  downscaling experiment.
- **Exit condition (M1):** clean inputs, climatology baseline, selected event, reproducible loader.

---

## 2. Downscaling support — what to actually build

- Build the **first baseline interpolation/resampling method** so the team always has a
  working comparison. Coordinate with Navaneedan so `src/downscaling/baseline.py` has one
  owner per function, not two people editing the same file.
- Provide the **12 km input field and the reference field** used for the downscaling
  comparison. Navaneedan cannot score a model without your reference.
- Support **batching and GPU experiments** — get paired patches into `DataLoader`s with a
  fixed seed.
- Generate **before/after visual evidence** for historical extremes.
- **Exit condition (M7):** interpolation/learned baseline + extreme-preservation metric exist.

---

## 3. Experiment configuration and reproducibility

You own experiment hygiene. For every run, record:

- dataset name **and exact version/release date**
- access date and the exact variable list
- spatial and temporal subset requested
- preprocessing applied
- random seed
- the config used

> **Rule:** "ERA5" alone is not reproducible. A number without its config cannot be
> defended in judge Q&A.

Update `configs/data.yaml` — every `null` in there is waiting for you. Fill in
`coarse_resolution_deg` and `fine_resolution_deg` as soon as they are verified; that single
change unblocks the whole downscaling track.

---

## Evidence you must save

- [ ] Preprocessing scripts (committed, runnable)
- [ ] Aligned / model-ready datasets
- [ ] Paired coarse + fine sample
- [ ] Baseline downscaler output
- [ ] Coarse/reference/refined comparison plot
- [ ] Error / extreme-value comparison
- [ ] Experiment configuration + seed
- [ ] Reproducible event loader (offline)

---

## PPT contribution

| Slide | Your input |
|---|---|
| 2 IDEA | Coarse-to-localized concept — how a coarse global field becomes a precise local threat |
| 3 TECHNICAL | Dataset pipeline + downscaling inputs block |
| 4 FEASIBILITY | **Co-owner.** Data availability, data volume, storage and compute feasibility. Be honest about what is obtainable. |
| 5 IMPACT | Historical event evidence with measured output |
| 6 REFERENCES | ERA5 / IMDAA / downscaling references — add rows to `docs/references.md` first |

**Required evidence (from the team checklist):** dataset/data-flow evidence · coarse/fine
pair · baseline comparison · experiment settings.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 20–45s | Load data + climatology (with Pushpa) | Raw/coarse field + baseline climatology |
| 150–175s | Localize/downscale (with Navaneedan) | 12 km → ~5 km field + extreme-preservation metric |

**Hard requirement:** the event must load **without internet access**. Keep a prepared case
rather than relying on a live download — if the venue Wi-Fi fails, the demo still runs.

---

## Fallback

| If this fails | Fall back to |
|---|---|
| Chosen dataset unavailable / access denied | A smaller prepared sample committed to `data/samples/` |
| Paired fine-resolution reference unobtainable | Interpolation baseline scored on the coarse grid; label it as no-reference |
| Files too large for memory | `xarray` + Dask chunking, spatial/temporal subset |
| Live download fails at the venue | Prepared offline case + saved comparison figures |
| GPU unavailable | Run batching on CPU; reduce patch count and say so |

---

## Definition of done

- **DATA:** a script/notebook loads the same case reproducibly; variables and time window are documented
- **DOWNSCALING:** baseline exists; coarse/refined comparison and extreme-preservation metric exist
- Another teammate must be able to run the pipeline and reproduce the **same** event
  input/output
- Plus the team-wide rule: *the artifact runs from a reproducible input, produces a saved
  output, has at least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Data
- [ ] Dataset selected and version recorded
- [ ] Variables selected and units documented
- [ ] NetCDF/GRIB loader works
- [ ] Coordinates standardized
- [ ] Timestamps standardized (UTC)
- [ ] Missing-value policy implemented
- [ ] Spatial alignment works
- [ ] Temporal alignment works
- [ ] Precipitation accumulation semantics verified
- [ ] Climatology baseline generated
- [ ] `configs/data.yaml` filled in
- [ ] Model-ready dataset saved

### Historical event
- [ ] Event selected
- [ ] Metadata documented
- [ ] Event crop created in `data/samples/`
- [ ] Event visualization created
- [ ] Reproducible offline event loader created

### Downscaling support
- [ ] Paired coarse/fine arrays built
- [ ] Baseline downscaler works (shared with Navaneedan)
- [ ] 12 km input + reference field delivered
- [ ] Coarse/reference plot created
- [ ] Coarse/refined plot created
- [ ] Error plot created
- [ ] Extreme-value comparison created
- [ ] Batching + fixed seed implemented

### Reproducibility
- [ ] Configuration saved
- [ ] Random seed recorded
- [ ] Dataset version recorded
- [ ] Experiment script committed
- [ ] Results saved

### PPT
- [ ] Slide 2 coarse-to-localized visual supplied
- [ ] Slide 3 data pipeline diagram supplied
- [ ] Slide 4 data/compute feasibility evidence supplied
- [ ] Slide 5 event evidence supplied
- [ ] Slide 6 references supplied
- [ ] No invented numbers

### Demo
- [ ] Event loads without internet dependency
- [ ] 12 km field displayed
- [ ] Downscaled output displayed
- [ ] Comparison saved as fallback
