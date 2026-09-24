# PUSHPA — THREAT-X

> **Primary:** Extreme Weather Anomaly Detection
> **Secondary:** Weather Data & Preprocessing (with Aravinth)
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 1 DATA (with Aravinth), 2 ANOMALY, 9 VALIDATION (with Navaneedan) |
| **Milestones** | M1 (with Aravinth), **M2**, M8 (with Navaneedan) |
| **Main artifact** | Anomaly detector · anomaly masks · clustered regions · threshold configuration · event maps · threshold-sensitivity evidence |
| **PPT** | Slides 2–6: anomaly evidence / references (Slide 6 co-owner with Varnika) |
| **Demo segments** | 20–45s load data + climatology · 45–70s detect anomaly |
| **Stack** | Python, NumPy, Xarray, SciPy, scikit-learn, GeoPandas, Cartopy |

You produce the **input to everything downstream**. If your detector is noisy, tracking
chases ghosts and the transition layer learns nonsense. Your stage is also the one that
decides whether the system is *trustworthy*, so documenting your thresholds matters as much
as the code.

---

## 1. Anomaly detection — what to actually build (`src/detection/`)

### The required chain

```text
Weather Field
     ↓
Climatology Difference
     ↓
Threshold Mask
     ↓
Candidate Region
```

Never skip the climatology step. **Absolute values are not acceptable** — a value extreme in
one region or season is normal in another. The baseline comes from
`src/data/climatology.py` (M1, built with Aravinth).

### Step 1 — Baseline detector (`anomaly_detector.py`)

Build the **interpretable** version first:

- z-score / standardized anomaly
- percentile thresholds
- seasonal climatology difference

Read every parameter from `configs/model.yaml` → `anomaly:`. `zscore_threshold`,
`percentile_threshold`, `min_cluster_cells` and `climatology_window_days` are all **`null` on
purpose** — tune them against historical events and document the choice.

> **Keep the first detector interpretable and reproducible.** Advanced learning is
> **optional** and only worth attempting after the baseline works and is measured.

### Step 2 — Thresholding (`thresholding.py`)

Produce the spatial anomaly mask. Record:

- which threshold was used and why
- how it behaves near the threshold (a hard cutoff creates flicker between time steps,
  which damages tracking)
- whether any smoothing is applied — and if so, **prove it is not erasing the extreme peak**

### Step 3 — Clustering (`clustering.py`)

Convert the mask into **candidate regions**. For each candidate compute: area, centroid,
maximum intensity, mean intensity, shape, duration, growth, movement.

**Exit condition (M2):** mask + candidate regions + threshold configuration. Hand candidates
to Sachin's Threat Object / tracker.

### Step 4 — Event maps

Produce annotated maps for **at least one historical extreme event**. This is your slide 5
evidence and your demo asset.

---

## 2. Threshold sensitivity — the part that earns credibility

Run a **threshold sweep** and record how results move:

| Threshold | Precision | Recall | False alarm rate | Miss rate |
|---|---|---|---|---|
| *unset* | — | — | — | — |

Also required (`src/validation/detection_metrics.py`, with Navaneedan):

- **false positives checked** — what did the detector flag that wasn't an event?
- **missed events checked** — what did it miss entirely?

This is what lets you answer "why did you pick that threshold?" in judge Q&A with data rather
than intuition.

> **No invented numbers.** An empty sensitivity table is honest; a fabricated one is not.

---

## 3. Data support (with Aravinth)

- Help build and verify `src/data/` so the pipeline has clean, standardized inputs.
- Standardize the variables and time windows your detector requires.
- Handle missing values explicitly and document the policy.
- **Exit condition (M1, shared):** clean inputs, climatology baseline, selected event,
  reproducible loader.

---

## Evidence you must save

- [ ] Anomaly detector (committed)
- [ ] Anomaly mask (saved artifact, per time step)
- [ ] Clustered candidate regions
- [ ] Threshold configuration used
- [ ] Event map for the historical case
- [ ] Threshold-sensitivity evidence
- [ ] False-positive / missed-event analysis
- [ ] Reproducible detection output

---

## PPT contribution

| Slide | Your input |
|---|---|
| 2 IDEA | The **"needle in the haystack"** visual — how a huge multidimensional field becomes one candidate threat |
| 3 TECHNICAL | Anomaly + data-preprocessing stage block |
| 4 FEASIBILITY | **Co-owner.** Data availability and threshold-sensitivity risks — be honest that extremes are rare and thresholds are a trade-off |
| 5 IMPACT | **Alert-fatigue evidence.** A detector that fires everywhere is useless; show the false-alarm story |
| 6 REFERENCES | **Co-owner with Varnika.** ERA5 / IMDAA / anomaly-detection references — add rows to `docs/references.md` first |

**Required evidence (from the team checklist):** anomaly mask · threshold explanation ·
event map · validation/reference evidence.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 20–45s | Load data + climatology (with Aravinth) | Raw/coarse field + baseline climatology |
| 45–70s | Detect anomaly | Anomaly mask + candidate region |

**Show the chain live:** raw/coarse weather field → anomaly mask → candidate threat region.
Make the climatology comparison visible — that's what proves the detection isn't arbitrary.

---

## Fallback

| If this fails | Fall back to |
|---|---|
| Climatology baseline incomplete | Fixed percentile threshold on the case itself, clearly labelled as provisional |
| Detector too noisy | Raise the threshold and report the recall you traded away — don't hide it |
| Advanced detector untested | Ship the interpretable baseline; it is defensible, an untested model is not |
| Historical event data missing | Use the prepared sample case in `data/samples/`; document its limitations |

---

## Definition of done

- **ANOMALY:** mask and candidate region are saved; threshold/normalization choices are documented
- **VALIDATION (shared with Navaneedan):** at least one quantitative metric reported; degraded cases visible
- The same input must produce a **reproducible** mask/candidate region, and the threshold
  choice must be **explainable**
- Plus the team-wide rule: *runs from a reproducible input, produces a saved output, has at
  least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Data
- [ ] Historical baseline (climatology) selected and generated
- [ ] Variables selected
- [ ] Data loader works (with Aravinth)
- [ ] Time windows standardized
- [ ] Spatial grid standardized
- [ ] Missing values handled and documented
- [ ] Precipitation accumulation semantics verified

### Detection
- [ ] Z-score / percentile baseline implemented
- [ ] Thresholds read from `configs/model.yaml` (not hard-coded)
- [ ] Threshold documented with justification
- [ ] Spatial mask generated
- [ ] Clusters generated
- [ ] Candidate centroid generated
- [ ] Candidate footprint generated
- [ ] Candidate severity generated
- [ ] Handoff schema agreed with Sachin's tracker

### Validation
- [ ] Threshold sensitivity sweep run
- [ ] False positives checked
- [ ] Missed events checked
- [ ] Historical event tested
- [ ] Event map saved
- [ ] Detection metrics implemented (with Navaneedan)

### PPT
- [ ] Slide 2 "needle in the haystack" visual supplied
- [ ] Slide 3 detection + preprocessing pipeline supplied
- [ ] Slide 4 threshold/feasibility risk supplied
- [ ] Slide 5 alert-fatigue evidence supplied
- [ ] Slide 6 references supplied
- [ ] No invented numbers

### Demo
- [ ] Raw field loads offline
- [ ] Anomaly mask displays
- [ ] Candidate region displays
- [ ] Output passed to the tracker
