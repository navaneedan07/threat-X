# References

Citations for Threat-X. Every claim in the README, PPT and demo must point at a
row in this file.

> **STATUS: no citations verified yet.**
> A row marked *unverified* must not be quoted as support for a design decision.

---

## How to use this file

1. Add the row **before** citing the source anywhere else.
2. Read enough of the source to fill in what it actually claims — do not cite from
   a title or a search-result snippet.
3. Set `Verified` only after confirming the details (authors, year, venue, and that
   it says what you claim it says).
4. Note the **exact** dataset or library version, not just its name.

Link format: DOI or publisher URL. Avoid bare "available online" links.

---

## Research directions

### 1. Extreme Forecast Index (EFI)

The methodology behind comparing a forecast distribution against a model climate
distribution. Underpins the climatology-vs-forecast framing in `src/data/`.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| EFI-1 | Lalaurette, F. (2003), *Two proposals to improve the accuracy of Extreme Forecast Index*, ECMWF Newsletter | EFI methodology | no |
| EFI-2 | ECMWF Extreme Forecast Index product documentation | Operational EFI definition | no |

### 2. Ensemble prediction systems

Probabilistic forecasting and how ensemble spread represents uncertainty. Feeds
`ensemble_agreement` on the Threat Object.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| EPS-1 | *to be filled in* | ensemble spread -> uncertainty | no |

### 3. Object-based weather-event tracking

Storm/object tracking with persistent identity across timesteps. Directly informs
`src/tracking/`.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| TRK-1 | TITAN (Thunderstorm Identification, Tracking, Analysis and Nowcasting) literature | object identification + tracking | no |

### 4. Graph-based weather forecasting

Graph and mesh representations of atmospheric state. Informs `src/models/gnn/`.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| GNN-1 | GraphCast — learning skillful medium-range global weather forecasting | graph/mesh representation | no |

### 5. Diffusion / generative downscaling

Conditional generative super-resolution for weather fields. Informs the advanced
(speculative) downscaling path.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| DWN-1 | Recent conditional-diffusion weather downscaling literature | diffusion downscaling | no |
| DWN-2 | *to be filled in* | super-resolution baseline comparison | no |

### 6. Extreme-event evaluation

Metrics designed for rare, high-impact events rather than mean error.

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| EVL-1 | *to be filled in* | peak preservation / upper-tail metrics | no |

### 7. Indian atmospheric datasets

| Ref | Citation | Used for | Verified |
|---|---|---|---|
| DAT-1 | ERA5 dataset documentation (Copernicus CDS) | reanalysis / climatology | no |
| DAT-2 | IMDAA reanalysis documentation | regional reanalysis | no |
| DAT-3 | IMD / NCMRWF operational forecasting material | NWP input, regional context | no |

---

## Software

Pin versions when the first experiment runs and record them here.

| Library | Version used | Recorded |
|---|---|---|
| xarray | *unset* | no |
| dask | *unset* | no |
| PyTorch | *unset* | no |
| PyTorch Geometric | *unset* | no |
| scikit-learn | *unset* | no |
| FastAPI | *unset* | no |

---

## Rules

- Unverified citations must not appear in the PPT.
- Do not cite a source for a claim it does not make. "GraphCast uses GNNs" does
  not support "our GNN improves forecast skill".
- Record the **exact** dataset version, not the dataset name.
- If a component is Planned or Experimental, the reference supports the *approach*,
  not a result. Say so in the slide.
