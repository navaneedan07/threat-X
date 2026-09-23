# ARAVINTH --- SIH 26078

## Primary Work

**12 km → \~5 km Downscaling**

## Secondary Work

**Weather Dataset Pipeline & Preprocessing**

## Main Tasks

-   Prepare paired coarse/fine data.
-   Load and process ERA5/IMDAA/forecast NetCDF or GRIB data.
-   Standardize coordinates and timestamps.
-   Align spatial and temporal grids.
-   Handle missing values.
-   Extract/crop historical events.
-   Build the interpolation/resampling baseline.
-   Support learned downscaling experiments with Navaneedan.
-   Maintain experiment configuration and reproducibility.
-   Prepare event samples and comparison plots.

## Required Deliverables

-   Preprocessing scripts
-   Model-ready datasets
-   Historical event samples
-   Baseline downscaler
-   Coarse/reference/refined plots
-   Error/extreme-value comparisons
-   Experiment configuration

## PPT Contribution

-   **Slide 2:** Coarse forecast → localized threat concept.
-   **Slide 3:** Dataset pipeline and downscaling inputs.
-   **Slide 4:** Data availability, data-volume and compute feasibility.
-   **Slide 5:** Historical event evidence and measured output.
-   **Slide 6:** ERA5/IMDAA and downscaling references.

## Demo

Make the selected historical case reproducibly load: **Data →
preprocessing → event → 12 km field → downscaling**

Use a prepared case rather than relying on live downloads.

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Data

-   [ ] Dataset selected
-   [ ] Variables selected
-   [ ] NetCDF/GRIB loader works
-   [ ] Coordinates standardized
-   [ ] Timestamps standardized
-   [ ] Missing-value handling works
-   [ ] Spatial alignment works
-   [ ] Temporal alignment works
-   [ ] Event extraction works
-   [ ] Model-ready dataset saved

### Historical Event

-   [ ] Event selected
-   [ ] Metadata documented
-   [ ] Event crop created
-   [ ] Event visualization created
-   [ ] Reproducible event loader created

### Downscaling

-   [ ] Interpolation baseline works
-   [ ] Learned baseline investigated
-   [ ] Diffusion investigated if feasible
-   [ ] Coarse/reference plot created
-   [ ] Coarse/refined plot created
-   [ ] Error plot created
-   [ ] Extreme-value comparison created

### Reproducibility

-   [ ] Configuration saved
-   [ ] Random seed recorded
-   [ ] Dataset version recorded
-   [ ] Experiment script committed
-   [ ] Results saved

### PPT

-   [ ] Slide 2 evidence supplied
-   [ ] Slide 3 diagram supplied
-   [ ] Slide 4 feasibility evidence supplied
-   [ ] Slide 5 event evidence supplied
-   [ ] Slide 6 references supplied

### Demo

-   [ ] Event loads without internet dependency
-   [ ] 12 km field displayed
-   [ ] Downscaled output displayed
-   [ ] Comparison saved as fallback

## Definition of Done

Another teammate must be able to run the pipeline and reproduce the same
event input/output.
