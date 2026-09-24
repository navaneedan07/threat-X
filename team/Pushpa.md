# PUSHPA --- THREAT-X

## Primary Work

**Extreme Weather Anomaly Detection**

## Secondary Work

**Weather Data & Preprocessing**

## Main Tasks

-   Build an interpretable anomaly detector.
-   Start with z-score, percentile and climatology-based thresholds.
-   Generate spatial anomaly masks.
-   Cluster/identify candidate extreme regions.
-   Produce candidate centroid, footprint and severity.
-   Standardize variables and time windows.
-   Prepare event maps for historical extreme cases.
-   Test threshold sensitivity and false positives/missed events.
-   Pass candidate regions to the Threat Object/tracker.

## Required Detection Chain

**Weather Field → Climatology Difference → Threshold Mask → Candidate
Region**

## Required Deliverables

-   Anomaly detector
-   Anomaly mask
-   Clustered regions
-   Threshold configuration
-   Event maps
-   Threshold-sensitivity evidence
-   Reproducible detection output

## PPT Contribution

-   **Slide 2:** "Needle in the haystack" visual.
-   **Slide 3:** Data preprocessing + anomaly detection pipeline.
-   **Slide 4:** Data/threshold risks and fallback.
-   **Slide 5:** Historical event/anomaly evidence.
-   **Slide 6:** ERA5/IMDAA/anomaly-detection references.

## Demo

Show: **Forecast field → anomaly mask → candidate region**

Then pass the output to the tracker.

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Data

-   [ ] Historical baseline selected
-   [ ] Variables selected
-   [ ] Data loader works
-   [ ] Time windows standardized
-   [ ] Spatial grid standardized
-   [ ] Missing values handled

### Detection

-   [ ] Z-score/percentile baseline implemented
-   [ ] Threshold documented
-   [ ] Spatial mask generated
-   [ ] Clusters generated
-   [ ] Candidate centroid generated
-   [ ] Candidate footprint generated
-   [ ] Candidate severity generated

### Validation

-   [ ] Threshold sensitivity tested
-   [ ] False positives checked
-   [ ] Missed events checked
-   [ ] Historical event tested
-   [ ] Event map saved

### Integration

-   [ ] Anomaly output schema defined
-   [ ] Tracker consumes output
-   [ ] Threat Object can be created
-   [ ] Saved anomaly artifact available for demo

### PPT

-   [ ] Slide 2 anomaly visual supplied
-   [ ] Slide 3 detection pipeline supplied
-   [ ] Slide 4 risk/feasibility supplied
-   [ ] Slide 5 event evidence supplied
-   [ ] Slide 6 references supplied

### Demo

-   [ ] Raw field loads
-   [ ] Anomaly mask appears
-   [ ] Candidate region appears
-   [ ] Output passed to tracker

## Definition of Done

The same input must produce a reproducible anomaly mask/candidate region
and the threshold choice must be explainable.
