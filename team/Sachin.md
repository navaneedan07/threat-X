# SACHIN --- THREAT-X

## Primary Work

**Threat Tracking & Trajectory**

## Secondary Work

-   Interactive Dashboard / UI
-   GIS Visualization
-   PPT / Architecture

## Main Tasks

-   Generate persistent Threat IDs.
-   Match anomaly regions across forecast time steps.
-   Calculate centroid, footprint, area, movement direction, velocity
    and duration.
-   Track intensification, weakening and expansion.
-   Consider merge/split handling.
-   Build the GIS/dashboard layer.
-   Display threat position, trajectory, footprint and localized \~5 km
    region.
-   Build threat cards and evolution timelines.
-   Connect dashboard to Hariharan's API.
-   Own final architecture diagrams and visual consistency.

## Required Deliverables

-   Threat tracker
-   Threat state schema
-   Trajectory data
-   GIS map
-   Threat cards
-   Evolution timeline
-   Architecture diagrams
-   Final demo screen
-   Static/video fallback

## PPT Contribution

-   **Slide 1:** Complete visual design.
-   **Slide 2:** Detect → Track → Localize → Validate → Alert visual.
-   **Slide 3:** Full architecture diagram.
-   **Slide 4:** Deployment/fallback diagram.
-   **Slide 5:** Dashboard and impact visuals.
-   **Slide 6:** Clean reference layout.

## Demo

Show: 1. Weather input 2. Anomaly 3. Threat ID 4. Trajectory 5.
Footprint 6. Transition output 7. Localized field 8. Final alert

Keep a recorded/static fallback.

## Technology

-   Python
-   NumPy
-   SciPy
-   GeoPandas
-   Shapely
-   NetworkX
-   React / Next.js
-   TypeScript / JavaScript
-   Tailwind
-   Leaflet / Mapbox
-   GeoJSON
-   Figma / PowerPoint

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Tracking

-   [ ] Threat ID generated
-   [ ] Threat matching implemented
-   [ ] Centroid calculated
-   [ ] Footprint calculated
-   [ ] Area calculated
-   [ ] Velocity calculated
-   [ ] Direction calculated
-   [ ] Duration calculated
-   [ ] Growth/decay calculated
-   [ ] Trajectory saved
-   [ ] Merge/split behavior considered

### Dashboard

-   [ ] Map works
-   [ ] Threat marker works
-   [ ] Footprint layer works
-   [ ] Trajectory layer works
-   [ ] Threat card works
-   [ ] Severity displayed
-   [ ] Intensity displayed
-   [ ] Transition probability displayed
-   [ ] Expected window displayed
-   [ ] Confidence displayed when available
-   [ ] API connected

### PPT

-   [ ] Slide 1 design completed
-   [ ] Slide 2 visual completed
-   [ ] Slide 3 architecture completed
-   [ ] Slide 4 feasibility diagram completed
-   [ ] Slide 5 dashboard visual completed
-   [ ] Slide 6 reference layout completed

### Demo

-   [ ] End-to-end dashboard tested
-   [ ] Static screenshots saved
-   [ ] Demo recording saved
-   [ ] Fallback tested
-   [ ] 3--5 minute narrative rehearsed

## Definition of Done

A threat must be followable across time and the judge must understand
its movement, evolution, footprint and final alert visually.
