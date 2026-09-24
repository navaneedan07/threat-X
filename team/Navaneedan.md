# NAVANEEDAN --- THREAT-X

## Primary Work

**12 km → \~5 km Downscaling**

## Secondary Work

-   Validation & Evaluation
-   Threat Transition Intelligence
-   End-to-end integration

## Main Tasks

-   Build a reproducible 12 km → \~5 km downscaling pipeline.
-   Start with interpolation/resampling as the baseline.
-   Test a learned/CNN baseline if feasible.
-   Investigate conditional diffusion only if time, data and compute
    permit.
-   Measure extreme-value preservation, not just visual quality.
-   Define the Threat Object lifecycle: Formation → Intensification →
    Expansion → Peak → Decay.
-   Produce transition probability and expected transition window for
    defined horizons.
-   Integrate anomaly, tracking, GNN, precursor, downscaling and
    validation outputs.
-   Define PASS / DEGRADE / SUPPRESS validation gates.
-   Never hard-code or invent probabilities/metrics.

## Required Deliverables

-   Threat Object schema
-   Transition model/output
-   12 km vs refined field comparison
-   Extreme-preservation metrics
-   Validation table
-   Integration interfaces
-   Final demo output

## PPT Contribution

-   **Slide 2:** Threat Transition Intelligence, persistent Threat
    Object and lifecycle.
-   **Slide 3:** Transition + downscaling + validation architecture.
-   **Slide 4:** Feasibility, fallbacks and compute risks.
-   **Slide 5:** Quantitative results and localized-warning benefits.
-   **Slide 6:** Downscaling/validation references.

## Demo

Show: 1. Threat state 2. Transition probability 3. Expected transition
window 4. 12 km input 5. \~5 km output 6. Extreme-preservation metric 7.
PASS/DEGRADE/SUPPRESS result

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Threat Intelligence

-   [ ] Threat Object schema created
-   [ ] Lifecycle defined
-   [ ] Transition target defined
-   [ ] Transition dataset prepared
-   [ ] Transition model implemented
-   [ ] Actual probability output generated
-   [ ] Model evaluated
-   [ ] Expected transition window implemented

### Downscaling

-   [ ] Coarse input pipeline works
-   [ ] Interpolation baseline works
-   [ ] Learned baseline tested
-   [ ] Diffusion tested/investigated if feasible
-   [ ] Coarse vs refined plot created
-   [ ] Extreme-preservation metric calculated
-   [ ] Results saved

### Validation & Integration

-   [ ] Validation metrics implemented
-   [ ] PASS/DEGRADE/SUPPRESS rule defined
-   [ ] Historical validation completed
-   [ ] Threat Object connected to other modules
-   [ ] API receives final state

### PPT

-   [ ] Slide 2 content supplied
-   [ ] Slide 3 content supplied
-   [ ] Slide 4 content supplied
-   [ ] Slide 5 metrics supplied
-   [ ] Slide 6 references supplied

### Demo

-   [ ] Transition demo works
-   [ ] Downscaling demo works
-   [ ] Validation result works
-   [ ] Fallback prepared

## Definition of Done

The work is finished only when the artifact runs, produces saved
evidence, and is integrated into the final pipeline.
