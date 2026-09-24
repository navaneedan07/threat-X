# VARNIKA --- THREAT-X

## Primary Work

**GNN & Spatio-Temporal Modelling**

## Secondary Work

**Historical Event Analysis + Visualization Support**

## Main Tasks

-   Define weather-grid/mesh nodes.
-   Define spatial and temporal edges.
-   Prototype spherical/mesh representation.
-   Define node features and temporal handling.
-   Prototype the GNN/message-passing architecture.
-   Define how GNN output connects to the Threat Object and tracker.
-   Replay selected historical events.
-   Produce comparison plots and visualization assets.
-   Maintain the research/reference matrix.
-   Do not claim a trained GNN until actual training and evaluation
    exist.

## Graph Structure

### Nodes

Potential features: - Latitude - Longitude - Weather variables - Anomaly
values - Time - Derived features

### Edges

Represent: - Spatial neighbors - Temporal connections - Mesh
connectivity

## Prototype Flow

**Weather Field → Graph/Mesh → Message Passing → Spatio-Temporal
Representation → Threat Output**

## Required Deliverables

-   Graph construction code
-   Graph/mesh visualization
-   GNN prototype
-   Actual metrics if trained
-   Historical replay scripts
-   Comparison plots
-   Research matrix

## PPT Contribution

-   **Slide 2:** Conceptual GNN contribution and historical evidence.
-   **Slide 3:** Spherical/mesh GNN architecture.
-   **Slide 4:** Research-backed feasibility and GPU requirements.
-   **Slide 5:** Historical replay visualization.
-   **Slide 6:** GNN/event-analysis literature matrix.

## Demo

If implemented: - Show graph/mesh - Show model output - Show real metric

If not implemented: - Show graph construction - Show
architecture/prototype - Clearly label it as prototype/concept - Do not
invent model results

## Technology

-   Python
-   Xarray
-   NumPy
-   PyTorch
-   PyTorch Geometric / DGL
-   NetworkX
-   CUDA
-   Matplotlib
-   Cartopy

## CHECKLIST --- CHECK IT WHEN YOU DO SOMETHING

> **When you complete something, check the box immediately.**

### Graph

-   [ ] Node definition completed
-   [ ] Node features defined
-   [ ] Spatial edges defined
-   [ ] Temporal edges defined
-   [ ] Mesh representation documented
-   [ ] Graph construction code works
-   [ ] Graph visualization created

### GNN

-   [ ] Architecture defined
-   [ ] Prototype implemented
-   [ ] Input/output documented
-   [ ] Training completed if attempted
-   [ ] Evaluation metric calculated if trained
-   [ ] Results saved
-   [ ] No unsupported performance claim

### Historical Analysis

-   [ ] Event selected
-   [ ] Event timeline created
-   [ ] Weather maps created
-   [ ] Anomaly maps created
-   [ ] Replay script created
-   [ ] Comparison plot created

### Integration

-   [ ] GNN output schema defined
-   [ ] Tracker interface defined
-   [ ] Threat Object interface defined
-   [ ] GNN output can be visualized

### Research

-   [ ] GNN papers collected
-   [ ] Global weather-model papers collected
-   [ ] Event-tracking references collected
-   [ ] Citation matrix created
-   [ ] PPT claims checked against sources

### PPT

-   [ ] Slide 2 GNN visual supplied
-   [ ] Slide 3 GNN architecture supplied
-   [ ] Slide 4 feasibility evidence supplied
-   [ ] Slide 5 historical replay supplied
-   [ ] Slide 6 literature matrix supplied

### Demo

-   [ ] Graph/mesh can be shown
-   [ ] Model output can be shown if implemented
-   [ ] Historical replay works
-   [ ] Static fallback saved

## Definition of Done

The GNN representation must be reproducible, its interface to the Threat
Object must be clear, and every claimed model result must have real
experimental evidence.
