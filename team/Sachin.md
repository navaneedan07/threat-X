# SACHIN — THREAT-X

> **Primary:** Threat Tracking & Trajectory
> **Secondary:** Interactive Dashboard / UI · GIS Visualization · PPT / Architecture
> Shared context, milestones and demo timings: [team/README.md](README.md)

---

## At a glance

| | |
|---|---|
| **Pipeline stages owned** | 3 THREAT OBJECT (with Navaneedan), 5 TRACKING, 10 API + GIS (with Hariharan) |
| **Milestones** | M3 (with Navaneedan), M9 (with Hariharan), **M10 lead** |
| **Main artifact** | Threat tracker · trajectory/state schema · GIS dashboard · map layers · architecture diagrams · final demo screen |
| **PPT** | Slide 1 (with Navaneedan) · **Slide 3 lead** · Slides 2, 4, 5 · Slide 6 layout |
| **Demo segments** | 00–20s select case · 70–95s create/track threat · 175–195s validate + alert · **coordinate the whole narrative** |
| **Stack** | Python, NumPy, SciPy, GeoPandas, Shapely, NetworkX · React/Next.js, TypeScript/JS, Tailwind, Leaflet/Mapbox, GeoJSON, Figma/PowerPoint |

You own the **"WHERE is it going?"** answer and the entire visual layer. You also coordinate
the 3–5 minute narrative — if the demo doesn't flow, it's your problem to solve.

---

## 1. Threat tracking — what to actually build (`src/tracking/`)

### Step 1 — Threat Object + persistent ID (`threat_object.py`)

Build this **at M3, with Navaneedan** — four people (you, Hariharan, Varnika, Navaneedan)
serialize this object, so it blocks progress until frozen. The field list is already declared
in `configs/tracking.yaml` → `threat_object_fields`.

The contract is shared from `src/shared/contracts.py`. Do not create a second definition.

### Step 2 — Tracker (`tracker.py`)

Match anomaly regions between consecutive time steps using combinations of:

- centroid distance
- footprint overlap (**IoU**)
- footprint similarity
- intensity similarity
- motion consistency
- temporal continuity

Handle **merge and split** — the lifecycle events in `configs/tracking.yaml` include
`MERGER`, `SPLIT`, `WEAKENING`, `RELOCATION`. Decide explicitly what a threat ID does when
two threats merge.

Association thresholds (`max_centroid_distance_km`, `min_iou`, `intensity_tolerance`) are
**`null` in the config on purpose** — derive them from observed displacement and overlap in
real data. Do not guess them.

**Exit condition (M3):** persistent IDs, centroid, footprint, velocity, duration.

### Step 3 — Trajectory and derived quantities (`trajectory.py`)

From `T0 → T1 → T2 → T3 → T4` derive:

- displacement · velocity (km/h) · direction (degrees clockwise from north)
- growth rate · duration · spatial expansion · intensification / weakening

Store the threat as a **time-indexed object** so the API and dashboard read the same thing.
**Exit condition:** trajectory saved.

---

## 2. GIS dashboard — what to actually build (`frontend/`)

> ⚠️ **Current state:** `frontend/` contains only a `.gitkeep`. There is no React app yet —
> scaffolding it is your first dashboard task. Nothing else in the repo blocks you.

### Map layers

- anomaly field
- threat footprint
- threat centroid
- trajectory
- uncertainty corridor
- localized impact zone (~5 km, from Navaneedan)
- administrative boundaries
- historical/reference footprint

### Threat card

Show: `threat_id` · type · severity · movement direction · speed · persistence ·
**transition probability** · expected window · ensemble agreement when available.

**Every number must come from the API.** If a value is `null`, render "—". Never display a
placeholder as though it were an inference result.

### The questions the dashboard answers

> WHERE? · WHERE IS IT GOING? · WHY IS IT CHANGING? · WHEN COULD IT ESCALATE? · HOW CERTAIN IS THE FORECAST?

Connect to Hariharan's API (M9). Keep a **static/video fallback** — the dashboard failing at
the venue is the single most visible way to lose the demo.

---

## 3. Architecture and visuals

You own the visual system end to end:

- the end-to-end architecture diagram (slide 3)
- the deployment / fallback diagram (slide 4)
- map legends and visual consistency
- the final demo screen
- **Slide 1 design**: PS ID, exact title, theme/category, Team ID/Name, clean title visual —
  **no architecture dump on the title slide**

---

## Evidence you must save

- [ ] Threat tracker (committed)
- [ ] Threat state / trajectory schema
- [ ] Trajectory data for the demo event
- [ ] GIS dashboard running
- [ ] Map with threat + trajectory + localized footprint
- [ ] Threat card rendering real API values
- [ ] Architecture diagram (slide 3)
- [ ] Dashboard screenshots (fallback)
- [ ] Demo recording

---

## PPT contribution

| Slide | Your input |
|---|---|
| 1 TITLE | **Co-lead** with Navaneedan. Clean title visual only. |
| 2 IDEA | **Co-lead.** System visual: Detect → Track → Localize → Validate → Alert |
| 3 TECHNICAL | **Lead.** One readable end-to-end pipeline diagram showing outputs and actual model/baseline status |
| 4 FEASIBILITY | Deployment/fallback diagram — what runs where, and what happens if it doesn't |
| 5 IMPACT | Dashboard / impact visual |
| 6 REFERENCES | Clean reference layout |

**Required evidence (from the team checklist):** architecture diagram · trajectory map ·
dashboard screenshot · final visual system.

> **PPT discipline:** every slide must be defensible in judge Q&A. No invented numbers, no
> "first/only" claims without evidence.

---

## Demo segment

| Time | You do | Judge sees |
|---|---|---|
| 00–20s | Select case (with Varnika) | Prepared historical extreme-weather event / forecast replay |
| 70–95s | Create/track threat (with Varnika) | Persistent Threat ID + path + centroid + footprint |
| 175–195s | Validate + alert (with Navaneedan, Hariharan) | PASS/DEGRADE + JSON alert + map |

**You coordinate the 3–5 minute narrative.** Rehearse the handoffs so ownership changes are
invisible to the judges. Keep a recorded/static fallback ready.

---

## Fallback

| If this fails | Fall back to |
|---|---|
| API not serving | Static JSON fixtures in `data/samples/`, same schema |
| Dashboard won't run at the venue | Recorded video + saved screenshots |
| Map tiles unreachable offline | Bundled GeoJSON boundaries, no basemap tiles |
| Localized (~5 km) footprint unavailable | Show the coarse footprint and label it clearly |
| Tracking thresholds unsuited to the case | Deliver trajectory for the selected demo event, document the limitation |

---

## Definition of done

- **THREAT TRACKING:** threat ID persists across time; centroid, footprint, direction, velocity and duration are saved
- **GIS:** the map shows threat, trajectory and localized footprint
- A threat must be **followable across time**, and a judge must **visually understand** its
  movement, evolution, footprint and final alert
- Plus the team-wide rule: *runs from a reproducible input, produces a saved output, has at
  least one evidence plot/table/screenshot, and is integrated into the final workflow.*

---

## Checklist

### Tracking
- [ ] Threat Object schema frozen (with Navaneedan)
- [ ] Threat ID generated and persistent
- [ ] Threat matching implemented
- [ ] Centroid calculated
- [ ] Footprint calculated
- [ ] Area calculated
- [ ] Velocity calculated (km/h)
- [ ] Direction calculated (deg clockwise from north)
- [ ] Duration calculated
- [ ] Growth / decay calculated
- [ ] Trajectory saved
- [ ] Merge / split behaviour decided and implemented
- [ ] Association thresholds derived from data, not guessed

### Dashboard
- [ ] React app scaffolded in `frontend/`
- [ ] Map renders
- [ ] Threat marker works
- [ ] Footprint layer works
- [ ] Trajectory layer works
- [ ] Anomaly field layer works
- [ ] Uncertainty corridor works
- [ ] Localized (~5 km) zone works
- [ ] Threat card works
- [ ] Severity displayed
- [ ] Intensity / footprint change displayed
- [ ] Movement displayed
- [ ] Transition probability displayed
- [ ] Expected window displayed
- [ ] Ensemble agreement displayed when available
- [ ] `null` renders as "—", not a fake number
- [ ] API connected

### PPT / visuals
- [ ] Slide 1 design completed
- [ ] Slide 2 visual completed
- [ ] Slide 3 architecture completed
- [ ] Slide 4 deployment/fallback diagram completed
- [ ] Slide 5 dashboard visual completed
- [ ] Slide 6 reference layout completed

### Demo
- [ ] End-to-end path tested
- [ ] Static screenshots saved
- [ ] Demo recording saved
- [ ] Fallback tested offline
- [ ] 3–5 minute narrative rehearsed with handoffs
