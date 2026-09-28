"""Reproducible end-to-end demo — the README's twelve steps, on real data.

The README defines the demonstration as a fixed sequence (``# 🧪 Reproducible
Demo``), and until now nothing ran it: each stage had its own CLI and the operator
was expected to chain them by hand and copy JSON between the steps. This module is
that sequence, executable::

    python -m src.shared.demo --event amphan
    python -m src.shared.demo --event heatwave --variable 2m_temperature
    python -m src.shared.demo --event amphan --only 1,2,4

Design rules, all of them the same rule
---------------------------------------
* **Real data or an explicit gap.** Every step runs against the local ERA5 /
  ERA5-Land archives. A step that cannot be computed on the available data says so
  and is skipped — it is never replaced by a synthetic stand-in. A demo that
  quietly substitutes a fabricated number is worse than one that names the gap,
  because the gap is fixable and the fabricated number is not.
* **A step failure does not kill the run.** Stages are wrapped so a missing
  climatology region or an unscorable pair degrades into a printed reason, not a
  traceback two minutes into a presentation.
* **Nothing is selected by eye.** The field frame is chosen by the documented
  direction of the extreme, the track is the one with the most detections, and the
  downscaling frame is the one the metric module already reported.
* **Severity is not invented.** ``configs/tracking.yaml -> severity_bands`` is
  still null, so the demo carries ``severity: null`` and says why, rather than
  printing a tier nobody derived.

Usage
-----
``--only`` takes a comma-separated list of step numbers, which is how you rehearse
just the part you are about to show.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["DEMO_EVENTS", "VARIABLES", "DemoContext", "main", "run"]

DEMO_EVENTS: dict[str, dict[str, str]] = {
    "amphan": {
        "variable": "mean_sea_level_pressure",
        "dataset_key": "era5_amphan",
        "why": (
            "Cyclone Amphan is a low-pressure extreme over the Bay of Bengal, and its "
            "event domain is the one the 1991-2020 climatology actually covers."
        ),
    },
    "heatwave": {
        "variable": "2m_temperature",
        "dataset_key": "era5_heatwave",
        "why": (
            "The May 2022 North India heatwave: the event the real ERA5 -> ERA5-Land "
            "downscaling pair is scored on."
        ),
    },
}

VARIABLES: dict[str, tuple[str, str, float]] = {
    # name -> (ERA5 short name, units, factor to canonical units)
    "total_precipitation": ("tp", "mm/3h", 1000.0),
    "2m_temperature": ("t2m", "K", 1.0),
    "mean_sea_level_pressure": ("msl", "hPa", 0.01),
    "10m_u_component_of_wind": ("u10", "m/s", 1.0),
    "10m_v_component_of_wind": ("v10", "m/s", 1.0),
}


@dataclass
class DemoContext:
    """State carried between steps, plus where the run's artefacts go."""

    event: str
    variable: str
    out_dir: Path
    plots_dir: Path
    dataset_key: str
    dataset: Any = None
    frame_index: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    trajectories: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def write(self, name: str, payload: Any) -> Path:
        """Save a step's output and remember it for the closing summary."""
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        self.artifacts[name] = path
        return path

    def figure(self, name: str) -> Path:
        return self.plots_dir / name


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _frame_index(values: Any, direction: str) -> int:
    """Index of the frame where the chosen extreme is strongest.

    ``direction`` comes from the detector's per-variable table: a cyclone's extreme
    is the deepest *low*, so searching for the maximum would plot the wrong frame
    for the one event we demo with a pressure field.
    """
    import numpy as np

    per_step = values.reshape(values.shape[0], -1)
    if direction == "low":
        return int(np.nanargmin(np.nanmin(per_step, axis=1)))
    return int(np.nanargmax(np.nanmax(per_step, axis=1)))


def _bbox_area_km2(box: dict[str, Any]) -> float | None:
    """Area of a detection's bounding box in km², as a footprint scale.

    An upper bound on the flagged footprint, not the flagged area: the detector
    reports a box, and calling a box an area would overstate it. Labelled as such
    wherever it is printed.
    """
    import numpy as np

    from src.shared.geo import haversine_km

    required = ("lat_min", "lat_max", "lon_min", "lon_max")
    if any(box.get(key) is None for key in required):
        return None
    height = float(haversine_km(box["lat_min"], box["lon_min"], box["lat_max"], box["lon_min"]))
    width = float(haversine_km(box["lat_min"], box["lon_min"], box["lat_min"], box["lon_max"]))
    return float(np.round(height * width, 1))


# ---------------------------------------------------------------------------
# The twelve steps
# ---------------------------------------------------------------------------


def step_01_select_event(context: DemoContext) -> None:
    """1. Select historical event."""
    from src.precursors.pipeline import KNOWN_EVENTS

    entry = KNOWN_EVENTS[context.dataset_key]
    print(f"event        {entry['event_name']}")
    print(f"type         {entry['event_type']}")
    print(f"threat id    {entry['threat_id']}  (from the precursor pipeline's case table)")
    print(f"dataset      {entry['dataset_file']}")
    print(f"variable     {context.variable}  [{VARIABLES[context.variable][1]}]")
    print(f"why          {DEMO_EVENTS[context.event]['why']}")
    context.write(
        "step01_event.json",
        {
            "event": context.event,
            "dataset_key": context.dataset_key,
            "variable": context.variable,
            "case": entry,
        },
    )


def step_02_load_sample(context: DemoContext) -> None:
    """2. Load forecast/reanalysis sample."""
    from src.data.loader import get_time_step_hours, load_dataset, spatial_resolution_deg

    entry_path = {
        "amphan": "data/raw/era5_amphan.nc",
        "heatwave": "data/raw/era5_heatwave.nc",
    }[context.event]

    dataset = load_dataset(entry_path)
    context.dataset = dataset

    short_name = VARIABLES[context.variable][0]
    if short_name not in dataset:
        raise KeyError(
            f"{entry_path} has no '{short_name}' variable; it carries "
            f"{sorted(dataset.data_vars)}. Fetch the variable first "
            "(python -m src.data.cds_fetch) or pick another --variable."
        )

    from src.detection.anomaly_detection import ANOMALY_DIRECTION

    direction = ANOMALY_DIRECTION.get(context.variable, "high")
    values = dataset[short_name].values
    times = dataset["valid_time"].values
    context.frame_index = _frame_index(values, direction)

    print(f"file         {entry_path}")
    print(
        f"grid         {values.shape[1]}x{values.shape[2]} @ "
        f"{spatial_resolution_deg(dataset):g} deg"
    )
    print(f"steps        {values.shape[0]} every {get_time_step_hours(dataset):g} h")
    print(f"window       {str(times[0])[:19]} -> {str(times[-1])[:19]} UTC")
    print(
        f"extreme      direction={direction}, frame {context.frame_index} "
        f"({str(times[context.frame_index])[:19]})"
    )

    context.write(
        "step02_sample.json",
        {
            "file": entry_path,
            "shape": list(values.shape),
            "resolution_deg": spatial_resolution_deg(dataset),
            "step_hours": get_time_step_hours(dataset),
            "window": [str(times[0])[:19], str(times[-1])[:19]],
            "variable": short_name,
            "direction": direction,
            "frame_index": context.frame_index,
        },
    )


def step_03_display_field(context: DemoContext) -> None:
    """3. Display meteorological field."""
    from src.downscaling.real_pair import as_field
    from src.shared.visualization import plot_field_panels

    short_name, units, scale = VARIABLES[context.variable]
    field = as_field(
        context.dataset, short_name, units, scale, context.frame_index
    ).ascending()
    timestamp = str(context.dataset["valid_time"].values[context.frame_index])[:19]

    print(
        f"field        {short_name} [{units}]  peak {field.peak:.2f}  "
        f"nan {field.nan_fraction:.1%}"
    )
    target = context.figure(f"step03_field_{context.event}.png")
    plot_field_panels(
        [(f"{context.event} {short_name}", field)],
        title=f"{context.event} — {short_name} at {timestamp} UTC",
        save_to=target,
    )
    print(f"[figure]     {target}")
    context.write(
        "step03_field.json",
        {
            "variable": short_name,
            "units": units,
            "frame_index": context.frame_index,
            "timestamp": timestamp,
            "peak": field.peak,
            "nan_fraction": field.nan_fraction,
            "figure": str(target),
        },
    )


def step_04_detect(context: DemoContext) -> None:
    """4. Detect anomaly."""
    from src.detection.anomaly_detection import detect_anomalies

    context.frames = detect_anomalies(context.event, context.variable)
    with_boxes = [frame for frame in context.frames if frame["boxes"]]
    if not with_boxes:
        raise RuntimeError(
            "the detector returned no regions for any frame; the run continues but "
            "every later stage needs detections, so treat this as a data problem "
            "rather than an absence of extreme weather"
        )

    ranked = sorted(
        (box for frame in with_boxes for box in frame["boxes"]),
        key=lambda box: box["peak_zscore"],
        reverse=True,
    )
    strongest = ranked[0]
    print(f"frames       {len(context.frames)} ({len(with_boxes)} with regions)")
    print(f"regions      {len(ranked)} total, {len(ranked) / len(context.frames):.1f} per frame")
    print(
        f"strongest    {strongest['peak_zscore']:.2f} sigma at "
        f"({strongest['centroid_lat']:.2f}, {strongest['centroid_lon']:.2f}), "
        f"{strongest['cell_count']} cells"
    )
    area = _bbox_area_km2(strongest)
    if area is not None:
        print(f"bbox area    {area:,.0f} km²  (box area, an upper bound on the footprint)")

    context.write(
        "step04_detections.json",
        {
            "frames": len(context.frames),
            "frames_with_regions": len(with_boxes),
            "regions": len(ranked),
            "strongest": strongest,
            "strongest_bbox_area_km2": area,
        },
    )


def _primary_track(tracks: list[Any]) -> Any | None:
    """The track with the most detections — deterministic, and not chosen by eye."""
    if not tracks:
        return None
    return max(tracks, key=lambda track: len(track.detections))


def step_05_threat_ids(context: DemoContext) -> None:
    """5. Generate Threat ID."""
    from src.tracking.pipeline import run_tracking_pipeline
    from src.tracking.tracker import ThreatTracker, TrackingConfig

    year = int(str(context.dataset["valid_time"].values[0])[:4])

    # The contract artefact the API and dashboard consume ...
    context.trajectories = run_tracking_pipeline(
        context.frames,
        output_path=context.out_dir / "step05_trajectories.json",
        default_year=year,
    )
    # ... and the detection-level series, which carries the intensity and footprint
    # per step that the Trajectory contract deliberately does not.
    tracker = ThreatTracker(config=TrackingConfig.from_config(), default_year=year)
    tracker.process_frames(context.frames)
    context.tracks = list(tracker.tracks)

    tracks = context.trajectories["trajectories"]
    print(f"threat ids   {len(tracks)} minted for {year}")
    for threat_id, payload in sorted(tracks.items()):
        steps = payload.get("historical_steps") or []
        print(f"  {threat_id}  {len(steps)} observations")

    primary = _primary_track(context.tracks)
    if primary is not None:
        print(
            f"primary      {primary.threat_id} with {len(primary.detections)} detections"
        )
    context.write(
        "step05_threat_ids.json",
        {
            "trajectories_file": str(context.out_dir / "step05_trajectories.json"),
            "ids": sorted(tracks),
            "primary": None if primary is None else primary.threat_id,
        },
    )


def step_06_trajectory(context: DemoContext) -> None:
    """6. Show threat trajectory."""
    from src.shared.contracts import Trajectory, TrajectoryPoint
    from src.shared.geo import bearing_deg, haversine_km, speed_kmh, step_distances_km

    primary = _primary_track(context.tracks)
    if primary is None:
        raise RuntimeError("no threat track was produced in step 5")

    centroids = primary.centroids
    timestamps = primary.timestamps
    if len(centroids) < 2:
        raise RuntimeError(
            f"{primary.threat_id} has {len(centroids)} observations; a trajectory needs 2"
        )

    step_hours = (
        (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0 / (len(timestamps) - 1)
    )
    distances = step_distances_km(centroids)
    speeds = speed_kmh(distances, step_hours)
    total = float(haversine_km(*centroids[0], *centroids[-1]))
    start_bearing = float(
        bearing_deg(centroids[0][0], centroids[0][1], centroids[1][0], centroids[1][1])
    )

    print(f"track        {primary.threat_id}, {len(centroids)} points")
    print(
        f"from         ({centroids[0][0]:.2f}, {centroids[0][1]:.2f}) "
        f"at {str(timestamps[0])[:19]}"
    )
    print(
        f"to           ({centroids[-1][0]:.2f}, {centroids[-1][1]:.2f}) "
        f"at {str(timestamps[-1])[:19]}"
    )
    print(f"displacement {total:,.1f} km net, initial bearing {start_bearing:.0f} deg")
    if speeds is not None:
        print(
            f"speed        {float(speeds.mean()):.1f} km/h mean, "
            f"{float(speeds.max()):.1f} km/h max (step {step_hours:g} h)"
        )

    trajectory = Trajectory(
        threat_id=primary.threat_id,
        historical_steps=[
            TrajectoryPoint(timestamp=stamp, centroid_lat=lat, centroid_lon=lon)
            for stamp, (lat, lon) in zip(timestamps, centroids, strict=True)
        ],
    )
    context.write("step06_trajectory.json", trajectory.to_dict())
    context.write(
        "step06_track.geojson",
        {
            "type": "Feature",
            "properties": {"threat_id": primary.threat_id, "source": "src.shared.demo"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in centroids],
            },
        },
    )
    print(f"[written]    {context.out_dir / 'step06_track.geojson'}")


def step_07_evolution(context: DemoContext) -> None:
    """7. Show intensity + footprint evolution."""
    from src.shared.visualization import plot_threat_series

    primary = _primary_track(context.tracks)
    if primary is None:
        raise RuntimeError("no threat track was produced in step 5")

    detections = primary.detections
    timestamps = [detection.timestamp for detection in detections]
    intensity = [detection.peak_zscore for detection in detections]
    footprint = [
        _bbox_area_km2(
            {
                "lat_min": detection.lat_min,
                "lat_max": detection.lat_max,
                "lon_min": detection.lon_min,
                "lon_max": detection.lon_max,
            }
        )
        for detection in detections
    ]

    print(f"series       {len(detections)} steps for {primary.threat_id}")
    for label, index in (("first", 0), ("middle", len(detections) // 2), ("last", -1)):
        area = footprint[index]
        sigma = intensity[index]
        print(
            f"  {label:6s} {str(timestamps[index])[:19]}  "
            f"intensity {'—' if sigma is None else f'{sigma:.2f} sigma'}  "
            f"footprint {'—' if area is None else f'{area:,.0f} km²'}"
        )

    target = context.figure(f"step07_intensity_footprint_{context.event}.png")
    plot_threat_series(
        timestamps,
        intensity,
        footprint,
        title=(
            f"{context.event} — {primary.threat_id} intensity and footprint "
            "(footprint is bounding-box area, an upper bound)"
        ),
        intensity_label="peak anomaly (sigma)",
        save_to=target,
    )
    print(f"[figure]     {target}")
    context.write(
        "step07_evolution.json",
        {
            "threat_id": primary.threat_id,
            "timestamps": [str(stamp) for stamp in timestamps],
            "intensity_sigma": intensity,
            "footprint_bbox_km2": footprint,
            "figure": str(target),
        },
    )


def step_08_precursors(context: DemoContext) -> None:
    """8. Show atmospheric precursor signals."""
    from src.precursors.pipeline import run_precursor_pipeline

    entry = run_precursor_pipeline(
        context.dataset_key,
        window_hours=48,
        output_dir=str(context.out_dir.parent.parent),
        plot_output_dir=str(context.plots_dir),
    )
    if not entry:
        raise RuntimeError("the precursor pipeline produced no series for this dataset")

    series = entry.get("series") or []
    print(f"threat       {entry.get('threat_id')}")
    print(f"steps        {len(series)} precursor time steps")
    if series:
        latest = series[-1]
        for key in (
            "pressure_tendency_3h_hpa",
            "wind_divergence_s1",
            "wind_speed_ms",
            "precipitation_rate_mm3h",
            "t2m_anomaly_k",
        ):
            value = latest.get(key)
            print(f"  {key:32s} {'—' if value is None else f'{value:.4g}'}")
    unavailable = entry.get("unavailable_variables") or []
    if unavailable:
        print(
            "unavailable  "
            + ", ".join(unavailable)
            + "\n             (the local archive is surface-only; upper-air precursors "
            "need pressure-level data)"
        )
    print(f"computed     {', '.join(entry.get('variables_analyzed') or []) or '—'}")
    summary = entry.get("explanatory_summary")
    if summary:
        print(f"summary      {summary}")
    context.write("step08_precursors.json", entry)


def step_09_transition(context: DemoContext) -> None:
    """9. Show transition intelligence."""
    from src.models.transition.transition_model import (
        load_precursor_entries,
        train_all_horizons,
    )

    entries = load_precursor_entries()
    matched = [
        (threat_id, entry)
        for threat_id, entry in sorted(entries.items())
        if entry.get("event_type") == context.event
    ]
    if not matched:
        raise KeyError(
            f"no precursor entry for event '{context.event}'; the file carries "
            f"{sorted(entries)}. Run: python -m src.precursors.pipeline --all-events"
        )

    threat_id, entry = matched[0]
    report = train_all_horizons({threat_id: entry})
    paths = report.save(context.out_dir / "transition" / threat_id)

    print(report.to_markdown())
    print()
    print(f"publishing   {'PUBLISHABLE' if report.can_publish() else 'NOT PUBLISHABLE'}")
    print(f"[written]    {paths['json']}")

    trained = report.trained_horizons()
    if trained:
        table = trained[0].metrics
        if table is not None:
            context.tables["transition"] = table
    context.write(
        "step09_transition.json",
        {
            "threat_id": threat_id,
            "can_publish": report.can_publish(),
            "trained_horizons": len(trained),
            "report": paths["json"],
        },
    )


def step_10_downscaling(context: DemoContext) -> None:
    """10. Show local downscaling (coarse -> fine)."""
    from src.downscaling import real_pair
    from src.models.downscaling import super_resolution

    print("Two models are scored against the same fine reference on the same grid:")
    print("  baseline — the deterministic interpolation rung (src/downscaling/baseline.py)")
    print("  learned  — one learned convolution layer (src/models/downscaling/)")
    print()

    for event in real_pair.EVENTS:
        print("-" * 78)
        print(f"{event}  [{real_pair.EVENTS[event][0]}]")
        print("-" * 78)

        result = real_pair.evaluate_event(event)
        if result["status"] != "scored":
            print(f"baseline     NOT SCOREABLE — {result['reason']}")
        else:
            table = result["table"]
            print(f"baseline     peak_preservation {_render(table, 'peak_preservation')}")
            context.tables.setdefault(f"downscaling_baseline_{event}", table)

        try:
            learned = super_resolution.train_and_evaluate(event)
        except (FileNotFoundError, ValueError) as error:
            print(f"learned      NOT SCORED — {error}")
            continue
        if learned["status"] != "scored":
            print(f"learned      NOT SCOREABLE — {learned['reason']}")
            continue

        table = learned["learned"]
        print(
            f"learned      peak_preservation {_render(table, 'peak_preservation')}  "
            f"(model {learned['model']['architecture']}, "
            f"alpha {learned['selection']['chosen']:g})"
        )
        comparison = learned["comparison"]
        print()
        print(f"{'metric':22s} {'baseline':>12s} {'learned':>12s} {'delta':>12s}")
        for name in sorted(comparison):
            row = comparison[name]
            delta = "—" if row["delta"] is None else f"{row['delta']:+.4g}"
            print(
                f"{name:22s} "
                f"{learned['baseline'].render(name, row['baseline']):>12s} "
                f"{learned['learned'].render(name, row['learned']):>12s} "
                f"{delta:>12s}"
            )
        print()
        print(f"gate         {learned['verdict']['status']} — {learned['verdict']['reason']}")
        if event == context.event:
            context.tables["downscaling"] = table
        context.write(f"step10_downscaling_{event}.json", {
            "event": event,
            "model": learned["model"],
            "selection": learned["selection"],
            "holdout": learned["holdout"],
            "comparison": comparison,
            "verdict": learned["verdict"],
        })


def _render(table: Any, name: str) -> str:
    return table.render(name, table.get(name))


def step_11_validation(context: DemoContext) -> None:
    """11. Show validation metrics."""
    from src.validation.evaluation import evaluate_validation

    if not context.tables:
        raise RuntimeError(
            "no metric table was produced by earlier steps, so there is nothing to "
            "gate; validation compares measured metrics against configured thresholds "
            "and does not invent either"
        )

    report = evaluate_validation(context.tables, subject=context.event)
    print(report.to_markdown())
    undecided = report.undecided()
    if undecided:
        print()
        print(
            "undecided    " + ", ".join(undecided) + "\n"
            "             (configs/validation.yaml -> gates thresholds are null; a gate "
            "is not a result until it is justified on measured distributions)"
        )
    paths = report.save(context.out_dir / "validation")
    print(f"[written]    {paths.get('markdown') or paths}")
    context.write(
        "step11_validation.json",
        {"tables": sorted(context.tables), "undecided": undecided},
    )


def step_12_alert(context: DemoContext) -> None:
    """12. Display final GIS alert."""
    from fastapi.testclient import TestClient

    from backend.main import app

    primary = _primary_track(context.tracks)
    client = TestClient(app)

    health = client.get("/health")
    print(f"health       {health.status_code}  pipeline stages: {health.json().get('pipeline_stages')}")

    alerts = client.get("/api/v1/alerts")
    print(f"alerts API   {alerts.status_code}")
    if alerts.status_code == 200:
        payload = alerts.json()
        entries = payload.get("alerts") or payload.get("items") or []
        print(f"served       {len(entries)} alert(s)")
        for item in entries[:3]:
            print(
                f"  {item.get('threat_id')}  {item.get('event_type')}  "
                f"severity={item.get('severity')}  confidence={item.get('confidence')}"
            )
        if not entries:
            print("             (the API's alert feed is fixture-backed until pipeline "
                  "output is wired into data/processed)")

    alert: dict[str, Any] = {
        "threat_id": None if primary is None else primary.threat_id,
        "event_type": context.event,
        "centroid": None if primary is None else list(primary.latest_centroid or ()),
        "window": None,
        "severity": None,
        "severity_basis": (
            "not assigned: configs/tracking.yaml -> severity_bands is null, so no band "
            "has been derived from the anomaly-severity distribution"
        ),
        "peak_anomaly_sigma": (
            None
            if primary is None
            else max(
                (d.peak_zscore for d in primary.detections if d.peak_zscore is not None),
                default=None,
            )
        ),
        "observations": 0 if primary is None else len(primary.detections),
        "transition": (
            "undecided (see step 9: publishing gate)"
        ),
        "downscaling": (
            "gate undecided (see step 10)"
        ),
        "served_from": "src.shared.demo (live artefacts), API alerts are fixture-backed",
    }
    if primary is not None and primary.detections:
        alert["window"] = [
            str(primary.detections[0].timestamp)[:19],
            str(primary.detections[-1].timestamp)[:19],
        ]
    path = context.write("step12_alert.json", alert)
    print()
    print(f"alert        {alert['threat_id']}  {alert['window']}")
    print(f"[written]    {path}")


STEPS: list[tuple[str, Callable[[DemoContext], None]]] = [
    ("1/12  Select historical event", step_01_select_event),
    ("2/12  Load forecast/reanalysis sample", step_02_load_sample),
    ("3/12  Display meteorological field", step_03_display_field),
    ("4/12  Detect anomaly", step_04_detect),
    ("5/12  Generate Threat ID", step_05_threat_ids),
    ("6/12  Show threat trajectory", step_06_trajectory),
    ("7/12  Show intensity + footprint evolution", step_07_evolution),
    ("8/12  Show atmospheric precursor signals", step_08_precursors),
    ("9/12  Show transition intelligence", step_09_transition),
    ("10/12 Show local downscaling", step_10_downscaling),
    ("11/12 Show validation metrics", step_11_validation),
    ("12/12 Display final GIS alert", step_12_alert),
]
"""The README's demo sequence, in order. One entry per numbered step."""


def run(context: DemoContext, only: set[int] | None = None) -> int:
    """Run the sequence, reporting gaps instead of dying on them."""
    completed: list[str] = []
    gaps: list[str] = []
    for index, (title, function) in enumerate(STEPS, start=1):
        if only is not None and index not in only:
            continue
        _rule(title)
        try:
            function(context)
        except Exception as error:  # a demo must survive a stage gap
            gaps.append(f"{title}: {type(error).__name__}: {error}")
            print(f"NOT COMPUTED IN THIS RUN — {type(error).__name__}: {error}")
        else:
            completed.append(title)

    _rule("demo summary")
    print(f"completed    {len(completed)} step(s)")
    for title in completed:
        print(f"  ok         {title}")
    if gaps:
        print(f"not computed {len(gaps)} step(s)")
        for gap in gaps:
            print(f"  gap        {gap}")
    print()
    print(f"artefacts    {context.out_dir}")
    print(f"figures      {context.plots_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the README's twelve-step demo on the local real archives."
    )
    parser.add_argument("--event", choices=sorted(DEMO_EVENTS), default="amphan")
    parser.add_argument(
        "--variable",
        choices=sorted(VARIABLES),
        help="Override the event's default variable.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated step numbers to run, e.g. 1,2,4. Default: all twelve.",
    )
    parser.add_argument("--output-dir", default="data/processed/demo")
    parser.add_argument("--plots-dir", default="data/processed/plots/demo")
    args = parser.parse_args(argv)

    event = DEMO_EVENTS[args.event]
    variable = args.variable or event["variable"]
    if variable not in VARIABLES:
        parser.error(f"unknown variable {variable!r}")

    context = DemoContext(
        event=args.event,
        variable=variable,
        out_dir=Path(args.output_dir) / args.event,
        plots_dir=Path(args.plots_dir) / args.event,
        dataset_key=event["dataset_key"],
    )
    context.out_dir.mkdir(parents=True, exist_ok=True)
    context.plots_dir.mkdir(parents=True, exist_ok=True)

    only = None
    if args.only:
        try:
            only = {int(part) for part in args.only.replace(" ", "").split(",") if part}
        except ValueError:
            parser.error("--only takes comma-separated step numbers, e.g. 1,2,4")
        if not only or any(step < 1 or step > len(STEPS) for step in only):
            parser.error(f"--only steps must be between 1 and {len(STEPS)}")

    print("threat-X reproducible demo")
    print(f"event        {context.event}  variable {context.variable}")
    print(f"steps        {'all' if only is None else sorted(only)}")
    return run(context, only)


if __name__ == "__main__":  # pragma: no cover - manual end-to-end run
    raise SystemExit(main())
