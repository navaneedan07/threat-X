"""Build the evaluator-facing Threat-X dashboard from real ERA5 fields.

This is the single command that regenerates every demo figure:

    python frontend/dashboard.py --all-events --with-folium

It writes, under ``data/processed/plots/dashboard/``:

  * ``field_<event>.png``    -- real ERA5 field on a real coastline map, with the
                                detected anomaly boxes and tracked trajectory
                                drawn at the strongest frame
  * ``severity_<event>.png`` -- the tracked severity series
  * ``dashboard_<event>.html`` -- the interactive folium map (delegated to
                                ``frontend/visulisation.py``)
  * ``index.html``           -- one self-contained page embedding all of the
                                above plus the measured numbers

Two rules are inherited from the rest of the project and are enforced here:

* **Everything drawn is real.** The background field is the ERA5 reanalysis
  archive in ``data/raw/era5_<event>.nc``; the boxes and the trajectory are the
  detector's saved output in ``data/processed/detection/anomalies_<event>.json``.
  Nothing is interpolated, synthesised or smoothed for presentation.
* **No invented thresholds.** ``configs/tracking.yaml -> severity_bands`` is
  ``null``, so no point on any figure is labelled "severe" or "moderate". Colour
  encodes the measured z-score over the range actually observed for that event.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display required
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

# Running this file directly puts frontend/ on sys.path, not the repository root,
# and the panels read src.detection. Put the root first so the import resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.anomaly_detection import get_event_netcdf  # noqa: E402

DETECTION_DIR = PROJECT_ROOT / "data" / "processed" / "detection"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "plots" / "dashboard"


class EventSource:
    """Which ERA5 field to draw for an event, and how to present it."""

    def __init__(
        self,
        variable: str,
        short_name: str,
        label: str,
        unit: str,
        scale: float,
        offset: float,
        cmap: str,
    ) -> None:
        self.variable = variable
        self.short_name = short_name
        self.label = label
        self.unit = unit
        self.scale = scale
        self.offset = offset
        self.cmap = cmap

    def convert(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.offset


# The detector scored Amphan on the *low* tail of mean sea level pressure and the
# North India heatwave on the *high* tail of 2 m temperature
# (src/detection/anomaly_detection.py -> ANOMALY_DIRECTION). These are the same
# two fields, converted to the units the event is normally described in.
EVENT_SOURCES: dict[str, EventSource] = {
    "amphan": EventSource(
        variable="mean_sea_level_pressure",
        short_name="msl",
        label="Mean sea level pressure",
        unit="hPa",
        scale=0.01,  # Pa -> hPa
        offset=0.0,
        cmap="viridis",
    ),
    "heatwave": EventSource(
        variable="2m_temperature",
        short_name="t2m",
        label="2 m temperature",
        unit="\u00b0C",
        scale=1.0,
        offset=-273.15,  # K -> degC
        cmap="inferno",
    ),
}


# ---------------------------------------------------------------------
# Loading the real inputs
# ---------------------------------------------------------------------
def load_detections(event: str) -> dict:
    """Load the detector's saved output for one event."""
    path = DETECTION_DIR / f"anomalies_{event}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the detector first:\n"
            f"  python -m src.detection.anomaly_detection --event {event} "
            f"--variable {EVENT_SOURCES[event].variable}"
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_event_field(event: str, time_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (lat, lon, field, timestamp) for one ERA5 time index, in display units."""
    source = EVENT_SOURCES[event]
    dataset = xr.open_dataset(get_event_netcdf(event), engine="netcdf4")
    try:
        if source.short_name not in dataset.data_vars:
            raise KeyError(
                f"{source.short_name!r} is not in the {event} archive "
                f"(has {list(dataset.data_vars)})"
            )
        field = np.asarray(dataset[source.short_name].values, dtype=float)[time_index]
        latitudes = np.asarray(dataset["latitude"].values, dtype=float)
        longitudes = np.asarray(dataset["longitude"].values, dtype=float)
        timestamp = str(dataset["valid_time"].values[time_index])
    finally:
        dataset.close()
    return latitudes, longitudes, source.convert(field), timestamp


def strongest_frame(detections: dict) -> dict:
    """The frame holding the largest peak z-score across the whole event."""
    frames = detections.get("frames") or []
    if not frames:
        raise ValueError("detection output has no frames")
    return max(
        frames,
        key=lambda frame: max(
            (box["peak_zscore"] for box in frame["boxes"]), default=float("-inf")
        ),
    )


# ---------------------------------------------------------------------
# Static real-map panels
# ---------------------------------------------------------------------
def _base_map(axis, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
    """Add real coastlines, borders, land/ocean and a labelled graticule."""
    extent = [
        float(longitudes.min()),
        float(longitudes.max()),
        float(latitudes.min()),
        float(latitudes.max()),
    ]
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.OCEAN, facecolor="#dbe7f3", zorder=0)
    axis.add_feature(cfeature.LAND, facecolor="#f2efe6", zorder=0)
    axis.coastlines(resolution="50m", linewidth=0.8, zorder=4)
    axis.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#8a7f6d", zorder=4)
    graticule = axis.gridlines(
        draw_labels=True, linewidth=0.3, color="#5a5a5a", alpha=0.5, zorder=3
    )
    graticule.top_labels = False
    graticule.right_labels = False
    graticule.xlabel_style = {"size": 7}
    graticule.ylabel_style = {"size": 7}


def render_field_panel(event: str, out_path: Path) -> Path:
    """Real ERA5 field + detected boxes + tracked trajectory at the strongest frame."""
    source = EVENT_SOURCES[event]
    detections = load_detections(event)
    frame = strongest_frame(detections)
    latitudes, longitudes, field, timestamp = load_event_field(event, frame["time_index"])

    trajectory = detections.get("trajectory") or []
    z_values = [point["severity_zscore"] for point in trajectory]
    z_min, z_max = (min(z_values), max(z_values)) if z_values else (0.0, 1.0)

    figure = plt.figure(figsize=(9.5, 9.0))
    axis = plt.axes(projection=ccrs.PlateCarree())
    _base_map(axis, latitudes, longitudes)

    mesh = axis.pcolormesh(
        longitudes,
        latitudes,
        field,
        cmap=source.cmap,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    colorbar = figure.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.06, shrink=0.85)
    colorbar.set_label(f"{source.label} ({source.unit})", fontsize=9)

    for candidate in detections["frames"]:
        for box in candidate["boxes"]:
            axis.add_patch(
                plt.Rectangle(
                    (box["lon_min"], box["lat_min"]),
                    box["lon_max"] - box["lon_min"],
                    box["lat_max"] - box["lat_min"],
                    fill=False,
                    edgecolor="#1b1b1b",
                    linewidth=0.6,
                    alpha=0.35,
                    zorder=5,
                    transform=ccrs.PlateCarree(),
                )
            )

    for box in frame["boxes"]:
        axis.add_patch(
            plt.Rectangle(
                (box["lon_min"], box["lat_min"]),
                box["lon_max"] - box["lon_min"],
                box["lat_max"] - box["lat_min"],
                fill=False,
                edgecolor="#d62728",
                linewidth=2.0,
                zorder=7,
                transform=ccrs.PlateCarree(),
            )
        )

    if trajectory:
        axis.plot(
            [point["lon"] for point in trajectory],
            [point["lat"] for point in trajectory],
            color="#111111",
            linewidth=2.0,
            zorder=8,
            transform=ccrs.PlateCarree(),
            label="tracked trajectory",
        )
        peak = max(trajectory, key=lambda point: point["severity_zscore"])
        axis.plot(
            peak["lon"],
            peak["lat"],
            marker="*",
            markersize=18,
            markerfacecolor="#ff2d2d",
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            zorder=9,
            transform=ccrs.PlateCarree(),
            label=f"strongest anomaly {peak['severity_zscore']:.2f}\u03c3",
        )
        axis.legend(loc="lower left", fontsize=8, framealpha=0.9)

    axis.set_title(
        f"{event}: ERA5 {source.label} at the strongest detected frame\n{timestamp}\n"
        f"{len(frame['boxes'])} detected boxes this frame, "
        f"{len(detections['frames'])} frames, "
        f"observed z range {z_min:.2f}\u2013{z_max:.2f}\u03c3",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def render_severity_panel(event: str, out_path: Path) -> Path:
    """Tracked peak severity per frame, with the undetected frames left blank."""
    detections = load_detections(event)
    frames = detections["frames"]

    indices = [frame["time_index"] for frame in frames]
    severity = [
        max((box["peak_zscore"] for box in frame["boxes"]), default=float("nan"))
        for frame in frames
    ]
    cells = [sum(box["cell_count"] for box in frame["boxes"]) for frame in frames]
    detected = [bool(frame["boxes"]) for frame in frames]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top.plot(indices, severity, color="#1f4e8c", linewidth=1.6, zorder=3)
    top.scatter(
        [i for i, flag in zip(indices, detected, strict=True) if flag],
        [s for s, flag in zip(severity, detected, strict=True) if flag],
        s=14,
        color="#1f4e8c",
        zorder=4,
        label="frame with detections",
    )
    top.set_ylabel("peak anomaly (\u03c3)")
    top.set_title(
        f"{event}: tracked anomaly severity per frame\n"
        f"{sum(detected)}/{len(frames)} frames contain an anomaly",
        fontsize=10,
    )
    top.grid(True, linestyle=":", alpha=0.6)
    top.legend(fontsize=8)

    bottom.bar(indices, cells, color="#8a7f6d", width=0.8)
    bottom.set_ylabel("anomalous cells")
    bottom.set_xlabel("frame index (ERA5 hourly step)")
    bottom.grid(True, linestyle=":", alpha=0.6)

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------
# GNN panel: real ERA5 over the 66-node mesh
# ---------------------------------------------------------------------
GNN_DIR = PROJECT_ROOT / "weights" / "gnn" / "outputs"
GNN_PREDICTIONS = GNN_DIR / "real_era5" / "real_era5_predictions.npz"


def _mesh_nodes() -> tuple[np.ndarray, np.ndarray]:
    import csv

    latitudes, longitudes = [], []
    with (GNN_DIR / "node_table.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latitudes.append(float(row["latitude"]))
            longitudes.append(float(row["longitude"]))
    return np.array(latitudes), np.array(longitudes)


def _choose_gnn_hour(actual: np.ndarray, feature_index: int) -> int:
    """The held-out hour with the most spatial structure, so the map is informative."""
    return int(np.argmax(actual[:, :, feature_index].std(axis=1)))


def render_gnn_panel(out_path: Path, feature: str = "temperature") -> Path:
    """Observed vs GNN vs persistence on the real ERA5 66-node mesh."""
    if not GNN_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{GNN_PREDICTIONS} not found. Train the real-ERA5 model first:\n"
            "  python weights/gnn/09_train_real_era5.py"
        )

    with np.load(GNN_PREDICTIONS, allow_pickle=False) as payload:
        gnn = payload["gnn"]
        actual = payload["actual"]
        persistence = payload["persistence"]
        timestamps = payload["timestamps"]
        features = [str(name) for name in payload["features"]]
        units = [str(name) for name in payload["units"]]

    if feature not in features:
        raise ValueError(f"{feature!r} is not a GNN feature (have {features})")
    feature_index = features.index(feature)
    unit = units[feature_index]

    hour = _choose_gnn_hour(actual, feature_index)
    observed = actual[hour, :, feature_index]
    predicted = gnn[hour, :, feature_index]
    persisted = persistence[hour, :, feature_index]

    latitudes, longitudes = _mesh_nodes()
    lat_axis = np.unique(latitudes)
    lon_axis = np.unique(longitudes)
    shape = (lat_axis.size, lon_axis.size)
    if shape[0] * shape[1] != latitudes.size:
        raise ValueError(f"The 66-node mesh is not a full {shape} grid")

    edges = np.load(GNN_DIR / "edge_index.npy", allow_pickle=False)
    low = float(min(observed.min(), predicted.min(), persisted.min()))
    high = float(max(observed.max(), predicted.max(), persisted.max()))

    panels = (
        (f"ERA5 observation\n{timestamps[hour]}", observed),
        ("ST-GNN prediction (trained on real ERA5)", predicted),
        ("Persistence baseline", persisted),
    )

    figure = plt.figure(figsize=(15.5, 5.6))
    for index, (title, values) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection=ccrs.PlateCarree())
        _base_map(axis, lat_axis, lon_axis)
        mesh = axis.pcolormesh(
            lon_axis,
            lat_axis,
            values.reshape(shape),
            cmap="inferno",
            shading="nearest",
            vmin=low,
            vmax=high,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        for start, end in zip(edges[0], edges[1], strict=True):
            axis.plot(
                [longitudes[start], longitudes[end]],
                [latitudes[start], latitudes[end]],
                color="#ffffff",
                linewidth=0.5,
                alpha=0.55,
                zorder=5,
                transform=ccrs.PlateCarree(),
            )
        axis.scatter(
            longitudes,
            latitudes,
            s=6,
            color="#111111",
            zorder=6,
            transform=ccrs.PlateCarree(),
        )
        rmse = float(np.sqrt(np.mean((values - observed) ** 2)))
        axis.set_title(f"{title}\nRMSE vs observation {rmse:.3f} {unit}", fontsize=9)
        figure.colorbar(
            mesh, ax=axis, orientation="horizontal", pad=0.07, shrink=0.9,
            label=f"{feature} ({unit})",
        )

    figure.suptitle(
        "ST-GNN trained on real ERA5, scored on the held-out pilot week "
        "(one hour ahead, 66-node mesh)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _gnn_summary_rows() -> list[tuple[str, str]]:
    """Measured skill of the real-ERA5 model, read from its own metrics table."""
    import csv

    metrics_path = GNN_DIR / "real_era5" / "real_era5_metrics.csv"
    if not metrics_path.is_file():
        return [("Real-ERA5 ST-GNN", "not trained — run weights/gnn/09_train_real_era5.py")]
    rows = []
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    f"ST-GNN vs persistence: {row['feature']}",
                    f"{row['gnn_rmse']} vs {row['persistence_rmse']} {row['unit']} "
                    f"({float(row['gnn_vs_persistence_improvement_pct']):+.2f}%)",
                )
            )
    return rows


# ---------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------
def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _summary_rows(event: str) -> list[tuple[str, str]]:
    detections = load_detections(event)
    frames = detections["frames"]
    trajectory = detections.get("trajectory") or []
    boxes = [box for frame in frames for box in frame["boxes"]]
    peak = max(boxes, key=lambda box: box["peak_zscore"]) if boxes else None
    return [
        ("Event", event),
        ("Detector variable", EVENT_SOURCES[event].variable),
        ("Frames scored", str(len(frames))),
        ("Frames with detections", str(sum(1 for f in frames if f["boxes"]))),
        ("Detected boxes", str(len(boxes))),
        ("Anomalous cells", str(sum(box["cell_count"] for box in boxes))),
        ("Tracked trajectory points", str(len(trajectory))),
        (
            "Strongest anomaly",
            f"{peak['peak_zscore']:.2f}\u03c3 at "
            f"({peak['centroid_lat']:.2f}, {peak['centroid_lon']:.2f})"
            if peak
            else "\u2014",
        ),
        ("Severity tiers", "not assigned \u2014 configs/tracking.yaml severity_bands is null"),
    ]


def build_index(
    events: list[str],
    panels: dict[str, dict[str, Path]],
    gnn_panel: Path | None = None,
) -> Path:
    blocks = []
    for event in events:
        folium_map = panels[event].get("folium")
        iframe = (
            f'<iframe src="{folium_map.name}" width="100%" height="620" '
            'style="border:1px solid #ccc;border-radius:6px;"></iframe>'
            if folium_map is not None
            else "<p><em>Interactive map not built. Re-run with --with-folium.</em></p>"
        )
        blocks.append(
            f"""
            <section>
              <h2>{event}</h2>
              <table class="summary">
                {chr(10).join(
                    "<tr><th>{}</th><td>{}</td></tr>".format(*row)
                    for row in _summary_rows(event)
                )}
              </table>
              <img class="panel" src="{_data_uri(panels[event]['field'])}"
                   alt="{event} ERA5 field with detections">
              <img class="panel" src="{_data_uri(panels[event]['severity'])}"
                   alt="{event} severity series">
              <h3>Interactive map</h3>
              {iframe}
            </section>
            """
        )

    if gnn_panel is not None:
        blocks.append(
            "<section><h2>GNN stage (real ERA5)</h2>"
            '<table class="summary">'
            + "".join(
                "<tr><th>{}</th><td>{}</td></tr>".format(*row)
                for row in _gnn_summary_rows()
            )
            + "</table>"
            f'<img class="panel" src="{_data_uri(gnn_panel)}" alt="ST-GNN panel">'
            "<p class='note'>The model is trained on real ERA5 reanalysis for "
            "2020-04-01 through 2020-05-15 and scored on the held-out pilot week "
            "2020-05-16 through 2020-05-22, using the identical window indices as "
            "<code>weights/gnn/08_evaluate_era5_pilot.py</code> so the two results "
            "are directly comparable.</p></section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Threat-X demo dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
          max-width: 1180px; padding: 24px; color: #1b1b1b; background: #fafaf7; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #5a5a5a; margin-top: 0; }}
  section {{ background: #fff; border: 1px solid #e2e2dc; border-radius: 8px;
             padding: 18px 22px 26px; margin: 22px 0; }}
  .panel {{ width: 100%; height: auto; border-radius: 6px; margin-top: 12px; }}
  table.summary {{ border-collapse: collapse; margin: 8px 0 4px; font-size: 14px; }}
  table.summary th {{ text-align: left; padding: 3px 14px 3px 0; color: #5a5a5a;
                      font-weight: 600; white-space: nowrap; vertical-align: top; }}
  table.summary td {{ padding: 3px 0; }}
  .note {{ font-size: 13px; color: #5a5a5a; }}
  code {{ background: #f1f1ec; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
  <h1>Threat-X &mdash; extreme-weather detection and tracking demo</h1>
  <p class="sub">Team Bots &middot; every field, box and trajectory below is a real
  pipeline output, not a mock-up.</p>

  <section>
    <h2>What you are looking at</h2>
    <p class="note">
      The background field is the ERA5 reanalysis archive
      (<code>data/raw/era5_&lt;event&gt;.nc</code>). Each rectangle is a connected
      region of grid cells whose anomaly against the 1991&ndash;2020 May
      climatology exceeds the detector threshold; the red rectangle is the frame
      with the largest excursion. The black line is the tracker's reconstructed
      trajectory across frames. Severity is reported as a z-score magnitude and
      <strong>no severity tier is assigned</strong>, because
      <code>configs/tracking.yaml &rarr; severity_bands</code> is
      <code>null</code>: no severity distribution has been measured yet.
    </p>
  </section>
{chr(10).join(blocks)}
</body>
</html>
"""
    out_path = OUT_DIR / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def build_event(event: str, with_folium: bool) -> dict[str, Path]:
    """Render every panel for one event and return the artefact paths."""
    if event not in EVENT_SOURCES:
        raise ValueError(f"Unknown event {event!r}. Choices: {list(EVENT_SOURCES)}")
    panels = {
        "field": render_field_panel(event, OUT_DIR / f"field_{event}.png"),
        "severity": render_severity_panel(event, OUT_DIR / f"severity_{event}.png"),
    }
    if with_folium:
        from frontend.visulisation import build_dashboard

        panels["folium"] = build_dashboard(event)
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=sorted(EVENT_SOURCES), help="Build one event.")
    parser.add_argument("--all-events", action="store_true", help="Build every event.")
    parser.add_argument(
        "--with-folium",
        action="store_true",
        help="Also build the interactive folium map (needs the detector output).",
    )
    parser.add_argument(
        "--with-gnn",
        action="store_true",
        help=(
            "Also build the real-ERA5 ST-GNN panel (needs "
            "weights/gnn/09_train_real_era5.py to have run)."
        ),
    )
    parser.add_argument(
        "--gnn-feature",
        default="temperature",
        help="Feature to draw in the GNN panel.",
    )
    args = parser.parse_args()

    events = sorted(EVENT_SOURCES) if args.all_events else [args.event] if args.event else []
    if not events:
        parser.error("pass --event <name> or --all-events")

    panels: dict[str, dict[str, Path]] = {}
    for event in events:
        print(f"[dashboard] {event}")
        panels[event] = build_event(event, args.with_folium)
        for name, path in panels[event].items():
            print(f"  {name:9s} -> {path.relative_to(PROJECT_ROOT)}")

    gnn_panel = None
    if args.with_gnn:
        print("[dashboard] GNN (real ERA5)")
        gnn_panel = render_gnn_panel(OUT_DIR / "gnn_real_era5.png", args.gnn_feature)
        print(f"  {'gnn':9s} -> {gnn_panel.relative_to(PROJECT_ROOT)}")

    index = build_index(events, panels, gnn_panel)
    print(f"[dashboard] wrote {index.relative_to(PROJECT_ROOT)} -- open it to record the demo")


if __name__ == "__main__":
    main()
