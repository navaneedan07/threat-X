"""
Interactive Folium map of a detected event: the anomaly boxes, the tracked
trajectory, and the strongest anomaly.

USAGE::

    python -m src.detection.anomaly_detection --event amphan --variable mean_sea_level_pressure
    python frontend/visulisation.py --event amphan
    # writes data/processed/plots/dashboard/dashboard_amphan.html

Two rules this file follows, both taken from the project's own documentation:

* **No invented thresholds.** ``configs/tracking.yaml -> severity_bands`` is
  ``null`` because no severity distribution has been measured, so this map does
  not print "severe" / "moderate" / "low". It colours by peak anomaly in sigma
  over the range actually observed for that event and reports the number itself.
* **No invented distances.** An earlier version labelled every peak a "5 km
  impact radius". The input grid is 0.25 deg (~28 km), so that claim was wrong
  by roughly 5x; the popup reports the measured z-score and centroid instead.

It also used to read ``output/anomalies_<event>.json``, a path nothing writes to
since the detector moved to ``data/processed/detection/`` -- so the map could not
be generated at all. That is fixed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
from folium.plugins import AntPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "data" / "processed" / "detection"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "plots" / "dashboard"

# Ramp endpoints. These are colours, not thresholds: a marker is shaded by where
# its z-score falls between the smallest and largest anomaly actually observed.
LOW_COLOR = (0xF1, 0xC4, 0x0F)  # yellow
HIGH_COLOR = (0xE7, 0x4C, 0x3C)  # red


def rgb_to_hex(channel: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*channel)


def ramp_colour(zscore: float, z_min: float, z_max: float) -> str:
    """Hex colour for ``zscore`` mapped linearly onto the observed [z_min, z_max]."""
    if z_max <= z_min:
        fraction = 1.0
    else:
        fraction = min(max((zscore - z_min) / (z_max - z_min), 0.0), 1.0)
    return rgb_to_hex(
        tuple(
            int(round(lo + fraction * (hi - lo)))
            for lo, hi in zip(LOW_COLOR, HIGH_COLOR, strict=True)
        )
    )


def build_dashboard(event: str) -> Path:
    anomaly_path = DETECTION_DIR / f"anomalies_{event}.json"
    if not anomaly_path.exists():
        raise FileNotFoundError(
            f"{anomaly_path} not found. Run the detector first:\n"
            f"  python -m src.detection.anomaly_detection --event {event}"
        )

    with open(anomaly_path, encoding="utf-8") as handle:
        data = json.load(handle)

    trajectory = data["trajectory"]
    if not trajectory:
        raise ValueError(
            f"no tracked anomaly in {anomaly_path}: this event produced no "
            "detections, so there is nothing to map"
        )

    z_values = [point["severity_zscore"] for point in trajectory]
    z_min, z_max = min(z_values), max(z_values)

    centre_lat = sum(point["lat"] for point in trajectory) / len(trajectory)
    centre_lon = sum(point["lon"] for point in trajectory) / len(trajectory)

    # OpenStreetMap rather than CartoDB positron: CartoDB now requires an API key,
    # so its tiles silently failed to load and the map rendered on a blank canvas.
    the_map = folium.Map(location=[centre_lat, centre_lon], zoom_start=6, tiles="OpenStreetMap")

    AntPath(
        [[point["lat"], point["lon"]] for point in trajectory],
        color="#2c3e50",
        weight=3,
        delay=800,
    ).add_to(the_map)

    for point in trajectory:
        zscore = point["severity_zscore"]
        folium.CircleMarker(
            location=[point["lat"], point["lon"]],
            radius=6 + min(zscore, 12.0),
            color=ramp_colour(zscore, z_min, z_max),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{point['timestamp']}</b><br>"
                f"peak anomaly: {zscore:.2f} &sigma;<br>"
                "severity tier: not assigned<br>"
                "<small>configs/tracking.yaml &rarr; severity_bands is null</small>",
                max_width=260,
            ),
        ).add_to(the_map)

    box_count = 0
    for frame in data["frames"]:
        for box in frame["boxes"]:
            box_count += 1
            folium.Rectangle(
                bounds=[
                    [box["lat_min"], box["lon_min"]],
                    [box["lat_max"], box["lon_max"]],
                ],
                color="#7f8c8d",
                weight=1,
                fill=False,
                opacity=0.4,
            ).add_to(the_map)

    peak = max(trajectory, key=lambda point: point["severity_zscore"])
    folium.Marker(
        location=[peak["lat"], peak["lon"]],
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
        popup=(
            f"<b>STRONGEST ANOMALY</b><br>{peak['timestamp']}<br>"
            f"z = {peak['severity_zscore']:.2f} &sigma;<br>"
            f"centroid {peak['lat']:.2f}, {peak['lon']:.2f}"
        ),
    ).add_to(the_map)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-family: sans-serif; font-size: 13px;">
      <b>Peak anomaly (&sigma;)</b><br>
      <span style="color:{rgb_to_hex(LOW_COLOR)};">&#9679;</span> {z_min:.2f}<br>
      <span style="color:{rgb_to_hex(HIGH_COLOR)};">&#9679;</span> {z_max:.2f}<br>
      <small>{len(trajectory)} trajectory points, {box_count} detected boxes.<br>
      Severity tiers not assigned &mdash; <code>severity_bands</code> is <code>null</code>.</small>
    </div>
    """
    the_map.get_root().html.add_child(folium.Element(legend_html))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dashboard_{event}.html"
    the_map.save(str(out_path))
    print(f"[done] wrote {out_path} -- open it in a browser")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map the detector's output for one event.")
    parser.add_argument("--event", required=True, choices=["amphan", "heatwave"])
    args = parser.parse_args()
    build_dashboard(args.event)
