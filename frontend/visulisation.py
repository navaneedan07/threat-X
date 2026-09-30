"""
Visualization dashboard -- builds an interactive Folium map.

USAGE:
  python 05.py --event amphan
  -> writes output/dashboard_amphan.html, open it in a browser
"""

import argparse
import json
import os

import folium
from folium.plugins import AntPath

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

SEVERITY_COLOR = {
    "low": "#f1c40f",
    "moderate": "#e67e22",
    "severe": "#e74c3c",
}


def severity_from_zscore(z: float) -> str:
    if z >= 4.0:
        return "severe"
    elif z >= 3.0:
        return "moderate"
    else:
        return "low"


def build_dashboard(event: str):
    candidate_paths = [
        os.path.join(OUT_DIR, f"anomalies_{event}.json"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "detection", f"anomalies_{event}.json"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", f"anomalies_{event}.json"),
        os.path.join(os.path.dirname(__file__), "..", "data", "samples", f"anomalies_{event}.json"),
    ]
    
    anomaly_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            anomaly_path = p
            break
            
    if not anomaly_path:
        raise FileNotFoundError(
            f"Could not find anomalies_{event}.json. "
            f"Please run: python -m src.detection.anomaly_detection --event {event}"
        )

    with open(anomaly_path) as f:
        data = json.load(f)

    trajectory = data["trajectory"]
    if not trajectory:
        raise ValueError("No tracked anomaly in this event's data -- lower Z_THRESHOLD or check the input")

    center_lat = sum(p["lat"] for p in trajectory) / len(trajectory)
    center_lon = sum(p["lon"] for p in trajectory) / len(trajectory)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS",
    )

    path_coords = [[p["lat"], p["lon"]] for p in trajectory]
    AntPath(path_coords, color="#2c3e50", weight=3, delay=800).add_to(m)

    for point in trajectory:
        severity = severity_from_zscore(point["severity_zscore"])
        folium.CircleMarker(
            location=[point["lat"], point["lon"]],
            radius=6 + point["severity_zscore"],
            color=SEVERITY_COLOR[severity],
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{point['timestamp']}</b><br>"
                f"Severity: {severity}<br>"
                f"Z-score: {point['severity_zscore']:.2f}",
                max_width=250,
            ),
        ).add_to(m)

    for frame in data["frames"]:
        for box in frame["boxes"]:
            folium.Rectangle(
                bounds=[[box["lat_min"], box["lon_min"]], [box["lat_max"], box["lon_max"]]],
                color="#7f8c8d",
                weight=1,
                fill=False,
                opacity=0.4,
            ).add_to(m)

    peak = max(trajectory, key=lambda p: p["severity_zscore"])
    severity = severity_from_zscore(peak["severity_zscore"])
    folium.Marker(
        location=[peak["lat"], peak["lon"]],
        icon=folium.Icon(color="red" if severity == "severe" else "orange", icon="exclamation-triangle", prefix="fa"),
        popup=f"<b>PEAK ALERT</b><br>{peak['timestamp']}<br>{severity.upper()} (z={peak['severity_zscore']:.2f})<br>5km impact radius",
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-family: sans-serif; font-size: 13px;">
      <b>Alert Severity</b><br>
      <span style="color:#e74c3c;">&#9679;</span> Severe (z &ge; 4.0)<br>
      <span style="color:#e67e22;">&#9679;</span> Moderate (z &ge; 3.0)<br>
      <span style="color:#f1c40f;">&#9679;</span> Low (z &ge; 2.5)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dashboard_{event}.html")
    m.save(out_path)
    print(f"[done] wrote {out_path} -- open it in a browser")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, choices=["amphan", "heatwave"])
    args = ap.parse_args()
    build_dashboard(args.event)