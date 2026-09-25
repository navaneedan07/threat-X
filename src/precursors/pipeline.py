"""Precursor pipeline runner.

Entry point for running the atmospheric precursor analysis on the available
ERA5 datasets and writing the results to data/processed/ so the REST API
can serve real output instead of fixtures.

Usage:
    python -m src.precursors.pipeline

    or from Python:
        from src.precursors.pipeline import run_precursor_pipeline
        results = run_precursor_pipeline("data/raw/era5_amphan.nc", ...)

Outputs:
    data/processed/precursors.json       — API-ready precursor series
    data/processed/plots/<event>/...png  — diagnostic time-series plots
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.loader import load_dataset, get_time_step_hours, spatial_resolution_deg
from src.precursors.engine import (
    PrecursorResult,
    compute_precursor_series,
    generate_explanatory_summary,
)
from src.precursors.plots import plot_precursor_series

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event definitions — what we know about each available dataset
# These are the ground-truth case descriptions for the ERA5 sample data.
# ---------------------------------------------------------------------------
KNOWN_EVENTS = {
    "era5_amphan": {
        "dataset_file": "data/raw/era5_amphan.nc",
        "threat_id": "THR-2020-0001",
        "event_type": "cyclone",
        "event_name": "Cyclone Amphan (2020-05-16 to 2020-05-21)",
        # Approximate centroid from known Amphan track (Bay of Bengal):
        "centroid_lat": 14.0,
        "centroid_lon": 87.0,
        "description": "Super Cyclonic Storm Amphan, Bay of Bengal",
    },
    "era5_heatwave": {
        "dataset_file": "data/raw/era5_heatwave.nc",
        "threat_id": "THR-2022-0001",
        "event_type": "heatwave",
        "event_name": "North India Heatwave (May 2022)",
        # Approximate centroid: Punjab/Rajasthan area
        "centroid_lat": 28.5,
        "centroid_lon": 77.0,
        "description": "Severe heatwave, North India and Pakistan, May 2022",
    },
}


def run_precursor_pipeline(
    dataset_key: str,
    window_hours: int = 48,
    output_dir: str | Path = "data/processed",
    plot_output_dir: str | Path = "data/processed/plots",
    neighbourhood_deg: float = 1.5,
) -> dict:
    """Run precursor analysis for one known event dataset.

    Parameters
    ----------
    dataset_key : str
        One of the keys in KNOWN_EVENTS ('era5_amphan' or 'era5_heatwave').
    window_hours : int
        Temporal window to include in the precursor series.
    output_dir : path-like
        Directory where precursors.json is written/updated.
    plot_output_dir : path-like
        Directory where PNG plots are saved.
    neighbourhood_deg : float
        Spatial averaging radius around the centroid.

    Returns
    -------
    dict
        The processed entry for this event (suitable for the API response).
    """
    if dataset_key not in KNOWN_EVENTS:
        raise ValueError(f"Unknown dataset key: {dataset_key!r}. Use one of {list(KNOWN_EVENTS)}")

    event = KNOWN_EVENTS[dataset_key]
    dataset_path = Path(event["dataset_file"])
    threat_id = event["threat_id"]
    centroid_lat = event["centroid_lat"]
    centroid_lon = event["centroid_lon"]

    logger.info("Loading dataset: %s", dataset_path)
    ds = load_dataset(dataset_path)

    dt_h = get_time_step_hours(ds)
    res_deg = spatial_resolution_deg(ds)
    logger.info("Dataset: timestep=%.0fh, resolution=%.2f deg", dt_h, res_deg)

    # Compute precursor series
    logger.info(
        "Computing precursors for %s at (%.2f, %.2f), window=%dh",
        threat_id, centroid_lat, centroid_lon, window_hours,
    )
    series = compute_precursor_series(
        ds,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        window_hours=window_hours,
        neighbourhood_deg=neighbourhood_deg,
    )

    if not series:
        logger.error("No precursor data computed — check dataset coverage.")
        return {}

    logger.info("Computed %d precursor timesteps", len(series))

    # Summarise
    summary = generate_explanatory_summary(series)
    logger.info("Summary: %s", summary)

    # Determine which variables were actually computed
    sample = series[0]
    computed_vars = []
    if sample.pressure_tendency_3h_hpa is not None or any(
        r.pressure_tendency_3h_hpa is not None for r in series
    ):
        computed_vars.append("mslp_tendency")
    if any(r.wind_divergence_s1 is not None for r in series):
        computed_vars.append("wind_divergence_10m")
    if any(r.wind_speed_ms is not None for r in series):
        computed_vars.append("wind_speed_10m")
    if any(r.precipitation_rate_mm3h is not None for r in series):
        computed_vars.append("precipitation_rate")
    if any(r.t2m_anomaly_k is not None for r in series):
        computed_vars.append("t2m_anomaly")

    # Build API-compatible output entry
    api_entry = {
        "threat_id": threat_id,
        "event_type": event["event_type"],
        "dataset_key": dataset_key,
        "dataset_file": str(dataset_path),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "grid_resolution_deg": res_deg,
        "time_step_hours": dt_h,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "neighbourhood_deg": neighbourhood_deg,
        "variables_analyzed": computed_vars,
        "unavailable_variables": [
            v for v in series[0].unavailable_variables
        ],
        "series": [r.to_api_dict() for r in series],
        "explanatory_summary": summary,
    }

    # Write to processed output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precursors_json_path = output_dir / "precursors.json"

    # Load existing file if it exists, to merge entries
    existing: dict = {}
    if precursors_json_path.exists():
        with precursors_json_path.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    precursors_data = existing if isinstance(existing, dict) else {}
    precursors_data.setdefault("_meta", {})
    precursors_data["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    precursors_data["_meta"]["source"] = "src.precursors.pipeline"
    precursors_data["_meta"]["note"] = (
        "Real computed values from ERA5 surface-level data. "
        "Pressure-level variables (850 hPa vorticity, moisture flux convergence, "
        "bulk shear, theta-e gradient) are not available in this dataset and "
        "are represented as null — they have not been fabricated."
    )
    precursors_data.setdefault("precursors", {})
    precursors_data["precursors"][threat_id] = api_entry

    with precursors_json_path.open("w", encoding="utf-8") as f:
        json.dump(precursors_data, f, indent=2, default=str)
    logger.info("Precursor results written to: %s", precursors_json_path)

    # Generate and save plots
    plot_dir = Path(plot_output_dir) / dataset_key
    plot_path = plot_dir / "precursor_series.png"
    try:
        plot_precursor_series(
            series,
            event_name=event["event_name"],
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            save_to=plot_path,
        )
    except Exception as exc:
        logger.error("Plot generation failed: %s", exc)

    ds.close()
    return api_entry


def run_all_events(
    output_dir: str | Path = "data/processed",
    plot_output_dir: str | Path = "data/processed/plots",
    window_hours: int = 48,
) -> dict:
    """Run precursor analysis for all known events.

    Returns a dict keyed by threat_id.
    """
    results = {}
    for key in KNOWN_EVENTS:
        dataset_path = Path(KNOWN_EVENTS[key]["dataset_file"])
        if not dataset_path.exists():
            logger.warning("Dataset not found, skipping: %s", dataset_path)
            continue
        try:
            entry = run_precursor_pipeline(
                key,
                window_hours=window_hours,
                output_dir=output_dir,
                plot_output_dir=plot_output_dir,
            )
            if entry:
                results[entry["threat_id"]] = entry
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", key, exc)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_all_events()
    print(f"\nDone. Processed {len(results)} event(s).")
    for tid, entry in results.items():
        n = len(entry.get("series", []))
        summary = entry.get("explanatory_summary", "")
        print(f"  {tid}: {n} timesteps | {summary[:80]}...")
