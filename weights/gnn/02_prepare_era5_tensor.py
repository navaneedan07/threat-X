"""Prepare a timestamped ERA5 surface-field tensor for the existing ST-GNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_dataset  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "data" / "raw" / "era5_gnn_pilot_20200516_20200522.zip"
DEFAULT_NODE_TABLE = BASE_DIR / "outputs" / "node_table.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "era5_pilot"

SOURCE_VARIABLES = ("t2m", "d2m", "msl", "u10", "v10", "tp")
FEATURE_NAMES = ("temperature", "pressure", "humidity", "wind_speed", "precipitation")
FEATURE_UNITS = ("degC", "hPa", "%", "m/s", "mm/hour")
EXPECTED_LATITUDES = tuple(10.0 + 0.5 * index for index in range(11))
EXPECTED_LONGITUDES = tuple(76.0 + index for index in range(6))
EXPECTED_NODE_COUNT = 66
EXPECTED_PILOT_TIMESTEPS = 168
TIME_NAME = "valid_time"
LATITUDE_NAME = "latitude"
LONGITUDE_NAME = "longitude"
TIME_NS_PER_HOUR = 3_600_000_000_000

EXPECTED_SOURCE_UNITS = {
    "t2m": {"k"},
    "d2m": {"k"},
    "msl": {"pa"},
    "u10": {"m s**-1", "m s-1", "m/s"},
    "v10": {"m s**-1", "m s-1", "m/s"},
    "tp": {"m"},
}


def _coordinate_lookup(values: np.ndarray, name: str) -> dict[float, int]:
    lookup: dict[float, int] = {}
    for index, value in enumerate(values):
        key = round(float(value), 8)
        if key in lookup:
            raise ValueError(f"Source {name} coordinate is duplicated: {value}")
        lookup[key] = index
    return lookup


def _validate_node_table(node_table: pd.DataFrame) -> list[tuple[float, float]]:
    required_columns = {"node_id", "latitude", "longitude"}
    if not required_columns.issubset(node_table.columns):
        raise ValueError(f"Node table must contain columns {sorted(required_columns)}")
    if len(node_table) != EXPECTED_NODE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_NODE_COUNT} nodes, got {len(node_table)}")

    node_ids = pd.to_numeric(node_table["node_id"], errors="raise").to_numpy()
    if not np.array_equal(node_ids, np.arange(EXPECTED_NODE_COUNT)):
        raise ValueError("Node IDs must be sequential and ordered 0..65")

    coordinates = list(
        zip(
            pd.to_numeric(node_table["latitude"], errors="raise").astype(float),
            pd.to_numeric(node_table["longitude"], errors="raise").astype(float),
            strict=True,
        )
    )
    if len(set(coordinates)) != EXPECTED_NODE_COUNT:
        raise ValueError("Node table contains duplicate coordinates")

    expected = [
        (latitude, longitude)
        for latitude in EXPECTED_LATITUDES
        for longitude in EXPECTED_LONGITUDES
    ]
    if coordinates != expected:
        raise ValueError("Node table coordinates/order do not match the expected 66-node grid")
    return coordinates


def _require_valid_time_axis(dataset: xr.Dataset) -> pd.DatetimeIndex:
    if TIME_NAME not in dataset.coords:
        raise ValueError(f"Dataset has no common {TIME_NAME!r} coordinate")
    timestamps = pd.DatetimeIndex(pd.to_datetime(dataset[TIME_NAME].values))
    if timestamps.empty or timestamps.hasnans:
        raise ValueError("valid_time must contain at least one non-missing timestamp")
    if not timestamps.is_unique:
        raise ValueError("valid_time contains duplicate timestamps")
    timestamp_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(timestamps) > 1 and not np.all(np.diff(timestamp_ns) == TIME_NS_PER_HOUR):
        raise ValueError("valid_time must be strictly increasing with exactly hourly spacing")

    for name in SOURCE_VARIABLES:
        if name not in dataset.data_vars:
            raise ValueError(f"Required ERA5 variable is missing: {name}")
        variable = dataset[name]
        if TIME_NAME not in variable.dims:
            raise ValueError(f"{name} has no {TIME_NAME!r} dimension")
        variable_times = pd.DatetimeIndex(pd.to_datetime(variable[TIME_NAME].values))
        if not variable_times.equals(timestamps):
            raise ValueError(f"{name} does not share the common valid_time axis")
        units = str(variable.attrs.get("units", "")).strip().lower()
        if units not in EXPECTED_SOURCE_UNITS[name]:
            raise ValueError(
                f"Unexpected or missing units for {name}: {variable.attrs.get('units')!r}"
            )
        dimensions = set(variable.dims)
        expected_dimensions = {TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME}
        if dimensions != expected_dimensions or len(variable.dims) != 3:
            raise ValueError(f"{name} must have only {sorted(expected_dimensions)} dimensions")

        values = np.asarray(variable.values)
        if np.isnan(values).any():
            raise ValueError(f"{name} contains NaN values")
        if np.isinf(values).any():
            raise ValueError(f"{name} contains infinite values")

    return timestamps


def _validate_grid(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    for name in (LATITUDE_NAME, LONGITUDE_NAME):
        if name not in dataset.coords:
            raise ValueError(f"Dataset has no {name!r} coordinate")
        if dataset[name].ndim != 1:
            raise ValueError(f"{name} coordinate must be one-dimensional")

    latitudes = np.asarray(dataset[LATITUDE_NAME].values, dtype=float)
    longitudes = np.asarray(dataset[LONGITUDE_NAME].values, dtype=float)
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Source grid coordinates must be finite")
    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError("Source grid must have at least two latitude and longitude points")
    if not np.allclose(np.abs(np.diff(latitudes)), 0.25, atol=1e-8, rtol=0):
        raise ValueError("Source latitude grid must have regular 0.25-degree spacing")
    if not np.allclose(np.abs(np.diff(longitudes)), 0.25, atol=1e-8, rtol=0):
        raise ValueError("Source longitude grid must have regular 0.25-degree spacing")
    if not (np.all(np.diff(latitudes) > 0) or np.all(np.diff(latitudes) < 0)):
        raise ValueError("Source latitude coordinates must be monotonic")
    if not (np.all(np.diff(longitudes) > 0) or np.all(np.diff(longitudes) < 0)):
        raise ValueError("Source longitude coordinates must be monotonic")
    return latitudes, longitudes


def saturation_vapour_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    """Return saturation vapour pressure using the pilot-validated formula."""
    return 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def prepare_dataset(
    dataset: xr.Dataset,
    node_table: pd.DataFrame,
    source_archive: str,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    """Validate ERA5 fields and return the node-ordered float32 tensor in memory."""
    timestamps = _require_valid_time_axis(dataset)
    node_coordinates = _validate_node_table(node_table)
    source_latitudes, source_longitudes = _validate_grid(dataset)
    latitude_lookup = _coordinate_lookup(source_latitudes, LATITUDE_NAME)
    longitude_lookup = _coordinate_lookup(source_longitudes, LONGITUDE_NAME)

    source_indices: list[tuple[int, int]] = []
    for latitude, longitude in node_coordinates:
        lat_key, lon_key = round(latitude, 8), round(longitude, 8)
        if lat_key not in latitude_lookup or lon_key not in longitude_lookup:
            raise ValueError(f"Project coordinate {(latitude, longitude)} is absent from ERA5 grid")
        source_indices.append((latitude_lookup[lat_key], longitude_lookup[lon_key]))

    def values_at_nodes(name: str) -> np.ndarray:
        field = dataset[name].transpose(TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME).values
        values = np.asarray(field, dtype=np.float64)
        return np.stack(
            [
                values[:, latitude_index, longitude_index]
                for latitude_index, longitude_index in source_indices
            ],
            axis=1,
        )

    t2m = values_at_nodes("t2m")
    d2m = values_at_nodes("d2m")
    msl = values_at_nodes("msl")
    u10 = values_at_nodes("u10")
    v10 = values_at_nodes("v10")
    tp = values_at_nodes("tp")

    temperature_c = t2m - 273.15
    dewpoint_c = d2m - 273.15
    humidity = 100.0 * saturation_vapour_pressure_hpa(dewpoint_c) / saturation_vapour_pressure_hpa(
        temperature_c
    )
    pressure_hpa = msl / 100.0
    wind_speed = np.sqrt(u10**2 + v10**2)
    precipitation_mm_hour = tp * 1000.0

    tp_step_type = str(
        dataset["tp"].attrs.get("GRIB_stepType", dataset["tp"].attrs.get("stepType", ""))
    ).lower()
    if tp_step_type != "accum":
        raise ValueError(f"tp must be an ERA5 accumulation; found step type {tp_step_type!r}")
    if np.any(precipitation_mm_hour < 0):
        raise ValueError("ERA5 total precipitation contains negative accumulations")

    features = (temperature_c, pressure_hpa, humidity, wind_speed, precipitation_mm_hour)
    tensor = np.stack(features, axis=-1).astype(np.float32, copy=False)
    expected_shape = (len(timestamps), EXPECTED_NODE_COUNT, len(FEATURE_NAMES))
    if tensor.shape != expected_shape:
        raise ValueError(f"Unexpected tensor shape {tensor.shape}; expected {expected_shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("Prepared tensor contains NaN or infinite values")

    if len(timestamps) == EXPECTED_PILOT_TIMESTEPS:
        if tensor.shape != (168, 66, 5):
            raise ValueError(f"Pilot tensor has unexpected shape {tensor.shape}")

    metadata = {
        "source": {
            "name": "ERA5 reanalysis",
            "dataset": "reanalysis-era5-single-levels",
            "archive": source_archive,
        },
        "time": {
            "coordinate": TIME_NAME,
            "timezone": "UTC",
            "first_valid_time": timestamps[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_valid_time": timestamps[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frequency": "hourly",
            "forecast_reference_time": None,
            "lead_time_hours": None,
            "forecast_metadata_note": (
                "ERA5 reanalysis valid times; no forecast run or lead time applies."
            ),
        },
        "source_variables": {
            name: {"name": name, "units": str(dataset[name].attrs["units"])}
            for name in SOURCE_VARIABLES
        },
        "features": [
            {"index": index, "name": name, "units": unit}
            for index, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True))
        ],
        "preprocessing": {
            "temperature": "t2m - 273.15; K to degC",
            "pressure": "msl / 100; Pa to hPa",
            "humidity": (
                "100 * es(d2m_C) / es(t2m_C); RH at 2 m in percent; "
                "es(T) = 6.112 * exp((17.67*T)/(T+243.5)); no clipping"
            ),
            "wind_speed": "sqrt(u10**2 + v10**2); m/s",
            "precipitation": "tp * 1000; metres to mm per hourly accumulation ending at valid_time",
            "spatial_selection": "exact source-coordinate lookup; interpolation=false",
        },
        "tensor": {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "dimension_order": ["valid_time", "node_id", "feature"],
            "feature_order": list(FEATURE_NAMES),
        },
        "nodes": [
            {"node_id": index, "latitude": latitude, "longitude": longitude}
            for index, (latitude, longitude) in enumerate(node_coordinates)
        ],
        "validation": {
            "timestamp_count": len(timestamps),
            "timestamps_unique": bool(timestamps.is_unique),
            "timestamps_hourly": len(timestamps) < 2
            or bool(
                np.all(
                    np.diff(timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64))
                    == TIME_NS_PER_HOUR
                )
            ),
            "node_count": len(node_coordinates),
            "node_order_matches_node_table": True,
            "exact_coordinate_selection": True,
            "interpolation": False,
            "finite_tensor": bool(np.isfinite(tensor).all()),
            "nan_count": int(np.isnan(tensor).sum()),
            "infinite_count": int(np.isinf(tensor).sum()),
        },
    }
    return tensor, timestamps, metadata


def _write_outputs(
    tensor: np.ndarray,
    timestamps: pd.DatetimeIndex,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "tensor.npy", tensor)
    pd.DataFrame(
        {TIME_NAME: timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")}
    ).to_csv(output_dir / "timestamps.csv", index=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--node-table", type=Path, default=DEFAULT_NODE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(f"ERA5 archive not found: {args.archive}")
    if not args.node_table.is_file():
        raise FileNotFoundError(f"Project node table not found: {args.node_table}")

    dataset = load_dataset(args.archive)
    node_table = pd.read_csv(args.node_table)
    tensor, timestamps, metadata = prepare_dataset(
        dataset, node_table, str(args.archive.resolve())
    )
    if len(timestamps) != EXPECTED_PILOT_TIMESTEPS:
        raise ValueError(
            f"Pilot preparation expects {EXPECTED_PILOT_TIMESTEPS} hourly timestamps; "
            f"found {len(timestamps)}"
        )
    expected_pilot_times = pd.date_range(
        "2020-05-16 00:00:00", "2020-05-22 23:00:00", freq="h"
    )
    if not timestamps.equals(expected_pilot_times):
        raise ValueError("Pilot valid_time must span 2020-05-16 00:00 through 2020-05-22 23:00 UTC")
    _write_outputs(tensor, timestamps, metadata, args.output_dir)

    source_latitudes = dataset[LATITUDE_NAME].values
    source_longitudes = dataset[LONGITUDE_NAME].values
    print("ERA5 PREPARATION: PASS")
    print(f"Archive: {args.archive.resolve()}")
    print(f"Time: {len(timestamps)} timestamps")
    print(f"  {timestamps[0]:%Y-%m-%d %H:%M} through {timestamps[-1]:%Y-%m-%d %H:%M} UTC")
    print("  hourly; no duplicates; no gaps")
    print(
        f"Grid: {len(source_latitudes)} x {len(source_longitudes)} source at 0.25 degrees; "
        "66 exact target nodes; interpolation = false"
    )
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(
        f"Tensor: shape = {tensor.shape}; dtype = {tensor.dtype}; "
        f"NaN = {np.isnan(tensor).sum()}; Inf = {np.isinf(tensor).sum()}"
    )
    print("Node order: matches node_table.csv = true")
    print("Metadata: timestamps preserved = true; source = ERA5 reanalysis")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()