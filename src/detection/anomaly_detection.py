"""
Anomaly detection module (Stage 1 substitute for the full GNN tracker).

This version automatically handles CDS downloads that are returned as
ZIP archives containing NetCDF files.

It supports:
    data/era5_<event>.nc
    data/era5_<event>.zip
    data/era5_<event>.nc   <-- even if this file is actually a ZIP archive

USAGE:
    python 02.py --event amphan --variable total_precipitation
    python 02.py --event heatwave --variable 2m_temperature
"""

import argparse
import glob
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------
#
# Resolved from the project root, not from this file's parent. The previous
# ``os.path.join(os.path.dirname(__file__), "..", "data")`` pointed at
# ``src/data``, which is not the data directory: it holds no event archive and no
# climatology, so ``get_event_netcdf`` could never find anything and
# ``get_climatology_files()`` silently globbed an empty directory. Boxes were then
# scored against no baseline without the failure being visible in a path.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = str(PROJECT_ROOT / "data")
RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
CLIM_DIR = str(PROJECT_ROOT / "data" / "climatology")
OUT_DIR = str(PROJECT_ROOT / "data" / "processed" / "detection")
EXTRACT_DIR = str(PROJECT_ROOT / "data" / "extracted")

_EVENT_SEARCH_DIRS = (RAW_DIR, DATA_DIR)
"""Where an event archive may live. ``data/raw`` is the current layout; ``data`` is
kept in the search path so an older flat checkout still runs."""

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------

VAR_MAP = {
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}


Z_THRESHOLD = 2.5


def detect_field_anomalies(
    field,
    baseline,
    nodes,
    timestamp,
    variable="total_precipitation",
):
    """Detect positive z-score regions in a model field on the project node grid.

    This uses the same positive ``Z_THRESHOLD`` and two-cell minimum as
    :func:`detect_anomalies`. ``baseline`` is the supplied prototype reference
    field history; it is not a climatology and must not be described as one.
    ``nodes`` must be in node_id order and cover a complete rectangular grid.
    """
    field = np.asarray(field, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    nodes = list(nodes)
    if field.ndim != 1 or baseline.ndim != 2 or baseline.shape[1] != len(field):
        raise ValueError("Expected field (N,) and baseline (T,N) arrays")
    if len(nodes) != len(field) or not np.isfinite(field).all() or not np.isfinite(baseline).all():
        raise ValueError("Field, baseline, and node coordinates must be finite and aligned")

    latitudes = np.array(sorted({float(node["latitude"]) for node in nodes}))
    longitudes = np.array(sorted({float(node["longitude"]) for node in nodes}))
    if len(latitudes) * len(longitudes) != len(nodes):
        raise ValueError("Project nodes must form a complete rectangular grid")

    mean = baseline.mean(axis=0)
    std = np.maximum(baseline.std(axis=0), 1e-6)
    z = (field - mean) / std
    grid = np.full((len(latitudes), len(longitudes)), np.nan)
    for node, score in zip(nodes, z):
        yi = int(np.searchsorted(latitudes, float(node["latitude"])))
        xi = int(np.searchsorted(longitudes, float(node["longitude"])))
        if np.isfinite(grid[yi, xi]):
            raise ValueError("Node coordinates must be unique")
        grid[yi, xi] = score
    flagged = grid > Z_THRESHOLD
    labeled, count = ndimage.label(flagged)
    boxes = []
    for label_id in range(1, count + 1):
        ys, xs = np.where(labeled == label_id)
        if len(ys) < 2:
            continue
        boxes.append({
            "label_id": int(label_id),
            "lat_min": float(latitudes[ys].min()),
            "lat_max": float(latitudes[ys].max()),
            "lon_min": float(longitudes[xs].min()),
            "lon_max": float(longitudes[xs].max()),
            "centroid_lat": float(latitudes[ys].mean()),
            "centroid_lon": float(longitudes[xs].mean()),
            "peak_zscore": float(np.nanmax(np.where(labeled == label_id, grid, np.nan))),
            "cell_count": int(len(ys)),
        })
    boxes.sort(key=lambda box: box["peak_zscore"], reverse=True)
    return [{"time_index": 0, "timestamp": str(timestamp), "variable": variable, "boxes": boxes}]

# ---------------------------------------------------------------------
# Which climatology each event needs
# ---------------------------------------------------------------------

EVENT_REGIONS = {
    "amphan": "bay_of_bengal",
    "heatwave": "north_india",
}
"""Event -> the CDS region its anomaly baseline must be drawn from.

Mirrors ``src/data/cds_fetch.py::EVENTS``; that module is not imported here because
it needs ``cdsapi``, a download-time-only dependency the detection stage should not
require.

The two regions do **not** overlap (Bay of Bengal 5-25 N / 80-95 E vs North India
20-35 N / 68-90 E), so a single shared baseline is wrong for both: scoring a heatwave
against a Bay of Bengal mean measures it against the wrong climate entirely.
"""

LEGACY_CLIMATOLOGY_REGION = "bay_of_bengal"
"""The baseline that predates the region suffix and is stored as ``era5_clim_<year>.nc``.

Kept so an existing checkout keeps working: those files are the only Bay of Bengal
climatology on disk, and they are named without the region.
"""

LEGACY_CLIMATOLOGY_PATTERN = re.compile(r"^era5_clim_\d{4}\.nc$")
"""Matches the un-suffixed legacy filename and nothing that carries a region."""

MIN_CLIMATOLOGY_COVERAGE = 0.9
"""Share of the event domain the climatology baseline must cover to be usable.

Below this the anomaly is being measured against a baseline from somewhere else.
That has to fail loudly: the symptom is an empty detection list, which reads exactly
like "no extreme weather in this forecast".
"""

ANOMALY_DIRECTION = {
    "total_precipitation": "high",
    "2m_temperature": "high",
    # A cyclone is a *negative* pressure anomaly. Thresholding only the upper tail
    # meant the one event type this project demos with a sea-level-pressure field
    # could never be detected: the Amphan low sits at -7 sigma and was scored as
    # nothing at all.
    "mean_sea_level_pressure": "low",
    "10m_u_component_of_wind": "high",
    "10m_v_component_of_wind": "high",
}
"""Which tail of the z distribution is the extreme, per variable.

Defaults to ``high``. The flagged intensity is always reported as a positive
magnitude of the excursion, so the tracker's severity bands stay comparable
across event types.
"""


# ---------------------------------------------------------------------
# ZIP / NetCDF handling
# ---------------------------------------------------------------------

def _normalise_time_dim(dataset, candidates=("valid_time", "time")):
    """
    Rename a time-like dimension to ``time``.

    The ERA5 archives in this repo carry ``valid_time``, while the rest of this
    module was written against ``time``. Renaming once at the door keeps
    ``isel(time=...)``, ``sizes["time"]`` and ``da.time`` working, rather than
    threading the axis name through the detection loop.
    """

    for name in candidates:

        if name in dataset.dims and name != "time":

            return dataset.rename({name: "time"})

    return dataset


def find_netcdf_in_zip(zip_path):
    """
    Find the first .nc file inside a ZIP archive.
    """

    with zipfile.ZipFile(zip_path, "r") as z:
        nc_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".nc")
        ]

    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF (.nc) file found inside {zip_path}"
        )

    return nc_files[0]


def extract_zip(zip_path):
    """
    Extract a ZIP archive into data/extracted/<archive_name>/.

    Returns the path to the first NetCDF file.
    """

    archive_name = os.path.splitext(
        os.path.basename(zip_path)
    )[0]

    extract_path = os.path.join(
        EXTRACT_DIR,
        archive_name
    )

    os.makedirs(extract_path, exist_ok=True)

    nc_inside = find_netcdf_in_zip(zip_path)

    # Only extract if the target file does not already exist.
    target_nc = os.path.join(
        extract_path,
        os.path.basename(nc_inside)
    )

    if not os.path.exists(target_nc):

        print("[zip] extracting:")
        print(f"      {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract(nc_inside, extract_path)

        print("[zip] extracted:")
        print(f"      {target_nc}")

    else:
        print("[zip] already extracted:")
        print(f"      {target_nc}")

    return target_nc


def get_event_netcdf(event_key):
    """
    Locate the event data.

    Handles:
        1. A normal NetCDF file.
        2. A .nc file that is actually a ZIP archive.
        3. A .zip file containing NetCDF.
    """

    searched = []

    for directory in _EVENT_SEARCH_DIRS:

        nc_path = os.path.join(directory, f"era5_{event_key}.nc")
        zip_path = os.path.join(directory, f"era5_{event_key}.zip")
        searched.extend([nc_path, zip_path])

        # ---------------------------------------------------------
        # Case 1: a .nc file (CDS sometimes returns a ZIP under that name)
        # ---------------------------------------------------------

        if os.path.exists(nc_path):

            if zipfile.is_zipfile(nc_path):

                print(f"[event] {nc_path} is a ZIP archive")
                return extract_zip(nc_path)

            print("[event] using NetCDF:")
            print(f"        {nc_path}")

            return nc_path

        # ---------------------------------------------------------
        # Case 2: an actual ZIP file
        # ---------------------------------------------------------

        if os.path.exists(zip_path):

            print("[event] found ZIP archive:")
            print(f"        {zip_path}")

            return extract_zip(zip_path)

        # ---------------------------------------------------------
        # Case 3: any matching archive
        # ---------------------------------------------------------

        for candidate in sorted(glob.glob(os.path.join(directory, f"era5_{event_key}.*"))):

            if zipfile.is_zipfile(candidate):

                print("[event] found archive:")
                print(f"        {candidate}")

                return extract_zip(candidate)

    searched_text = "\n".join(f"    {path}" for path in searched)
    raise FileNotFoundError(
        f"""
Could not find ERA5 data for event '{event_key}'.

Expected one of:

{searched_text}

Make sure the ERA5 event data has been downloaded.
"""
    )


# ---------------------------------------------------------------------
# Climatology file handling
# ---------------------------------------------------------------------

def get_climatology_files(region_key=None):
    """Climatology archives for one region (or every region when ``None``).

    Scoped by region on purpose. The previous version globbed ``era5_clim_*.nc``
    regardless of the caller's region, so once a second domain was downloaded both
    baselines were concatenated along the synthetic year axis. The two regions have
    different extents, so the result was a union grid with NaN outside each region:
    every event would then have been scored against a baseline that was half from
    another climate. Silence is the dangerous part here -- it looks like a working
    baseline, not a broken one.

    Files still holding the legacy un-suffixed name are the Bay of Bengal baseline and
    are used only when no region-suffixed Bay of Bengal files exist.
    """

    directory = Path(CLIM_DIR)

    if region_key is None:

        files = sorted(directory.glob("era5_clim_*.nc"))

    else:

        files = sorted(directory.glob(f"era5_clim_{region_key}_*.nc"))

        if not files and region_key == LEGACY_CLIMATOLOGY_REGION:

            files = sorted(
                path
                for path in directory.glob("era5_clim_*.nc")
                if LEGACY_CLIMATOLOGY_PATTERN.match(path.name)
            )

    valid_files = []

    for file_path in files:

        file_path = str(file_path)

        # A CDS file may have .nc extension but actually be ZIP.
        if zipfile.is_zipfile(file_path):

            print(f"[clim] archive detected: {file_path}")

            extracted = extract_zip(file_path)
            valid_files.append(extracted)

        else:
            valid_files.append(file_path)

    return valid_files


# ---------------------------------------------------------------------
# Load climatology statistics
# ---------------------------------------------------------------------

def load_climatology_stats(event_key, varname):
    """Climatological mean and standard deviation for one event's region.

    Computed **file by file** as running sums, never by concatenating the archives.
    Concatenating 30 full-May archives along a synthetic ``year`` axis is both
    quadratic in practice -- ``xr.concat`` here produced ``year=N, time=N*248``, so
    30 files asked for a 4.51 GiB array and raised ``MemoryError`` before any
    anomaly could be scored -- and unnecessary: the mean and standard deviation over
    every pooled sample are exactly what a running sum computes, at a memory cost of
    one file at a time.

    NaNs are skipped sample by sample, matching ``xarray``'s default ``skipna``, and
    the standard deviation is the population one (``ddof=0``), matching ``xarray``'s
    default. A cell that is NaN in every sample stays NaN, so the coverage guard
    downstream sees it as uncovered rather than as zero.
    """

    region_key = EVENT_REGIONS.get(event_key, event_key)

    files = get_climatology_files(region_key)

    if not files:

        raise FileNotFoundError(
            f"""
No climatology files found for region '{region_key}' (event '{event_key}').

Expected files such as:

    data/climatology/era5_clim_{region_key}_1991.nc
    data/climatology/era5_clim_{region_key}_1992.nc
    ...

Fetch them first:

    python -m src.data.cds_fetch --climatology --region {region_key} --years 1991-2020
"""
        )

    print(
        f"[clim] loading {len(files)} '{region_key}' climatology files"
    )

    total = None        # running sum of every finite sample
    total_of_squares = None  # running sum of squares
    counts = None       # running count of finite samples
    latitude = longitude = None
    covered_files = 0

    for file_path in files:

        dataset = _normalise_time_dim(
            xr.open_dataset(file_path, engine="netcdf4")
        )

        if varname not in dataset:

            dataset.close()

            raise KeyError(
                f"Variable '{varname}' not found in climatology file {file_path}. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        field = dataset[varname]

        if latitude is None:

            latitude = field["latitude"].values

            longitude = field["longitude"].values

        elif not (
            np.array_equal(latitude, field["latitude"].values)
            and np.array_equal(longitude, field["longitude"].values)
        ):

            dataset.close()

            raise ValueError(
                f"{file_path} is on a different grid from {files[0]}; the "
                "climatology must be one consistent domain, so the files cannot be "
                "pooled into a single baseline"
            )

        # Roll every non-spatial dimension into one axis, then fold that axis into
        # the running sums. Only the requested variable is materialised.
        sample_dims = [
            dim for dim in field.dims if dim not in ("latitude", "longitude")
        ]

        if sample_dims:

            field = field.transpose(*sample_dims, "latitude", "longitude")

        values = np.asarray(field.values, dtype=np.float64)

        finite = np.isfinite(values)

        axes = tuple(range(values.ndim - 2))

        if axes:

            block_total = np.where(finite, values, 0.0).sum(axis=axes)

            block_squares = np.where(finite, values * values, 0.0).sum(axis=axes)

            block_counts = finite.sum(axis=axes)

        else:

            # A field with no sample dimension is itself one sample.
            block_total = np.where(finite, values, 0.0)

            block_squares = np.where(finite, values * values, 0.0)

            block_counts = finite.astype(np.int64)

        if total is None:

            total = block_total

            total_of_squares = block_squares

            counts = block_counts

        else:

            total = total + block_total

            total_of_squares = total_of_squares + block_squares

            counts = counts + block_counts

        covered_files += 1

        dataset.close()

    if total is None:

        raise ValueError(
            f"no climatology files were readable for region '{region_key}'"
        )

    with np.errstate(invalid="ignore", divide="ignore"):

        mean_values = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

        second_moment = np.where(
            counts > 0,
            total_of_squares / np.maximum(counts, 1),
            np.nan,
        )

        # ``mean^2`` can exceed the second moment by float rounding; clip rather than
        # emit a negative variance that sqrt would turn into NaN everywhere.
        variance = np.maximum(second_moment - mean_values * mean_values, 0.0)

        std_values = np.where(counts > 0, np.sqrt(variance), np.nan)

    coords = {"latitude": latitude, "longitude": longitude}

    mean = xr.DataArray(
        mean_values,
        dims=("latitude", "longitude"),
        coords=coords,
        name=varname,
    )
    std = xr.DataArray(
        std_values,
        dims=("latitude", "longitude"),
        coords=coords,
        name=varname,
    )

    print(
        f"[clim] {varname}: mean and std over {int(counts.max())} samples "
        f"per cell, {covered_files} file(s)"
    )

    return mean, std


# ---------------------------------------------------------------------
# Detect anomalies
# ---------------------------------------------------------------------

def detect_anomalies(event_key, variable):

    if variable not in VAR_MAP:

        raise ValueError(
            f"Unknown variable '{variable}'. "
            f"Choices: {list(VAR_MAP)}"
        )

    varname = VAR_MAP[variable]

    # -------------------------------------------------------------
    # Locate and extract event data
    # -------------------------------------------------------------

    event_path = get_event_netcdf(event_key)

    print(f"[event] loading {event_path}")

    # Explicitly use netCDF4.
    ds = _normalise_time_dim(
        xr.open_dataset(
            event_path,
            engine="netcdf4"
        )
    )

    if varname not in ds:

        raise KeyError(
            f"Variable '{varname}' not found in event dataset.\n"
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[varname]

    print(
        f"[event] variable={varname} "
        f"dimensions={da.dims} "
        f"shape={da.shape}"
    )

    # -------------------------------------------------------------
    # Load climatology
    # -------------------------------------------------------------

    clim_mean, clim_std = load_climatology_stats(
        event_key,
        varname
    )

    # -------------------------------------------------------------
    # Match climatology grid to event grid
    # -------------------------------------------------------------

    reference_frame = da.isel(time=0)

    clim_mean = clim_mean.interp_like(
        reference_frame
    )

    clim_std = clim_std.interp_like(
        reference_frame
    ).clip(
        min=1e-6
    )

    # -------------------------------------------------------------
    # Refuse a climatology that does not cover the event domain
    # -------------------------------------------------------------

    coverage = float(
        np.mean(np.isfinite(clim_mean.values))
    )

    if coverage < MIN_CLIMATOLOGY_COVERAGE:

        raise ValueError(
            f"The climatology baseline covers only {coverage:.1%} of the "
            f"'{event_key}' domain, below the {MIN_CLIMATOLOGY_COVERAGE:.0%} "
            f"floor.\n"
            f"    climatology: lat {float(clim_mean.latitude.min()):g}-"
            f"{float(clim_mean.latitude.max()):g}, "
            f"lon {float(clim_mean.longitude.min()):g}-"
            f"{float(clim_mean.longitude.max()):g}\n"
            f"    event:       lat {float(da.latitude.min()):g}-{float(da.latitude.max()):g}, "
            f"lon {float(da.longitude.min()):g}-{float(da.longitude.max()):g}\n"
            "Scoring this event against that baseline would report 'no anomaly' "
            "where no comparison was possible. Fetch a climatology for this region "
            "(docs/dataset.md) instead of relaxing the floor."
        )

    # -------------------------------------------------------------
    # Detect anomalous regions
    # -------------------------------------------------------------

    results = []

    direction = ANOMALY_DIRECTION.get(variable, "high")

    print(
        f"[event] extreme direction={direction} "
        f"threshold={Z_THRESHOLD} sigma"
    )

    for t in range(da.sizes["time"]):

        frame = da.isel(time=t)

        z = (
            frame - clim_mean
        ) / clim_std

        if direction == "high":

            flagged = (
                z.values > Z_THRESHOLD
            )

        else:

            flagged = (
                z.values < -Z_THRESHOLD
            )

        if not flagged.any():

            results.append(
                {
                    "time_index": t,
                    "timestamp": str(
                        da.time.values[t]
                    ),
                    "boxes": [],
                }
            )

            continue

        # ---------------------------------------------------------
        # Connected-component labeling
        # ---------------------------------------------------------

        labeled, num_features = ndimage.label(
            flagged
        )

        boxes = []

        for label_id in range(
            1,
            num_features + 1
        ):

            ys, xs = np.where(
                labeled == label_id
            )

            # Ignore extremely tiny regions.
            if len(ys) < 2:
                continue

            lat_vals = (
                da.latitude.values[ys]
            )

            lon_vals = (
                da.longitude.values[xs]
            )

            boxes.append(
                {
                    "label_id": int(label_id),

                    "lat_min": float(
                        lat_vals.min()
                    ),

                    "lat_max": float(
                        lat_vals.max()
                    ),

                    "lon_min": float(
                        lon_vals.min()
                    ),

                    "lon_max": float(
                        lon_vals.max()
                    ),

                    "centroid_lat": float(
                        lat_vals.mean()
                    ),

                    "centroid_lon": float(
                        lon_vals.mean()
                    ),

                    # The magnitude of the excursion on the tail that was flagged. A
                    # cyclonic low would otherwise rank as the weakest box in the
                    # frame, despite being the most extreme value in the field.
                    "peak_zscore": float(
                        z.values[ys, xs].max()
                        if direction == "high"
                        else -z.values[ys, xs].min()
                    ),

                    "cell_count": int(
                        len(ys)
                    ),
                }
            )

        boxes.sort(
            key=lambda b: b["peak_zscore"],
            reverse=True
        )

        results.append(
            {
                "time_index": t,

                "timestamp": str(
                    da.time.values[t]
                ),

                "boxes": boxes,
            }
        )

    ds.close()

    return results


# ---------------------------------------------------------------------
# Build trajectory
# ---------------------------------------------------------------------

def build_trajectory(results):

    trajectory = []

    for frame in results:

        if frame["boxes"]:

            top = frame["boxes"][0]

            trajectory.append(
                {
                    "timestamp": frame["timestamp"],

                    "lat": top["centroid_lat"],

                    "lon": top["centroid_lon"],

                    "severity_zscore": top[
                        "peak_zscore"
                    ],
                }
            )

    return trajectory


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--event",
        required=True,
        choices=["amphan", "heatwave"],
    )

    ap.add_argument(
        "--variable",
        default="total_precipitation",
        choices=list(VAR_MAP),
    )

    args = ap.parse_args()

    print()
    print("=" * 70)
    print("ERA5 ANOMALY DETECTION")
    print("=" * 70)
    print(f"Event    : {args.event}")
    print(f"Variable : {args.variable}")
    print("=" * 70)
    print()

    results = detect_anomalies(
        args.event,
        args.variable
    )

    trajectory = build_trajectory(
        results
    )

    out_path = os.path.join(
        OUT_DIR,
        f"anomalies_{args.event}.json"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "frames": results,
                "trajectory": trajectory,
            },
            f,
            indent=2
        )

    print()
    print(f"[done] wrote {out_path}")

    print(
        f"[summary] "
        f"{len(trajectory)} timesteps "
        f"with a tracked anomaly"
    )

    if trajectory:

        peak = max(
            trajectory,
            key=lambda p: p["severity_zscore"]
        )

        print(
            f"[peak] {peak}"
        )
