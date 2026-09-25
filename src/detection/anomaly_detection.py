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
import zipfile

import numpy as np
import xarray as xr
from scipy import ndimage


# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output")
)

EXTRACT_DIR = os.path.join(DATA_DIR, "extracted")

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


# ---------------------------------------------------------------------
# ZIP / NetCDF handling
# ---------------------------------------------------------------------

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

        print(f"[zip] extracting:")
        print(f"      {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract(nc_inside, extract_path)

        print(f"[zip] extracted:")
        print(f"      {target_nc}")

    else:
        print(f"[zip] already extracted:")
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

    nc_path = os.path.join(
        DATA_DIR,
        f"era5_{event_key}.nc"
    )

    zip_path = os.path.join(
        DATA_DIR,
        f"era5_{event_key}.zip"
    )

    # -------------------------------------------------------------
    # Case 1:
    # Normal NetCDF file
    # -------------------------------------------------------------

    if os.path.exists(nc_path):

        if zipfile.is_zipfile(nc_path):

            print(f"[event] {nc_path} is a ZIP archive")
            return extract_zip(nc_path)

        print(f"[event] using NetCDF:")
        print(f"        {nc_path}")

        return nc_path

    # -------------------------------------------------------------
    # Case 2:
    # Actual ZIP file
    # -------------------------------------------------------------

    if os.path.exists(zip_path):

        print(f"[event] found ZIP archive:")
        print(f"        {zip_path}")

        return extract_zip(zip_path)

    # -------------------------------------------------------------
    # Case 3:
    # Try to find any matching ZIP/NC file
    # -------------------------------------------------------------

    candidates = glob.glob(
        os.path.join(
            DATA_DIR,
            f"era5_{event_key}.*"
        )
    )

    for candidate in candidates:

        if zipfile.is_zipfile(candidate):

            print(f"[event] found archive:")
            print(f"        {candidate}")

            return extract_zip(candidate)

    raise FileNotFoundError(
        f"""
Could not find ERA5 data for event '{event_key}'.

Expected one of:

    {nc_path}
    {zip_path}

Make sure the ERA5 event data has been downloaded.
"""
    )


# ---------------------------------------------------------------------
# Climatology file handling
# ---------------------------------------------------------------------

def get_climatology_files():

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "climatology",
                "era5_clim_*.nc"
            )
        )
    )

    valid_files = []

    for file_path in files:

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

def load_climatology_stats(region_key, varname):

    files = get_climatology_files()

    if not files:

        raise FileNotFoundError(
            """
No climatology files found.

Expected files such as:

    data/climatology/era5_clim_1991.nc
    data/climatology/era5_clim_1992.nc
    ...

Run the climatology download first.
"""
        )

    print(
        f"[clim] loading {len(files)} climatology files"
    )

    # Open each file separately so ZIP-extracted files work reliably.
    datasets = []

    for file_path in files:

        print(f"[clim] opening {file_path}")

        datasets.append(
            xr.open_dataset(
                file_path,
                engine="netcdf4"
            )
        )

    # Combine datasets along a synthetic year dimension.
    ds = xr.concat(
        datasets,
        dim="year"
    )

    if varname not in ds:

        raise KeyError(
            f"Variable '{varname}' not found in climatology data. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[varname]

    mean = da.mean(
        dim=["year", "time"]
    )

    std = da.std(
        dim=["year", "time"]
    )

    return mean.compute(), std.compute()


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
    ds = xr.open_dataset(
        event_path,
        engine="netcdf4"
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
    # Detect anomalous regions
    # -------------------------------------------------------------

    results = []

    for t in range(da.sizes["time"]):

        frame = da.isel(time=t)

        z = (
            frame - clim_mean
        ) / clim_std

        flagged = (
            z.values > Z_THRESHOLD
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

                    "peak_zscore": float(
                        z.values[ys, xs].max()
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