"""ERA5 dataset loader.

Handles the CDS-downloaded ZIP-wrapped NetCDF files in data/raw/.
Each file is a ZIP archive containing two inner NetCDF files:
  - data_stream-oper_stepType-instant.nc  (instantaneous: u10, v10, t2m, msl)
  - data_stream-oper_stepType-accum.nc    (accumulated: tp)

All outputs are returned as xarray Datasets with coordinates standardised to:
  - valid_time (datetime64): observation timestamps (3-hourly)
  - latitude  (float): 0.25 deg grid
  - longitude (float): 0.25 deg grid

Units are preserved as-is from ERA5:
  - u10, v10: m/s
  - t2m:      K
  - msl:      Pa  (divide by 100 for hPa)
  - tp:       m   (multiply by 1000 for mm)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Canonical inner-file names inside the CDS ZIP archives.
_INNER_INSTANT = "data_stream-oper_stepType-instant.nc"
_INNER_ACCUM = "data_stream-oper_stepType-accum.nc"


def _open_inner(zf: zipfile.ZipFile, inner_name: str) -> xr.Dataset:
    """Read one inner NetCDF from the ZIP into an xarray Dataset."""
    data = zf.read(inner_name)
    buf = io.BytesIO(data)
    # h5netcdf is the only working backend for these HDF5-format files
    # on this Python / netCDF4 combination (netCDF4 does not support file objects).
    return xr.open_dataset(buf, engine="h5netcdf")


def load_dataset(archive_path: str | Path) -> xr.Dataset:
    """Load an ERA5 CDS archive and return a merged xarray Dataset.

    Parameters
    ----------
    archive_path : path-like
        Path to the outer .nc file (which is actually a ZIP archive).

    Returns
    -------
    xr.Dataset
        Merged dataset containing all available variables, with a unified
        ``valid_time`` dimension and squeezed singleton dimensions (number, expver).
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"{archive_path.name} is not a ZIP/CDS archive. "
            "Only CDS-downloaded archives are supported at this stage."
        )

    logger.info("Loading ERA5 archive: %s", archive_path.name)

    with zipfile.ZipFile(archive_path) as zf:
        available = set(zf.namelist())
        parts: list[xr.Dataset] = []

        for inner_name in [_INNER_INSTANT, _INNER_ACCUM]:
            if inner_name in available:
                try:
                    ds = _open_inner(zf, inner_name)
                    # Drop nuisance coordinates introduced by the CDS download format.
                    # 'number': scalar ensemble member index (always 0 for deterministic run).
                    # 'expver': string version tag repeated along valid_time; not needed.
                    coords_to_drop = [c for c in ["number", "expver"] if c in ds.coords]
                    if coords_to_drop:
                        ds = ds.drop_vars(coords_to_drop)
                    parts.append(ds)
                    logger.debug("Loaded inner file: %s (%s)", inner_name, list(ds.data_vars))
                except Exception as exc:
                    logger.warning("Could not read %s: %s", inner_name, exc)
            else:
                logger.warning("Inner file not found in archive: %s", inner_name)

    if not parts:
        raise RuntimeError(f"Could not read any inner NetCDF from: {archive_path}")

    if len(parts) == 1:
        merged = parts[0]
    else:
        # Merge on the common dimensions.  Use compat='override' to avoid
        # conflicts on shared coordinate values.
        merged = xr.merge(parts, compat="override")

    logger.info(
        "Loaded %s: vars=%s, times=%d, grid=%dx%d",
        archive_path.name,
        list(merged.data_vars),
        merged.sizes.get("valid_time", 0),
        merged.sizes.get("latitude", 0),
        merged.sizes.get("longitude", 0),
    )
    return merged


def get_time_step_hours(ds: xr.Dataset) -> float:
    """Return the uniform time step of the dataset in hours.

    Returns NaN if the dataset has fewer than two time steps.
    """
    if "valid_time" not in ds.coords or ds.sizes["valid_time"] < 2:
        return float("nan")
    times = ds["valid_time"].values
    delta = times[1] - times[0]
    # numpy timedelta64 subtraction result type varies by numpy/Python version:
    # - numpy timedelta64: .item() returns datetime.timedelta or int (ns)
    # - Handle both cases robustly.
    if hasattr(delta, 'item'):
        item = delta.item()  # convert numpy timedelta64 to Python object
    else:
        item = delta
    if hasattr(item, 'total_seconds'):
        # datetime.timedelta
        return item.total_seconds() / 3600.0
    else:
        # integer nanoseconds
        return float(item) / 1e9 / 3600.0


def spatial_resolution_deg(ds: xr.Dataset) -> float:
    """Return the (assumed uniform) lat/lon grid spacing in degrees."""
    if "latitude" not in ds.coords or ds.sizes["latitude"] < 2:
        return float("nan")
    lats = ds["latitude"].values
    return float(abs(lats[1] - lats[0]))
