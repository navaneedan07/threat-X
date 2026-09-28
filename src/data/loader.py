"""ERA5 dataset loader.

Handles the CDS-downloaded ZIP-wrapped NetCDF files in data/raw/.
Each file is a ZIP archive. ERA5 ships two inner NetCDF files; ERA5-Land ships one:
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
import os
import tempfile
import zipfile
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

# Canonical inner-file names inside the CDS ZIP archives.
_INNER_INSTANT = "data_stream-oper_stepType-instant.nc"
_INNER_ACCUM = "data_stream-oper_stepType-accum.nc"

# Engines in preference order: in-memory first, then a temporary-file fallback.
_ENGINES: tuple[str, ...] = ("h5netcdf", "netcdf4")
_ENGINE_ATTR = "loader_engine"


def _open_inner(zf: zipfile.ZipFile, inner_name: str) -> xr.Dataset:
    """Read one inner NetCDF from the ZIP into an xarray Dataset.

    h5netcdf is tried first because it reads straight from the in-memory buffer.

    It is not the *only* option though, and must not be a hard requirement: as of
    h5netcdf v1.x the HDF5 backend is an optional dependency, so
    ``pip install -r requirements.txt`` can leave h5netcdf installed but unable to
    open anything ("No module named 'h5py'"). netCDF4 cannot read a file object,
    so the fallback writes the inner member to a temporary file, opens it from
    there, and materialises it into memory before removing the file — lazily
    reading a deleted temp file would fail on Windows.

    The returned Dataset is the same either way; the engine that produced it is
    recorded in ``ds.attrs`` so a result can always be traced to its reader.
    """
    data = zf.read(inner_name)
    attempts: list[str] = []

    for engine in _ENGINES:
        try:
            if engine == "h5netcdf":
                dataset = xr.open_dataset(io.BytesIO(data), engine=engine)
            else:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
                    handle.write(data)
                    temp_path = handle.name
                try:
                    # Opened as a context manager so the netCDF4 file handle is
                    # released before the temp file is removed. `.load()` alone
                    # copies the data but leaves the store open, and Windows then
                    # refuses the unlink with WinError 32.
                    with xr.open_dataset(temp_path, engine=engine) as opened:
                        dataset = opened.load()
                finally:
                    os.unlink(temp_path)
            dataset.attrs[_ENGINE_ATTR] = engine
            if attempts:
                logger.debug("Read %s with fallback engine %s", inner_name, engine)
            return dataset
        except Exception as exc:  # try the next engine
            attempts.append(f"{engine}: {exc}")

    raise RuntimeError(
        f"Could not read {inner_name} with any engine "
        f"({' | '.join(attempts)}). Install h5py to use the h5netcdf engine "
        "(see requirements.txt)."
    )


def _select_inner_names(available: set[str]) -> list[str]:
    """Choose which archive members to read.

    ERA5 single-levels ships two canonical members (instant + accum). ERA5-Land and
    some other CDS products ship a single combined ``data_0.nc``. Falling back to
    every top-level ``*.nc`` member supports both layouts without a second loader.
    """
    canonical = [name for name in (_INNER_INSTANT, _INNER_ACCUM) if name in available]
    if canonical:
        return canonical
    return sorted(
        name for name in available if name.endswith(".nc") and not name.startswith(".")
    )


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

        for inner_name in _select_inner_names(available):
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
