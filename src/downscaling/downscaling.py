"""Local-grid downscaling entry point: crop a coarse field, then resample it.

This module is a **thin, deterministic adapter** over
:mod:`src.downscaling.baseline`. It exists for the one shape the localization
stage actually needs — "take the threat's bounding box out of a coarse NWP field
and produce a finer local grid" — and it contains **no interpolation logic of its
own**.

Why there is only one implementation
------------------------------------
An earlier version of this file was a second, independent downscaler: it
multiplied by a hard-coded ``2.4`` zoom factor (so it could only ever do the one
12 km → 5 km case), added ``np.random.normal`` noise (making the output
non-reproducible and untestable), and used a different interpolation order to the
baseline, so the two modules disagreed about the field they were producing. That
is the exact class of defect the project forbids: a number nobody can reproduce,
contradicting the module the metrics are measured against.

Everything here therefore delegates to :mod:`src.downscaling.baseline`:

* the upscale factor is **derived** from the source and target grids;
* the interpolation order comes from ``configs/model.yaml``;
* the output carries ``trained: False`` and the same provenance the baseline
  records, so a metric table can never mistake this for a learned model;
* nothing is random — the same input produces the same output on every run.

The learned super-resolution / diffusion models, when they exist, live in
``src/models/downscaling/`` and are scored against this baseline on
``src/downscaling/metrics.py``.
"""

from __future__ import annotations

import numpy as np

from src.downscaling.baseline import (
    downscale,
    downscale_dataarray,
    downscale_to_reference,
    method_name,
    resample,
    resolve_order,
    target_grid,
    upscale_factor,
)
from src.shared.fields import FieldError, GriddedField

__all__ = [
    "downscale",
    "downscale_box",
    "downscale_dataarray",
    "downscale_field",
    "downscale_to_reference",
    "local_crop",
    "method_name",
    "resample",
    "resolve_order",
    "target_grid",
    "upscale_factor",
]


def _axis_slice(
    axis: np.ndarray,
    limits: tuple[float, float] | None,
    name: str,
) -> slice:
    """Index slice covering ``limits`` (inclusive) on a monotonic axis.

    Accepts a descending latitude axis, which is what ERA5 ships, because the
    indices come from ``flatnonzero`` on the axis and are therefore already in
    array order. A window that selects fewer than two cells is refused rather
    than silently producing a one-row field that every metric would then report
    on.
    """
    if limits is None:
        return slice(None)
    low, high = sorted(float(value) for value in limits)
    if not np.isfinite([low, high]).all():
        raise FieldError(f"{name}: bounds must be finite, got {limits!r}")
    indices = np.flatnonzero((axis >= low) & (axis <= high))
    if indices.size < 2:
        raise FieldError(
            f"{name}: bounds {limits!r} select {indices.size} cell(s) from "
            f"extent [{float(axis.min()):g}, {float(axis.max()):g}]; need at least 2"
        )
    if indices[-1] - indices[0] + 1 != indices.size:
        raise FieldError(
            f"{name}: bounds {limits!r} select a non-contiguous window; the field "
            "may be irregularly spaced"
        )
    return slice(int(indices[0]), int(indices[-1]) + 1)


def local_crop(
    field: GriddedField,
    *,
    latitude_range: tuple[float, float] | None = None,
    longitude_range: tuple[float, float] | None = None,
    name: str | None = None,
) -> GriddedField:
    """Subset ``field`` to a lat/lon window, keeping the grid spacing.

    Only the *bounding box* is cut out here; no interpolation happens. This is the
    "threat bounding box → local crop" step in the README's downscaling pipeline.
    """
    if latitude_range is None and longitude_range is None:
        raise ValueError("supply latitude_range and/or longitude_range to crop")
    lat_slice = _axis_slice(field.latitude, latitude_range, "latitude")
    lon_slice = _axis_slice(field.longitude, longitude_range, "longitude")
    return GriddedField(
        values=field.values[lat_slice, lon_slice].copy(),
        latitude=field.latitude[lat_slice].copy(),
        longitude=field.longitude[lon_slice].copy(),
        name=name or f"{field.name}_crop",
        units=field.units,
        attrs={**field.attrs, "crop": "src.downscaling.downscaling.local_crop"},
    )


def downscale_field(
    source: GriddedField,
    *,
    target_resolution_deg: float | None = None,
    target_latitude: np.ndarray | None = None,
    target_longitude: np.ndarray | None = None,
    order: int | None = None,
    config: dict | None = None,
    data_config: dict | None = None,
    name: str | None = None,
) -> GriddedField:
    """Resample a coarse field, delegating to the shared baseline.

    A named re-export rather than a reimplementation, so the baseline stays the
    single source of truth for how a field is refined.
    """
    return downscale(
        source,
        target_resolution_deg=target_resolution_deg,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        order=order,
        config=config,
        data_config=data_config,
        name=name,
    )


def downscale_box(
    source: GriddedField,
    *,
    latitude_range: tuple[float, float] | None = None,
    longitude_range: tuple[float, float] | None = None,
    target_resolution_deg: float | None = None,
    target_latitude: np.ndarray | None = None,
    target_longitude: np.ndarray | None = None,
    order: int | None = None,
    config: dict | None = None,
    data_config: dict | None = None,
) -> GriddedField:
    """Crop to a box, then resample it onto a finer grid.

    The composite step the localization stage performs: a threat bounding box is
    taken out of a coarse field and refined. The crop happens first so the
    interpolation only ever runs over the region of interest.
    """
    cropped = local_crop(
        source,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
    )
    return downscale(
        cropped,
        target_resolution_deg=target_resolution_deg,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        order=order,
        config=config,
        data_config=data_config,
    )


def _read_field(
    path: str,
    variable: str,
    *,
    time_index: int | None,
    latitude_range: tuple[float, float] | None,
    longitude_range: tuple[float, float] | None,
) -> GriddedField:
    """Open one variable of a NetCDF file as a :class:`GriddedField`."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        if variable not in dataset.data_vars:
            available = ", ".join(sorted(dataset.data_vars))
            raise FieldError(
                f"{path}: variable {variable!r} not found; available: {available}"
            )
        data = dataset[variable]
        if time_index is not None and "time" in data.dims:
            data = data.isel(time=time_index)
        field = GriddedField.from_dataarray(data, name=variable)
    if latitude_range is not None or longitude_range is not None:
        field = local_crop(
            field,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
        )
    return field


def main(argv: list[str] | None = None) -> int:
    """CLI: resample one variable of a NetCDF file onto a finer grid.

    Every parameter that decides the output is explicit — target resolution,
    interpolation order and the optional crop. There is no default upscale factor,
    so the command cannot silently produce a factor nobody chose.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Deterministic coarse -> fine downscaling using the interpolation "
            "baseline (no learned model, no added noise)."
        )
    )
    parser.add_argument("--input", required=True, help="coarse NetCDF file")
    parser.add_argument("--variable", required=True, help="variable name in the file")
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=None,
        help=(
            "target grid spacing in degrees; if omitted, taken from "
            "configs/data.yaml grid.fine_resolution_deg (which may be null)"
        ),
    )
    parser.add_argument("--time-index", type=int, default=None, help="index along 'time'")
    parser.add_argument(
        "--method",
        choices=("bilinear", "nearest"),
        default=None,
        help="interpolation method; default comes from configs/model.yaml",
    )
    parser.add_argument(
        "--lat-range", type=float, nargs=2, default=None, metavar=("LAT0", "LAT1")
    )
    parser.add_argument(
        "--lon-range", type=float, nargs=2, default=None, metavar=("LON0", "LON1")
    )
    parser.add_argument("--output", default=None, help="write the refined field to NetCDF")
    args = parser.parse_args(argv)

    order = None if args.method is None else (1 if args.method == "bilinear" else 0)

    field = _read_field(
        args.input,
        args.variable,
        time_index=args.time_index,
        latitude_range=tuple(args.lat_range) if args.lat_range else None,
        longitude_range=tuple(args.lon_range) if args.lon_range else None,
    )
    refined = downscale(field, target_resolution_deg=args.resolution_deg, order=order)

    source_dlat, _ = field.resolution_deg()
    target_dlat, _ = refined.resolution_deg()
    print(f"source      {field.shape} @ {source_dlat:g} deg")
    print(f"refined     {refined.shape} @ {target_dlat:g} deg")
    print(f"method      {refined.attrs.get('method')} (trained={refined.attrs.get('trained')})")
    print(f"upscale     {refined.attrs.get('upscale_factor')}")
    print(f"coverage    {refined.attrs.get('coverage_fraction'):.3f}")

    if args.output:
        refined.to_dataarray().to_netcdf(args.output)
        print(f"wrote       {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    raise SystemExit(main())
