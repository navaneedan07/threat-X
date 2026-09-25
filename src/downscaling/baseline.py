"""Deterministic interpolation downscaling baseline (coarse → fine).

This is the rung every downscaling claim is measured against. The README requires
the baseline to stay available and to be reported alongside any learned model,
because a bilinear field that preserves the peak can legitimately beat a
super-resolution network that smooths it away.

Deliberate design points
------------------------
* **The upscale factor is derived, never hard-coded.** ``configs/data.yaml`` has
  ``coarse_resolution_deg`` and ``fine_resolution_deg`` as ``null`` because no
  dataset is confirmed. Nothing here multiplies by a fixed 2.4 — the factor is
  computed from the source and target grids, so this works the day Aravinth
  confirms the resolutions, with no rewrite.
* **Only the two documented methods exist.** ``configs/model.yaml`` defines
  ``order: 1`` (bilinear) and ``order: 0`` (nearest-neighbour). Anything else
  raises rather than quietly substituting a method nobody evaluated.
* **Out-of-domain targets are NaN, not extrapolated.** A target grid larger than
  the source is a real mistake (or a demo reaching past its data), and inventing
  edge values would flatter the metrics. ``coverage_fraction`` records how much
  of the target was actually covered.

This module is deterministic and untrained. It lives in ``src/downscaling/``
rather than ``src/models/`` for that reason — see ``docs/architecture.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.shared.config import ConfigError, get_value, stage_config
from src.shared.fields import GriddedField

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

CONFIG_FILE = "model.yaml"

# Interpreted from configs/model.yaml -> downscaling.interpolation.order
_METHODS: dict[int, str] = {0: "nearest", 1: "linear"}


def resolve_order(order: int | None = None, config: dict | None = None) -> int:
    """Return the interpolation order, defaulting to ``configs/model.yaml``.

    An explicit ``order`` argument always wins, so a caller can compare methods
    without editing config.
    """
    if order is not None:
        selected = order
    else:
        loaded = config if config is not None else stage_config(CONFIG_FILE)
        selected = get_value(loaded, "downscaling.interpolation.order", default=1)
    selected = int(selected)
    if selected not in _METHODS:
        allowed = ", ".join(f"{key} ({name})" for key, name in sorted(_METHODS.items()))
        raise ValueError(
            f"interpolation order {selected} is not implemented; allowed: {allowed}. "
            "Set configs/model.yaml -> downscaling.interpolation.order, or use a "
            "learned model in src/models/downscaling/."
        )
    return selected


def method_name(order: int | None = None, config: dict | None = None) -> str:
    """Human-readable method name for provenance records."""
    return _METHODS[resolve_order(order, config)]


def upscale_factor(
    source_resolution_deg: float,
    target_resolution_deg: float,
) -> float:
    """Ratio of target to source cell count along one axis.

    ``upscale_factor(0.25, 0.05) == 5.0``. Not required to be an integer: the
    problem statement's 12 km → 5 km is a factor of 2.4, and a baseline that
    assumed integer factors would fail on exactly the case being demoed.
    """
    if source_resolution_deg <= 0 or target_resolution_deg <= 0:
        raise ValueError("resolutions must be positive")
    return float(source_resolution_deg / target_resolution_deg)


def target_grid(
    source: GriddedField,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a target grid covering the source extent at ``resolution_deg``.

    Coordinates are snapped to multiples of the resolution so the target grid is
    reproducible and shareable — a target grid that depends on float drift makes
    two runs incomparable.
    """
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")

    def _axis(values: np.ndarray) -> np.ndarray:
        low = float(np.ceil(values.min() / resolution_deg) * resolution_deg)
        high = float(np.floor(values.max() / resolution_deg) * resolution_deg)
        count = int(round((high - low) / resolution_deg)) + 1
        return np.round(low + resolution_deg * np.arange(count), 10)

    latitude = _axis(source.latitude)
    longitude = _axis(source.longitude)
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError(
            f"target resolution {resolution_deg} deg leaves fewer than 2 points on "
            f"the source extent; source resolution is {source.resolution_deg()}"
        )
    return latitude, longitude


def resample(
    source: GriddedField,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    *,
    order: int | None = None,
    config: dict | None = None,
    name: str | None = None,
) -> GriddedField:
    """Resample ``source`` onto the target axes.

    The target axes are parameters, which is what makes this usable for *both*
    directions the project needs: upsampling a coarse NWP field to a local grid,
    and downsampling a fine reference onto a coarse grid inside the tests.

    Cells whose target coordinate falls outside the source domain are NaN.
    """
    selected = resolve_order(order, config)
    method = _METHODS[selected]
    ascending = source.ascending()

    interpolator = RegularGridInterpolator(
        (ascending.latitude, ascending.longitude),
        ascending.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    target_latitude = np.asarray(target_latitude, dtype=float)
    target_longitude = np.asarray(target_longitude, dtype=float)
    mesh_lon, mesh_lat = np.meshgrid(target_longitude, target_latitude)
    points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
    values = interpolator(points).reshape(mesh_lat.shape)

    inside = (
        (mesh_lat >= ascending.latitude[0])
        & (mesh_lat <= ascending.latitude[-1])
        & (mesh_lon >= ascending.longitude[0])
        & (mesh_lon <= ascending.longitude[-1])
    )
    coverage = float(np.mean(inside))

    source_dlat, source_dlon = source.resolution_deg()
    target_dlat = (
        float(np.median(np.abs(np.diff(target_latitude))))
        if target_latitude.size > 1
        else source_dlat
    )
    target_dlon = (
        float(np.median(np.abs(np.diff(target_longitude))))
        if target_longitude.size > 1
        else source_dlon
    )

    return GriddedField(
        values=values,
        latitude=target_latitude,
        longitude=target_longitude,
        name=name or source.name,
        units=source.units,
        attrs={
            "provenance": "src.downscaling.baseline.resample",
            "method": f"interpolation:{method}",
            "order": selected,
            "source_resolution_deg": (source_dlat, source_dlon),
            "target_resolution_deg": (target_dlat, target_dlon),
            "source_shape": source.shape,
            "upscale_factor": (
                upscale_factor(source_dlat, target_dlat),
                upscale_factor(source_dlon, target_dlon),
            ),
            "coverage_fraction": coverage,
            "trained": False,
            "synthetic": source.attrs.get("synthetic", False),
            # Recorded so a metric table says plainly when config asked for a
            # learned method but the deterministic baseline produced the numbers.
            "configured_method": get_value(
                config if config is not None else stage_config(CONFIG_FILE),
                "downscaling.method",
                default="interpolation",
            ),
        },
    )


def downscale(
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
    """Upsample a coarse field, either to a resolution or onto an explicit grid.

    Supply ``target_latitude``/``target_longitude`` when the fine grid already
    exists (the reference to be compared against), or ``target_resolution_deg``
    to build one over the source extent.

    With neither, the resolution is read from
    ``configs/data.yaml -> grid.fine_resolution_deg``. That key is deliberately
    ``null`` today, so the error tells you to confirm the dataset rather than
    silently picking an upscale factor.
    """
    if target_latitude is not None or target_longitude is not None:
        if target_latitude is None or target_longitude is None:
            raise ValueError(
                "supply both target_latitude and target_longitude, or neither"
            )
        return resample(
            source,
            target_latitude,
            target_longitude,
            order=order,
            config=config,
            name=name,
        )

    resolution = target_resolution_deg
    if resolution is None:
        loaded = data_config if data_config is not None else stage_config("data.yaml")
        resolution = get_value(loaded, "grid.fine_resolution_deg")
    if resolution is None:
        raise ConfigError(
            "target resolution is unresolved: pass target_resolution_deg, or set "
            "configs/data.yaml -> grid.fine_resolution_deg once the dataset's coarse and "
            "fine grids are confirmed (see docs/dataset.md). Do not assume a factor."
        )

    latitude, longitude = target_grid(source, resolution)
    return resample(
        source, latitude, longitude, order=order, config=config, name=name
    )


def downscale_to_reference(
    source: GriddedField,
    reference: GriddedField,
    *,
    order: int | None = None,
    config: dict | None = None,
) -> GriddedField:
    """Upsample ``source`` onto ``reference``'s exact grid.

    The pairing the metrics need: interpolation output and reference must sit on
    identical axes before any element-wise comparison is meaningful.
    """
    return resample(
        source,
        reference.latitude,
        reference.longitude,
        order=order,
        config=config,
        name=f"{source.name}_downscaled",
    )


def downscale_dataarray(
    data: xr.DataArray,
    *,
    target_resolution_deg: float | None = None,
    target_latitude: np.ndarray | None = None,
    target_longitude: np.ndarray | None = None,
    order: int | None = None,
    config: dict | None = None,
) -> xr.DataArray:
    """xarray in, xarray out — the interface the pipeline actually calls.

    Kept as a thin wrapper so the algorithm above stays testable with numpy only.
    xarray is imported through :class:`GriddedField`, not here.
    """
    field = GriddedField.from_dataarray(data)
    result = downscale(
        field,
        target_resolution_deg=target_resolution_deg,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        order=order,
        config=config,
    )
    return result.to_dataarray()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from src.shared.synthetic import sharp_peak_field, smoothed_field

    fine = sharp_peak_field()
    coarse = resample(fine, fine.latitude[::10], fine.longitude[::10], order=1)
    refined = downscale_to_reference(coarse, fine)
    print(f"reference peak      {fine.peak:.3f} {fine.units}")
    print(f"coarse peak         {coarse.peak:.3f}")
    print(f"refined peak        {refined.peak:.3f}")
    print(f"smoothed peak       {smoothed_field(fine).peak:.3f}")
    print(f"coverage            {refined.attrs['coverage_fraction']:.3f}")
    print(f"upscale factor      {refined.attrs['upscale_factor']}")
