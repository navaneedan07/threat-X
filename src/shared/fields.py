"""Gridded field container and xarray bridges.

A gridded weather field is a 2-D array plus its latitude and longitude axes. The
pipeline has to move these between the loader (xarray/NetCDF), the interpolation
baseline (numpy/scipy) and the metrics (numpy), so the axis convention is fixed
here once:

* ``values`` is indexed ``[latitude, longitude]`` — the ERA5 / xarray order.
* ``latitude`` and ``longitude`` are 1-D coordinate arrays. Latitude may be
  **ascending or descending**: ERA5 ships it descending (90 → -90) and every
  scipy interpolator requires ascending, so handling both once here removes a
  whole class of silent axis-flip bugs from the stage modules.
* Longitude is ascending and is assumed to be within one world, i.e. no
  dateline wrap. Crossing 180° is a real limitation and is documented, not
  guessed at.

xarray is imported lazily inside the two bridge methods, so this module — and
everything that imports it — works on a numpy-only install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr


class FieldError(ValueError):
    """A field is malformed (shape mismatch, non-monotonic axis, bad range)."""


def _check_axis(axis: np.ndarray, name: str) -> np.ndarray:
    """Validate a 1-D coordinate axis and return it as float64."""
    array = np.asarray(axis, dtype=float)
    if array.ndim != 1:
        raise FieldError(f"{name}: expected a 1-D coordinate array, got {array.ndim}-D")
    if array.size < 2:
        raise FieldError(f"{name}: need at least 2 points, got {array.size}")
    if not np.isfinite(array).all():
        raise FieldError(f"{name}: contains NaN or inf")
    differences = np.diff(array)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise FieldError(
            f"{name}: must be strictly monotonic; found a direction change or a repeat"
        )
    return array


@dataclass(eq=False)
class GriddedField:
    """A 2-D field on a rectilinear lat/lon grid.

    Attributes are plain numpy arrays so nothing here needs a weather library.
    ``eq`` is disabled because dataclass equality on arrays raises instead of
    comparing; use :meth:`equals` when a tolerance-aware comparison is wanted.
    """

    values: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    name: str = "field"
    units: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.latitude = _check_axis(self.latitude, "latitude")
        self.longitude = _check_axis(self.longitude, "longitude")
        if self.values.ndim != 2:
            raise FieldError(
                f"values: expected 2-D [latitude, longitude], got {self.values.ndim}-D"
            )
        expected = (self.latitude.size, self.longitude.size)
        if self.values.shape != expected:
            raise FieldError(
                f"values shape {self.values.shape} does not match the axes {expected}"
            )
        if not np.isfinite(self.latitude).all() or np.abs(self.latitude).max() > 90:
            raise FieldError("latitude: values must lie within [-90, 90]")
        if np.abs(self.longitude).max() > 180:
            raise FieldError("longitude: values must lie within [-180, 180]")
        if np.isnan(self.values).all():
            raise FieldError("values: field is entirely NaN")

    # -- shape and orientation -------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def latitude_ascending(self) -> bool:
        return bool(self.latitude[1] > self.latitude[0])

    def resolution_deg(self) -> tuple[float, float]:
        """Median grid spacing as ``(dlat, dlon)`` in degrees, always positive."""
        return (
            float(np.median(np.abs(np.diff(self.latitude)))),
            float(np.median(np.abs(np.diff(self.longitude)))),
        )

    def resolution_km(self) -> float:
        """Approximate grid spacing in km, using the mean latitude.

        Uses the spherical approximation of 111.32 km per degree of latitude.
        This is an *estimate for reporting only* — a real distance calculation
        belongs in the geospatial layer with the actual earth model.
        """
        dlat, _ = self.resolution_deg()
        return float(dlat * 111.32)

    def ascending(self) -> GriddedField:
        """Return this field with latitude ascending (flip if needed).

        The metrics compare element-wise, so both operands must share an axis
        direction. Normalising here rather than in each metric is what keeps a
        descending ERA5 slice from silently producing garbage comparisons.
        """
        if self.latitude_ascending:
            return self
        return GriddedField(
            values=np.flip(self.values, axis=0).copy(),
            latitude=self.latitude[::-1].copy(),
            longitude=self.longitude.copy(),
            name=self.name,
            units=self.units,
            attrs=dict(self.attrs),
        )

    # -- summary statistics ----------------------------------------------------

    def finite_values(self) -> np.ndarray:
        """The finite values, as a flat array. Empty when the field is all-NaN."""
        return self.values[np.isfinite(self.values)]

    @property
    def peak(self) -> float | None:
        """Maximum finite value, or ``None`` when there is nothing finite.

        Returning ``None`` rather than ``nan`` matters: ``None`` propagates as
        "not computable" through the metrics, while ``nan`` silently poisons
        arithmetic and can make a broken comparison look like a passing one.
        """
        finite = self.finite_values()
        return float(finite.max()) if finite.size else None

    @property
    def trough(self) -> float | None:
        finite = self.finite_values()
        return float(finite.min()) if finite.size else None

    @property
    def nan_fraction(self) -> float:
        """Fraction of cells that are not finite — 0.0 for a clean field."""
        return float(np.mean(~np.isfinite(self.values)))

    def percentile(self, q: float) -> float | None:
        """Percentile of the finite values, or ``None`` when undefined."""
        finite = self.finite_values()
        return float(np.percentile(finite, q)) if finite.size else None

    def equals(self, other: GriddedField, atol: float = 1e-9) -> bool:
        """True when axes match exactly and values match within ``atol``."""
        return bool(
            self.values.shape == other.values.shape
            and np.allclose(self.latitude, other.latitude, atol=atol)
            and np.allclose(self.longitude, other.longitude, atol=atol)
            and np.allclose(self.values, other.values, atol=atol, equal_nan=True)
        )

    # -- xarray bridges --------------------------------------------------------

    def to_dataarray(self) -> xr.DataArray:
        """Convert to an ``xarray.DataArray`` with ``latitude``/``longitude`` dims."""
        import xarray as xr

        return xr.DataArray(
            self.values,
            dims=("latitude", "longitude"),
            coords={"latitude": self.latitude, "longitude": self.longitude},
            name=self.name,
            attrs={"units": self.units} if self.units else {},
        )

    @classmethod
    def from_dataarray(cls, data: xr.DataArray, name: str | None = None) -> GriddedField:
        """Build a field from an ``xarray.DataArray``.

        Accepts a DataArray with more than two dimensions by taking the first
        index along every extra dimension (e.g. a single time step of a
        ``(time, latitude, longitude)`` cube). That keeps the common "one
        timestep" case a one-liner instead of hand-written ``.isel`` chains,
        while a genuinely ambiguous cube still fails loudly on shape.
        """
        array = data
        extra = [dim for dim in array.dims if dim not in ("latitude", "longitude")]
        if extra:
            array = array.isel({dim: 0 for dim in extra}, drop=True)
        if "latitude" not in array.dims or "longitude" not in array.dims:
            raise FieldError(
                f"{getattr(data, 'name', 'DataArray')}: expected 'latitude' and 'longitude' "
                f"dims, got {tuple(data.dims)}"
            )
        units = array.attrs.get("units")
        return cls(
            values=array.values,
            latitude=np.asarray(array["latitude"].values, dtype=float),
            longitude=np.asarray(array["longitude"].values, dtype=float),
            name=name or array.name or "field",
            units=str(units) if units is not None else None,
        )
