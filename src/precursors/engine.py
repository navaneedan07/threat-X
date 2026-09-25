"""Atmospheric precursor computation engine.

Computes physically grounded precursor signals from ERA5 surface fields.

Dataset variables available (confirmed 2026-09-25):
  u10, v10  (m/s) — 10 m wind components
  t2m       (K)   — 2 m temperature
  msl       (Pa)  — Mean sea level pressure
  tp        (m)   — Total precipitation (accumulated per step)

Grid:  0.25 deg (~27.8 km), 3-hourly timesteps
Regions: Bay of Bengal (Amphan, 2020) / North India (Heatwave, 2022)

Precursor signals computed:
  1. MSLP tendency (hPa / 3h)
     Derivation: msl[t] - msl[t-1], converted to hPa.
     Significance: Rapid falls indicate intensification.

  2. 10-m Wind Divergence (s^-1)
     Derivation: finite-difference horizontal divergence of (u10, v10)
     on a lat/lon grid (metres-corrected).
     Significance: Negative divergence = convergence, feeds deep convection.

  3. 10-m Wind Speed (m/s)
     Derivation: sqrt(u10^2 + v10^2) at centroid.
     Significance: Strong near-surface winds indicate intensification.

  4. Precipitation Rate (mm/3h)
     Derivation: tp (accumulated) per step, converted m->mm.
     Significance: Large precipitation values indicate active convection.

  5. 2-m Temperature Anomaly (K)
     Derivation: t2m at centroid minus temporal mean at centroid.
     Significance: Positive anomaly = anomalous warming precursor (heatwave).

Variables noted as UNAVAILABLE (no pressure-level data):
  - 850 hPa moisture flux convergence
  - 850 hPa relative vorticity
  - Deep-layer bulk wind shear
  - Theta-e gradient
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Radius of Earth (m)
_R_EARTH_M = 6_371_000.0

# Degrees to radians
_DEG2RAD = math.pi / 180.0

# Neighbourhood half-width for spatial averaging around the centroid (degrees)
_CENTROID_RADIUS_DEG = 1.5  # ~167 km neighbourhood


@dataclass
class PrecursorResult:
    """Precursor feature values for a single time step.

    All Optional[float] fields are None when the required variables are absent
    from the dataset. Null means 'data unavailable', not zero.
    """

    timestamp: str  # ISO 8601 UTC

    # --- Computable from surface ERA5 ---
    pressure_tendency_3h_hpa: Optional[float] = None
    """3-hour MSLP tendency in hPa. Negative = falling = intensification signal."""

    wind_divergence_s1: Optional[float] = None
    """10 m horizontal wind divergence (s^-1). Negative = convergence."""

    wind_speed_ms: Optional[float] = None
    """10 m wind speed magnitude at centroid (m/s)."""

    precipitation_rate_mm3h: Optional[float] = None
    """Total precipitation per 3-hour step in mm."""

    t2m_anomaly_k: Optional[float] = None
    """2 m temperature anomaly relative to dataset temporal mean at centroid (K)."""

    # --- Variables UNAVAILABLE from surface-only dataset ---
    moisture_flux_convergence_g_kg_s: Optional[float] = None  # always None
    vorticity_850_s1: Optional[float] = None                  # always None
    theta_e_gradient_k_100km: Optional[float] = None          # always None
    bulk_shear_ms: Optional[float] = None                     # always None

    unavailable_variables: list[str] = field(default_factory=list)
    """Variables that could not be computed due to missing input data."""

    def to_api_dict(self) -> dict:
        """Return API-compatible dict (matches PrecursorSeriesPoint schema)."""
        return {
            "timestamp": self.timestamp,
            # Map to API field names:
            "pressure_tendency_3h_hpa": self.pressure_tendency_3h_hpa,
            "moisture_flux_convergence_g_kg_s": self.moisture_flux_convergence_g_kg_s,
            "vorticity_850_s1": self.vorticity_850_s1,
            "theta_e_gradient_k_100km": self.theta_e_gradient_k_100km,
            # Extended surface fields (additional to API spec):
            "wind_divergence_s1": self.wind_divergence_s1,
            "wind_speed_ms": self.wind_speed_ms,
            "precipitation_rate_mm3h": self.precipitation_rate_mm3h,
            "t2m_anomaly_k": self.t2m_anomaly_k,
        }


def _nearest_index(coord: xr.DataArray, value: float) -> int:
    """Return the index of the coordinate point nearest to value."""
    return int(np.argmin(np.abs(coord.values - value)))


def _extract_neighbourhood_mean(
    da: xr.DataArray,
    lat: float,
    lon: float,
    radius_deg: float = _CENTROID_RADIUS_DEG,
) -> xr.DataArray:
    """Spatially average da within a square neighbourhood around (lat, lon).

    Clips to domain edges if the neighbourhood extends outside the grid.

    Parameters
    ----------
    da : xr.DataArray
        Spatial field with 'latitude' and 'longitude' coordinates.
    lat, lon : float
        Centre of the neighbourhood (degrees).
    radius_deg : float
        Half-width of the averaging box (degrees).

    Returns
    -------
    xr.DataArray
        Result after spatial mean; retains the time dimension (valid_time).
    """
    lats = da["latitude"].values
    lons = da["longitude"].values
    lat_min = max(lat - radius_deg, lats.min())
    lat_max = min(lat + radius_deg, lats.max())
    lon_min = max(lon - radius_deg, lons.min())
    lon_max = min(lon + radius_deg, lons.max())
    cropped = da.sel(
        latitude=slice(lat_max, lat_min),  # ERA5 latitudes are decreasing
        longitude=slice(lon_min, lon_max),
    )
    if cropped.size == 0:
        # Neighbourhood outside the domain — fall back to nearest point.
        lat_idx = _nearest_index(da["latitude"], lat)
        lon_idx = _nearest_index(da["longitude"], lon)
        return da.isel(latitude=lat_idx, longitude=lon_idx)
    return cropped.mean(dim=["latitude", "longitude"])


def _compute_divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    lat_centre: float,
) -> xr.DataArray:
    """Estimate horizontal divergence of a 2-D (lat, lon) wind field.

    Uses centred finite differences.  Grid spacings are converted to metres
    using the local latitude for the zonal component.

    Parameters
    ----------
    u, v : xr.DataArray
        Wind components at one time step; dims (latitude, longitude).
    lat_centre : float
        Latitude of the centroid (degrees) — used to compute dx.

    Returns
    -------
    xr.DataArray
        Divergence field (s^-1).
    """
    lats = u["latitude"].values
    lons = u["longitude"].values

    # Grid spacing in metres
    dlat_deg = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25
    dlon_deg = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.25
    dy_m = dlat_deg * _DEG2RAD * _R_EARTH_M
    dx_m = dlon_deg * _DEG2RAD * _R_EARTH_M * math.cos(lat_centre * _DEG2RAD)

    if dx_m == 0 or dy_m == 0:
        return xr.zeros_like(u)

    # Centred differences (interior points)
    du_dx = xr.DataArray(
        np.gradient(u.values, axis=1) / dx_m,
        dims=u.dims,
        coords=u.coords,
    )
    dv_dy = xr.DataArray(
        np.gradient(v.values, axis=0) / dy_m,
        dims=v.dims,
        coords=v.coords,
    )
    # ERA5 latitudes are ordered N→S (decreasing). np.gradient along axis=0
    # computes d/d(index). Since index increases as lat decreases, we need
    # to negate the meridional derivative.
    if lats[0] > lats[-1]:  # N→S ordering confirmed
        dv_dy = -dv_dy

    return du_dx + dv_dy


def compute_precursors(
    ds: xr.Dataset,
    centroid_lat: float,
    centroid_lon: float,
    time_index: int,
    neighbourhood_deg: float = _CENTROID_RADIUS_DEG,
) -> PrecursorResult:
    """Compute all precursor signals for a single time step.

    Parameters
    ----------
    ds : xr.Dataset
        Merged ERA5 dataset (output of src.data.loader.load_dataset).
    centroid_lat, centroid_lon : float
        Geographic centre of the threat object (degrees).
    time_index : int
        Index into the valid_time dimension.
    neighbourhood_deg : float
        Spatial averaging half-width (degrees).

    Returns
    -------
    PrecursorResult
        Computed precursor features. Missing-variable fields are None.
    """
    unavailable: list[str] = []
    result_kwargs: dict = {}

    times = ds["valid_time"].values
    ts_str = str(np.datetime_as_string(times[time_index], unit="s")) + "Z"
    result_kwargs["timestamp"] = ts_str

    # -----------------------------------------------------------------------
    # 1. MSLP Tendency (3h)
    # -----------------------------------------------------------------------
    if "msl" in ds:
        msl = ds["msl"]
        msl_centre = _extract_neighbourhood_mean(msl, centroid_lat, centroid_lon, neighbourhood_deg)
        if time_index > 0:
            tend_pa = float(msl_centre.isel(valid_time=time_index).values) - float(
                msl_centre.isel(valid_time=time_index - 1).values
            )
            result_kwargs["pressure_tendency_3h_hpa"] = round(tend_pa / 100.0, 3)
        # No tendency for t=0 (no prior step) — leave as None.
    else:
        unavailable.append("msl_tendency")

    # -----------------------------------------------------------------------
    # 2. 10-m Wind Divergence
    # -----------------------------------------------------------------------
    if "u10" in ds and "v10" in ds:
        u = ds["u10"].isel(valid_time=time_index)
        v = ds["v10"].isel(valid_time=time_index)
        try:
            div = _compute_divergence(u, v, centroid_lat)
            div_centre = _extract_neighbourhood_mean(
                div.assign_coords(u.coords), centroid_lat, centroid_lon, neighbourhood_deg
            )
            result_kwargs["wind_divergence_s1"] = round(float(div_centre.values), 8)
        except Exception as exc:
            logger.warning("Wind divergence computation failed: %s", exc)
            unavailable.append("wind_divergence")
    else:
        unavailable.append("wind_divergence")

    # -----------------------------------------------------------------------
    # 3. 10-m Wind Speed
    # -----------------------------------------------------------------------
    if "u10" in ds and "v10" in ds:
        u_centre = _extract_neighbourhood_mean(
            ds["u10"], centroid_lat, centroid_lon, neighbourhood_deg
        )
        v_centre = _extract_neighbourhood_mean(
            ds["v10"], centroid_lat, centroid_lon, neighbourhood_deg
        )
        u_val = float(u_centre.isel(valid_time=time_index).values)
        v_val = float(v_centre.isel(valid_time=time_index).values)
        result_kwargs["wind_speed_ms"] = round(math.sqrt(u_val**2 + v_val**2), 3)
    else:
        unavailable.append("wind_speed")

    # -----------------------------------------------------------------------
    # 4. Precipitation Rate (mm per 3h step)
    # -----------------------------------------------------------------------
    if "tp" in ds:
        tp_centre = _extract_neighbourhood_mean(
            ds["tp"], centroid_lat, centroid_lon, neighbourhood_deg
        )
        tp_val_m = float(tp_centre.isel(valid_time=time_index).values)
        result_kwargs["precipitation_rate_mm3h"] = round(tp_val_m * 1000.0, 3)
    else:
        unavailable.append("precipitation_rate")

    # -----------------------------------------------------------------------
    # 5. 2-m Temperature Anomaly (relative to dataset temporal mean)
    # -----------------------------------------------------------------------
    if "t2m" in ds:
        t2m_centre = _extract_neighbourhood_mean(
            ds["t2m"], centroid_lat, centroid_lon, neighbourhood_deg
        )
        t2m_mean = float(t2m_centre.mean(dim="valid_time").values)
        t2m_now = float(t2m_centre.isel(valid_time=time_index).values)
        result_kwargs["t2m_anomaly_k"] = round(t2m_now - t2m_mean, 3)
    else:
        unavailable.append("t2m_anomaly")

    # -----------------------------------------------------------------------
    # Mark unavailable pressure-level variables
    # -----------------------------------------------------------------------
    unavailable.extend([
        "moisture_flux_convergence_g_kg_s (no 850 hPa q/wind data)",
        "vorticity_850_s1 (no 850 hPa wind data)",
        "theta_e_gradient_k_100km (no upper-air T/q data)",
        "bulk_shear_ms (no multi-level wind data)",
    ])

    return PrecursorResult(unavailable_variables=unavailable, **result_kwargs)


def compute_precursor_series(
    ds: xr.Dataset,
    centroid_lat: float,
    centroid_lon: float,
    window_hours: int = 24,
    neighbourhood_deg: float = _CENTROID_RADIUS_DEG,
) -> list[PrecursorResult]:
    """Compute precursor signals for all time steps within window_hours.

    Parameters
    ----------
    ds : xr.Dataset
        Merged ERA5 dataset.
    centroid_lat, centroid_lon : float
        Threat centroid.
    window_hours : int
        Number of hours back from the last timestep to include.
    neighbourhood_deg : float
        Spatial averaging radius.

    Returns
    -------
    list[PrecursorResult]
        One entry per qualifying time step, in chronological order.
    """
    times = ds["valid_time"].values
    last_t = times[-1]
    cutoff = last_t - np.timedelta64(int(window_hours * 3600), "s")

    results: list[PrecursorResult] = []
    for i, t in enumerate(times):
        if t < cutoff:
            continue
        try:
            pr = compute_precursors(ds, centroid_lat, centroid_lon, i, neighbourhood_deg)
            results.append(pr)
        except Exception as exc:
            logger.error("Failed at time_index=%d (%s): %s", i, t, exc)

    return results


def generate_explanatory_summary(series: list[PrecursorResult]) -> str:
    """Generate a deterministic rule-based text summary of precursor trends.

    This is a rule-based description derived from the actual computed values.
    It is NOT a model prediction.

    Parameters
    ----------
    series : list[PrecursorResult]
        Chronologically ordered precursor series (output of compute_precursor_series).

    Returns
    -------
    str
        Human-readable summary of the dominant atmospheric driver.
    """
    if not series:
        return "No precursor data available."

    # Collect available time steps (skip those with None pressure tendency)
    valid_tend = [r.pressure_tendency_3h_hpa for r in series if r.pressure_tendency_3h_hpa is not None]
    valid_div = [r.wind_divergence_s1 for r in series if r.wind_divergence_s1 is not None]
    valid_precip = [r.precipitation_rate_mm3h for r in series if r.precipitation_rate_mm3h is not None]
    valid_t2m = [r.t2m_anomaly_k for r in series if r.t2m_anomaly_k is not None]

    drivers: list[str] = []

    # Pressure tendency
    if valid_tend:
        min_tend = min(valid_tend)
        max_tend = max(valid_tend)
        last_tend = valid_tend[-1]
        if last_tend < -3.0:
            drivers.append(
                f"rapid 3-hour pressure falls (most recent: {last_tend:+.1f} hPa/3h)"
            )
        elif last_tend < -1.5:
            drivers.append(
                f"steady pressure falls (most recent: {last_tend:+.1f} hPa/3h)"
            )
        elif last_tend > 1.5:
            drivers.append(
                f"pressure rises (most recent: {last_tend:+.1f} hPa/3h), suggesting weakening"
            )
        else:
            drivers.append(f"near-steady pressure (most recent: {last_tend:+.1f} hPa/3h)")

    # Wind convergence
    if valid_div:
        last_div = valid_div[-1]
        if last_div < -5e-5:
            drivers.append(
                f"strong low-level wind convergence (div: {last_div:.2e} s⁻¹)"
            )
        elif last_div < -1e-5:
            drivers.append(
                f"moderate low-level convergence (div: {last_div:.2e} s⁻¹)"
            )

    # Precipitation
    if valid_precip:
        max_precip = max(valid_precip)
        last_precip = valid_precip[-1]
        if last_precip > 10.0:
            drivers.append(
                f"intense precipitation ({last_precip:.1f} mm/3h at centroid)"
            )
        elif last_precip > 2.0:
            drivers.append(
                f"moderate precipitation ({last_precip:.1f} mm/3h at centroid)"
            )

    # Temperature anomaly
    if valid_t2m:
        last_t2m = valid_t2m[-1]
        if abs(last_t2m) > 2.0:
            sign = "positive" if last_t2m > 0 else "negative"
            drivers.append(
                f"{sign} surface temperature anomaly ({last_t2m:+.1f} K)"
            )

    if not drivers:
        return "No significant precursor signal detected in the analysed window."

    if len(drivers) == 1:
        return f"Evolution driven by {drivers[0]}."
    else:
        main = drivers[0]
        rest = "; ".join(drivers[1:])
        return f"Evolution driven by {main}, with supporting signals: {rest}."
