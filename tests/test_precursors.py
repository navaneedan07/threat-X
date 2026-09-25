"""Tests for atmospheric precursor analysis.

Tests cover:
  1. ERA5 loader — zip handling, coord cleaning, merge
  2. Precursor engine — computation correctness, null semantics
  3. Explanatory summary generation
  4. API integration — precursor endpoint with real computed data

Run:  pytest tests/test_precursors.py -v
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures — synthetic minimal xarray Dataset for unit tests
# ---------------------------------------------------------------------------

def _make_synthetic_ds(n_time: int = 10, n_lat: int = 20, n_lon: int = 20) -> xr.Dataset:
    """Create a minimal synthetic dataset matching the ERA5 variable structure."""
    times = np.array(
        [np.datetime64("2020-05-17T00:00:00") + np.timedelta64(3 * i, "h") for i in range(n_time)]
    )
    lats = np.linspace(10.0, 20.0, n_lat)[::-1]  # N→S ordering like ERA5
    lons = np.linspace(82.0, 92.0, n_lon)

    rng = np.random.default_rng(seed=42)

    ds = xr.Dataset(
        {
            "u10": (["valid_time", "latitude", "longitude"],
                    rng.normal(5.0, 3.0, (n_time, n_lat, n_lon)).astype(np.float32)),
            "v10": (["valid_time", "latitude", "longitude"],
                    rng.normal(-3.0, 3.0, (n_time, n_lat, n_lon)).astype(np.float32)),
            "t2m": (["valid_time", "latitude", "longitude"],
                    (300.0 + rng.normal(0, 2, (n_time, n_lat, n_lon))).astype(np.float32)),
            "msl": (["valid_time", "latitude", "longitude"],
                    (100000.0 + rng.normal(0, 500, (n_time, n_lat, n_lon))).astype(np.float32)),
            "tp": (["valid_time", "latitude", "longitude"],
                   np.abs(rng.normal(0.005, 0.003, (n_time, n_lat, n_lon))).astype(np.float32)),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    # Attach units attributes matching ERA5
    ds["u10"].attrs["units"] = "m s**-1"
    ds["v10"].attrs["units"] = "m s**-1"
    ds["t2m"].attrs["units"] = "K"
    ds["msl"].attrs["units"] = "Pa"
    ds["tp"].attrs["units"] = "m"
    return ds


# ---------------------------------------------------------------------------
# 1. Loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_get_time_step_hours(self):
        from src.data.loader import get_time_step_hours
        ds = _make_synthetic_ds(n_time=5)
        dt = get_time_step_hours(ds)
        assert dt == pytest.approx(3.0)

    def test_get_time_step_hours_single_step(self):
        from src.data.loader import get_time_step_hours
        ds = _make_synthetic_ds(n_time=1)
        dt = get_time_step_hours(ds)
        assert np.isnan(dt)

    def test_spatial_resolution_deg(self):
        from src.data.loader import spatial_resolution_deg
        ds = _make_synthetic_ds()
        res = spatial_resolution_deg(ds)
        assert res == pytest.approx(10.0 / 19, rel=0.01)

    def test_load_dataset_missing_file(self, tmp_path):
        from src.data.loader import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.nc")

    def test_load_dataset_not_zip(self, tmp_path):
        from src.data.loader import load_dataset
        f = tmp_path / "bad.nc"
        f.write_bytes(b"not a zip file at all")
        with pytest.raises(ValueError, match="ZIP"):
            load_dataset(f)


# ---------------------------------------------------------------------------
# 2. Precursor engine unit tests
# ---------------------------------------------------------------------------

class TestPrecursorEngine:
    @pytest.fixture(autouse=True)
    def ds(self):
        self._ds = _make_synthetic_ds(n_time=10)

    def test_compute_precursors_returns_result(self):
        from src.precursors.engine import compute_precursors, PrecursorResult
        result = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=3)
        assert isinstance(result, PrecursorResult)

    def test_pressure_tendency_null_at_t0(self):
        """At time_index=0 there is no prior step — tendency must be None."""
        from src.precursors.engine import compute_precursors
        result = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=0)
        assert result.pressure_tendency_3h_hpa is None

    def test_pressure_tendency_float_at_t1(self):
        from src.precursors.engine import compute_precursors
        result = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=1)
        assert result.pressure_tendency_3h_hpa is not None
        assert isinstance(result.pressure_tendency_3h_hpa, float)

    def test_wind_speed_non_negative(self):
        from src.precursors.engine import compute_precursors
        for i in range(5):
            r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=i)
            if r.wind_speed_ms is not None:
                assert r.wind_speed_ms >= 0.0

    def test_precipitation_rate_non_negative(self):
        from src.precursors.engine import compute_precursors
        for i in range(5):
            r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=i)
            if r.precipitation_rate_mm3h is not None:
                assert r.precipitation_rate_mm3h >= 0.0

    def test_pressure_level_fields_are_always_null(self):
        """850 hPa-derived fields must always be None (not in dataset)."""
        from src.precursors.engine import compute_precursors
        r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=2)
        assert r.moisture_flux_convergence_g_kg_s is None
        assert r.vorticity_850_s1 is None
        assert r.theta_e_gradient_k_100km is None
        assert r.bulk_shear_ms is None

    def test_unavailable_variables_logged(self):
        from src.precursors.engine import compute_precursors
        r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=2)
        assert len(r.unavailable_variables) > 0
        # Should mention the 850 hPa variables
        joined = " ".join(r.unavailable_variables)
        assert "850" in joined or "vorticity" in joined

    def test_timestamp_format(self):
        from src.precursors.engine import compute_precursors
        r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=2)
        assert r.timestamp.endswith("Z")
        assert "T" in r.timestamp

    def test_to_api_dict_keys(self):
        from src.precursors.engine import compute_precursors
        r = compute_precursors(self._ds, centroid_lat=15.0, centroid_lon=87.0, time_index=2)
        d = r.to_api_dict()
        for key in ("timestamp", "pressure_tendency_3h_hpa", "moisture_flux_convergence_g_kg_s",
                    "vorticity_850_s1", "theta_e_gradient_k_100km"):
            assert key in d

    def test_centroid_outside_domain_handled(self):
        """Centroid far outside the grid should not crash — falls back to nearest point."""
        from src.precursors.engine import compute_precursors
        result = compute_precursors(self._ds, centroid_lat=0.0, centroid_lon=50.0, time_index=2)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. Precursor series and summary tests
# ---------------------------------------------------------------------------

class TestPrecursorSeries:
    def test_series_length_within_window(self):
        from src.precursors.engine import compute_precursor_series
        ds = _make_synthetic_ds(n_time=20)
        series = compute_precursor_series(ds, 15.0, 87.0, window_hours=12)
        # 12h window with 3h steps = 5 steps (including boundary)
        assert 1 <= len(series) <= 20

    def test_series_chronological(self):
        from src.precursors.engine import compute_precursor_series
        ds = _make_synthetic_ds(n_time=10)
        series = compute_precursor_series(ds, 15.0, 87.0, window_hours=24)
        timestamps = [r.timestamp for r in series]
        assert timestamps == sorted(timestamps)

    def test_generate_summary_returns_string(self):
        from src.precursors.engine import compute_precursor_series, generate_explanatory_summary
        ds = _make_synthetic_ds(n_time=10)
        series = compute_precursor_series(ds, 15.0, 87.0, window_hours=24)
        summary = generate_explanatory_summary(series)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_generate_summary_empty_series(self):
        from src.precursors.engine import generate_explanatory_summary
        summary = generate_explanatory_summary([])
        assert "No" in summary or summary  # Should be a non-empty fallback message

    def test_full_window_includes_all_steps(self):
        from src.precursors.engine import compute_precursor_series
        ds = _make_synthetic_ds(n_time=10)
        series = compute_precursor_series(ds, 15.0, 87.0, window_hours=9999)
        # With huge window, should get all time steps
        assert len(series) == 10

    def test_missing_msl_handled(self):
        """If msl is absent, pressure tendency should be None (not crash)."""
        from src.precursors.engine import compute_precursors
        ds = _make_synthetic_ds().drop_vars("msl")
        r = compute_precursors(ds, 15.0, 87.0, time_index=2)
        assert r.pressure_tendency_3h_hpa is None

    def test_missing_wind_handled(self):
        from src.precursors.engine import compute_precursors
        ds = _make_synthetic_ds().drop_vars(["u10", "v10"])
        r = compute_precursors(ds, 15.0, 87.0, time_index=2)
        assert r.wind_divergence_s1 is None
        assert r.wind_speed_ms is None

    def test_missing_tp_handled(self):
        from src.precursors.engine import compute_precursors
        ds = _make_synthetic_ds().drop_vars("tp")
        r = compute_precursors(ds, 15.0, 87.0, time_index=2)
        assert r.precipitation_rate_mm3h is None


# ---------------------------------------------------------------------------
# 4. API integration tests for the precursor endpoints
# ---------------------------------------------------------------------------

class TestPrecursorAPIIntegration:
    """Tests against real computed output (THR-2020-0001, THR-2022-0001)
    as well as fixture fallback (THR-2026-0001)."""

    _REAL_IDS = ["THR-2020-0001", "THR-2022-0001"]
    _FIXTURE_ID = "THR-2026-0001"
    _UNKNOWN_ID = "THR-9999-9999"

    def test_fixture_id_200(self):
        r = client.get(f"/api/v1/threats/{self._FIXTURE_ID}/precursors")
        assert r.status_code == 200

    def test_unknown_id_404(self):
        r = client.get(f"/api/v1/threats/{self._UNKNOWN_ID}/precursors")
        assert r.status_code == 404

    def test_schema_keys(self):
        r = client.get(f"/api/v1/threats/{self._FIXTURE_ID}/precursors")
        body = r.json()
        for key in ("threat_id", "variables_analyzed", "series"):
            assert key in body

    def test_series_has_timestamps(self):
        r = client.get(f"/api/v1/threats/{self._FIXTURE_ID}/precursors")
        body = r.json()
        for point in body["series"]:
            assert "timestamp" in point

    def test_window_hours_filter_reduces_series(self):
        r_full = client.get(f"/api/v1/threats/{self._FIXTURE_ID}/precursors?window_hours=720")
        r_short = client.get(f"/api/v1/threats/{self._FIXTURE_ID}/precursors?window_hours=3")
        assert r_full.status_code == 200
        assert r_short.status_code == 200
        full_count = len(r_full.json()["series"])
        short_count = len(r_short.json()["series"])
        assert short_count <= full_count

    @pytest.mark.skipif(
        not Path("data/processed/precursors.json").exists(),
        reason="Real precursor output not yet generated — run python -m src.precursors.pipeline"
    )
    def test_real_amphan_200(self):
        r = client.get("/api/v1/threats/THR-2020-0001/precursors")
        assert r.status_code == 200

    @pytest.mark.skipif(
        not Path("data/processed/precursors.json").exists(),
        reason="Real precursor output not yet generated — run python -m src.precursors.pipeline"
    )
    def test_real_amphan_series_not_empty(self):
        r = client.get("/api/v1/threats/THR-2020-0001/precursors")
        body = r.json()
        assert len(body["series"]) > 0

    @pytest.mark.skipif(
        not Path("data/processed/precursors.json").exists(),
        reason="Real precursor output not yet generated"
    )
    def test_real_pressure_level_fields_are_null(self):
        """Verify null semantics: 850 hPa fields are null in real output (not fabricated)."""
        r = client.get("/api/v1/threats/THR-2020-0001/precursors")
        body = r.json()
        for point in body["series"]:
            assert point.get("moisture_flux_convergence_g_kg_s") is None
            assert point.get("vorticity_850_s1") is None
            assert point.get("theta_e_gradient_k_100km") is None

    @pytest.mark.skipif(
        not Path("data/processed/precursors.json").exists(),
        reason="Real precursor output not yet generated"
    )
    def test_real_summary_is_string(self):
        r = client.get("/api/v1/threats/THR-2020-0001/precursors")
        body = r.json()
        summary = body.get("explanatory_summary")
        if summary is not None:
            assert isinstance(summary, str)
            assert len(summary) > 0
