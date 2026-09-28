"""Tests for the deterministic downscaling entry point.

Run:  pytest tests/test_downscaling_module.py -v

The regression this file guards: the module used to hard-code a 2.4 zoom factor
and add ``np.random.normal`` noise. Both are gone, so **determinism is asserted
directly** — two identical calls must produce identical fields — and the crop is
tested as the only thing this module adds over ``baseline.py``.
"""

from __future__ import annotations

import pytest

from src.downscaling import downscaling as DD
from src.shared.fields import FieldError, GriddedField
from src.shared.synthetic import sharp_peak_field


@pytest.fixture(scope="module")
def coarse() -> GriddedField:
    return sharp_peak_field(lat_range=(8.0, 18.0), lon_range=(75.0, 85.0), resolution_deg=0.1)


class TestLocalCrop:
    def test_crop_selects_the_requested_window(self, coarse: GriddedField):
        cropped = DD.local_crop(
            coarse, latitude_range=(10.0, 14.0), longitude_range=(78.0, 82.0)
        )
        assert cropped.latitude.min() == pytest.approx(10.0)
        assert cropped.latitude.max() == pytest.approx(14.0)
        assert cropped.longitude.min() == pytest.approx(78.0)
        assert cropped.longitude.max() == pytest.approx(82.0)
        assert cropped.values.shape == (cropped.latitude.size, cropped.longitude.size)

    def test_crop_preserves_the_spacing(self, coarse: GriddedField):
        cropped = DD.local_crop(coarse, latitude_range=(10.0, 12.0))
        assert cropped.resolution_deg() == pytest.approx(coarse.resolution_deg())

    def test_no_bounds_is_refused(self, coarse: GriddedField):
        with pytest.raises(ValueError, match="latitude_range"):
            DD.local_crop(coarse)

    def test_a_window_too_small_is_refused(self, coarse: GriddedField):
        with pytest.raises(FieldError, match="at least 2"):
            DD.local_crop(coarse, latitude_range=(10.0, 10.05))

    def test_descending_latitude_is_supported(self, coarse: GriddedField):
        flipped = GriddedField(
            values=coarse.values[::-1].copy(),
            latitude=coarse.latitude[::-1].copy(),
            longitude=coarse.longitude.copy(),
        )
        cropped = DD.local_crop(flipped, latitude_range=(10.0, 14.0))
        assert cropped.latitude.min() == pytest.approx(10.0)
        assert cropped.latitude.max() == pytest.approx(14.0)


class TestDownscaleBox:
    def test_box_is_cropped_then_refined(self, coarse: GriddedField):
        refined = DD.downscale_box(
            coarse,
            latitude_range=(10.0, 14.0),
            longitude_range=(78.0, 82.0),
            target_resolution_deg=0.05,
        )
        assert refined.resolution_deg() == pytest.approx((0.05, 0.05))
        # The refined grid must not reach past the crop it was built from.
        assert refined.latitude.min() >= 10.0
        assert refined.latitude.max() <= 14.0
        assert refined.attrs["upscale_factor"][0] == pytest.approx(2.0)

    def test_output_is_deterministic(self, coarse: GriddedField):
        """The old implementation added random noise; two runs must now agree."""
        first = DD.downscale_box(
            coarse, latitude_range=(10.0, 14.0), target_resolution_deg=0.05
        )
        second = DD.downscale_box(
            coarse, latitude_range=(10.0, 14.0), target_resolution_deg=0.05
        )
        assert first.equals(second)

    def test_output_is_marked_untrained(self, coarse: GriddedField):
        refined = DD.downscale_box(
            coarse, latitude_range=(10.0, 14.0), target_resolution_deg=0.05
        )
        assert refined.attrs["trained"] is False
        assert refined.attrs["method"].startswith("interpolation:")

    def test_upscale_factor_is_derived_not_hard_coded(self, coarse: GriddedField):
        """A 4x target must report 4x — nothing here multiplies by a fixed 2.4."""
        refined = DD.downscale_box(
            coarse, latitude_range=(10.0, 14.0), target_resolution_deg=0.025
        )
        assert refined.attrs["upscale_factor"][0] == pytest.approx(4.0)


class TestDelegation:
    def test_downscale_field_matches_the_baseline(self, coarse: GriddedField):
        from src.downscaling.baseline import downscale

        via_module = DD.downscale_field(coarse, target_resolution_deg=0.05)
        via_baseline = downscale(coarse, target_resolution_deg=0.05)
        assert via_module.equals(via_baseline)

    def test_module_delegates_to_the_baseline(self):
        """The module is an adapter: it must not define its own resample logic."""
        from src.downscaling import baseline

        assert DD.resample is baseline.resample
        assert DD.downscale is baseline.downscale
        # Guard against a hard-coded factor or the old zoom/noise constants
        # creeping back in at module scope.
        for name in ("ZOOM_FACTOR", "SOURCE_RES_KM", "TARGET_RES_KM"):
            assert not hasattr(DD, name), f"stale constant {name} reintroduced"
