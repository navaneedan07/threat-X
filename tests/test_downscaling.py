"""Tests for the field container, synthetic generators, baseline and metrics.

Run:  pytest tests/test_downscaling.py -v

The load-bearing test here is
``TestExtremePreservation::test_smoothed_field_is_flagged``. The work card's exit
condition is that a metric which passes a smoothed field is wrong, so the metric
is proven against a field that obviously lost its extreme.

Its complement matters just as much: a metric that fails *everything* is equally
useless. ``test_modest_coarsening_is_not_flagged`` and
``test_identical_field_is_not_flagged`` assert the opposite direction, so the
suite proves the metric **discriminates** rather than merely being strict.

Everything here runs on numpy and scipy only — no xarray, no weather data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.downscaling import metrics as M
from src.downscaling.baseline import (
    downscale,
    downscale_to_reference,
    method_name,
    resample,
    resolve_order,
    target_grid,
    upscale_factor,
)
from src.shared.config import ConfigError
from src.shared.fields import FieldError, GriddedField
from src.shared.metrics import MetricTable
from src.shared.synthetic import (
    gaussian_bump,
    grid,
    random_field,
    sharp_peak_field,
    smoothed_field,
    translating_event,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference() -> GriddedField:
    """The fine-resolution 'truth' with a sharp, flat-topped extreme."""
    return sharp_peak_field()


@pytest.fixture(scope="module")
def coarse(reference: GriddedField) -> GriddedField:
    """A 10x coarser observation of the reference (0.5 deg source grid)."""
    return resample(reference, reference.latitude[::10], reference.longitude[::10], order=1)


# ---------------------------------------------------------------------------
# GriddedField
# ---------------------------------------------------------------------------


class TestGriddedField:
    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(FieldError, match="does not match the axes"):
            GriddedField(np.zeros((5, 4)), np.arange(5.0), np.arange(6.0))

    def test_non_monotonic_axis_is_rejected(self):
        with pytest.raises(FieldError, match="monotonic"):
            GriddedField(np.zeros((3, 3)), np.array([1.0, 3.0, 2.0]), np.arange(3.0))

    def test_out_of_range_latitude_is_rejected(self):
        with pytest.raises(FieldError, match=r"\[-90, 90\]"):
            GriddedField(np.zeros((3, 3)), np.array([95.0, 96.0, 97.0]), np.arange(3.0))

    def test_all_nan_field_is_rejected(self):
        with pytest.raises(FieldError, match="entirely NaN"):
            GriddedField(np.full((3, 3), np.nan), np.arange(3.0), np.arange(3.0))

    def test_descending_latitude_is_accepted(self, reference: GriddedField):
        flipped = GriddedField(
            reference.values[::-1].copy(),
            reference.latitude[::-1].copy(),
            reference.longitude.copy(),
        )
        assert flipped.latitude_ascending is False
        assert flipped.ascending().equals(reference)

    def test_resolution_is_measured(self):
        latitude, longitude = grid((0.0, 2.0), (0.0, 4.0), 0.5)
        field = gaussian_bump(
            latitude, longitude, centre=(1.0, 2.0), amplitude=1.0, radius_deg=0.4
        )
        assert field.resolution_deg() == pytest.approx((0.5, 0.5))
        assert field.resolution_km() == pytest.approx(0.5 * 111.32)

    def test_peak_is_none_when_nothing_is_finite(self):
        field = GriddedField(
            np.array([[np.nan, np.nan], [np.nan, 5.0]]), np.arange(2.0), np.arange(2.0)
        )
        assert field.peak == 5.0
        assert field.nan_fraction == pytest.approx(0.75)

    def test_equal_axes_from_2x2_grid(self, reference: GriddedField):
        small = resample(reference, reference.latitude[::50], reference.longitude[::50])
        assert small.shape == (5, 5)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


class TestSynthetic:
    def test_sharp_peak_has_the_documented_amplitude(self, reference: GriddedField):
        assert reference.peak == pytest.approx(84.2)
        assert reference.units == "mm/24h"
        assert reference.attrs["synthetic"] is True

    def test_random_field_is_reproducible(self):
        axes = grid((8.0, 10.0), (76.0, 78.0), 0.5)
        first = random_field(*axes, seed=7)
        second = random_field(*axes, seed=7)
        third = random_field(*axes, seed=8)
        assert first.equals(second)
        assert not first.equals(third)

    def test_smoothed_field_loses_the_peak(self, reference: GriddedField):
        assert smoothed_field(reference).peak < reference.peak

    def test_translating_event_peaks_where_documented(self):
        events = translating_event()
        amplitudes = [event.amplitude for event in events]
        assert amplitudes.index(max(amplitudes)) == 3
        # Rises into the peak, falls away from it. The two ramps are different
        # lengths (3 intervals up, 2 down) so it is deliberately not symmetric.
        assert amplitudes[:4] == sorted(amplitudes[:4])
        assert amplitudes[3:] == sorted(amplitudes[3:], reverse=True)
        # Both ends sit at the start amplitude.
        assert amplitudes[0] == amplitudes[-1] == 24.0

    def test_translating_event_centroid_moves_monotonically(self):
        events = translating_event()
        latitudes = [event.centre[0] for event in events]
        longitudes = [event.centre[1] for event in events]
        assert latitudes == sorted(latitudes)
        assert longitudes == sorted(longitudes)

    def test_translating_event_stays_inside_its_grid(self):
        for event in translating_event():
            assert event.field.peak == pytest.approx(event.amplitude, rel=1e-6)

    def test_translating_event_rejects_a_degenerate_peak_step(self):
        with pytest.raises(ValueError, match="peak_step"):
            translating_event(steps=4, peak_step=3)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class TestUpscaleFactor:
    def test_problem_statement_ratio_is_not_integer(self):
        """12 km -> 5 km is 2.4, which is exactly why the factor is derived."""
        assert upscale_factor(12.0, 5.0) == pytest.approx(2.4)

    def test_factor_follows_the_grids_supplied(self):
        assert upscale_factor(0.25, 0.05) == pytest.approx(5.0)
        assert upscale_factor(1.0, 0.1) == pytest.approx(10.0)

    def test_non_positive_resolution_is_refused(self):
        with pytest.raises(ValueError):
            upscale_factor(0.0, 0.05)


class TestResampling:
    def test_bilinear_reproduces_a_linear_ramp(self):
        latitude, longitude = grid((0.0, 4.0), (0.0, 4.0), 1.0)
        ramp = np.add.outer(latitude, longitude)
        source = GriddedField(ramp, latitude, longitude, name="ramp")
        fine_latitude, fine_longitude = grid((0.0, 4.0), (0.0, 4.0), 0.25)
        result = resample(source, fine_latitude, fine_longitude, order=1)
        expected = np.add.outer(fine_latitude, fine_longitude)
        assert np.allclose(result.values, expected, atol=1e-9)

    def test_nearest_is_piecewise_constant(self):
        latitude, longitude = grid((0.0, 3.0), (0.0, 3.0), 1.0)
        source = GriddedField(
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            latitude,
            longitude,
        )
        fine = np.arange(0.0, 3.01, 0.5)
        result = resample(source, fine, fine, order=0)
        assert set(np.unique(result.values)) <= {0.0, 1.0}

    def test_target_grid_is_a_parameter_not_a_factor(self, reference: GriddedField):
        """Changing the target spacing changes the output shape accordingly."""
        one_degree = resample(
            reference, reference.latitude[::20], reference.longitude[::20], order=1
        )
        half_degree = resample(
            reference, reference.latitude[::10], reference.longitude[::10], order=1
        )
        assert half_degree.shape[0] > one_degree.shape[0]
        assert half_degree.attrs["upscale_factor"][0] != one_degree.attrs["upscale_factor"][0]

    def test_resampling_is_idempotent_on_the_same_grid(self, reference: GriddedField):
        same = resample(reference, reference.latitude, reference.longitude, order=1)
        assert np.allclose(same.values, reference.values, atol=1e-9)

    def test_out_of_domain_target_is_nan_with_reduced_coverage(self, reference: GriddedField):
        beyond = np.arange(8.0, 25.0, 0.5)
        result = resample(reference, beyond, reference.longitude, order=1)
        assert result.nan_fraction > 0
        assert result.attrs["coverage_fraction"] < 1.0

    def test_fully_covering_target_reports_full_coverage(self, coarse, reference):
        refined = downscale_to_reference(coarse, reference)
        assert refined.attrs["coverage_fraction"] == pytest.approx(1.0)

    def test_descending_source_latitude_gives_the_same_answer(self, reference: GriddedField):
        """ERA5 ships latitude descending; the result must not depend on that."""
        target_lat = reference.latitude[::10]
        target_lon = reference.longitude[::10]
        from_ascending = resample(reference, target_lat, target_lon, order=1)
        flipped = GriddedField(
            reference.values[::-1].copy(),
            reference.latitude[::-1].copy(),
            reference.longitude.copy(),
        )
        from_descending = resample(flipped, target_lat, target_lon, order=1)
        assert from_ascending.equals(from_descending)

    def test_provenance_records_method_and_that_it_is_untrained(self, coarse, reference):
        refined = downscale_to_reference(coarse, reference)
        assert refined.attrs["trained"] is False
        assert refined.attrs["method"] == "interpolation:linear"
        assert refined.attrs["configured_method"] == "interpolation"


class TestTargetGrid:
    def test_grid_is_snapped_to_the_resolution(self, reference: GriddedField):
        latitude, longitude = target_grid(reference, 0.5)
        assert np.allclose(np.diff(latitude), 0.5)
        assert np.allclose(latitude / 0.5, np.round(latitude / 0.5))

    def test_too_coarse_a_target_is_refused(self, reference: GriddedField):
        with pytest.raises(ValueError, match="fewer than 2 points"):
            target_grid(reference, 50.0)


class TestMethodSelection:
    def test_order_comes_from_config(self):
        """configs/model.yaml ships downscaling.interpolation.order: 1."""
        assert resolve_order() == 1
        assert method_name() == "linear"

    def test_explicit_order_overrides_config(self):
        assert resolve_order(0) == 0
        assert method_name(0) == "nearest"

    def test_unimplemented_order_raises_rather_than_substituting(self):
        with pytest.raises(ValueError, match="not implemented"):
            resolve_order(3)

    def test_unresolved_target_resolution_names_the_config_key(self):
        """An explicit null resolution is still refused, whatever the shipped config says.

        ``configs/data.yaml`` now carries the confirmed 0.10 deg target, so the null
        case is exercised with an explicit config to keep the guard alive.
        """
        source = sharp_peak_field()
        with pytest.raises(ConfigError, match="fine_resolution_deg"):
            downscale(source, data_config={"grid": {"fine_resolution_deg": None}})

    def test_shipped_config_resolves_the_confirmed_fine_resolution(self):
        """The confirmed pair (0.25 -> 0.10) makes the derived path work with no argument."""
        source = sharp_peak_field(
            lat_range=(13.0, 14.0), lon_range=(80.0, 81.0), resolution_deg=0.5
        )
        result = downscale(source)
        assert result.resolution_deg() == pytest.approx((0.10, 0.10))

    def test_downscale_accepts_an_explicit_resolution(self):
        source = sharp_peak_field(
            lat_range=(13.0, 14.0), lon_range=(80.0, 81.0), resolution_deg=0.2
        )
        result = downscale(source, target_resolution_deg=0.1)
        assert result.resolution_deg() == pytest.approx((0.1, 0.1))

    def test_supplying_only_one_target_axis_is_refused(self, reference: GriddedField):
        with pytest.raises(ValueError, match="both target_latitude"):
            downscale(reference, target_latitude=reference.latitude)


# ---------------------------------------------------------------------------
# Extreme preservation — the exit condition
# ---------------------------------------------------------------------------


class TestExtremePreservation:
    def test_smoothed_field_is_flagged(self, reference: GriddedField):
        """A field that obviously lost its extreme MUST be flagged.

        If this test ever passes while ``peak_preservation`` is high, the metric is
        wrong, not the field.
        """
        table = M.evaluate_downscaling(smoothed_field(reference), reference)
        assert table["peak_preservation"] is not None
        assert table["peak_preservation"] < 0.8
        assert table["extreme_bias"] < 0
        assert table.flags_extreme_loss(0.95) is True

    def test_identical_field_is_not_flagged(self, reference: GriddedField):
        """The metric must not fail everything — a perfect field scores perfectly."""
        table = M.evaluate_downscaling(reference, reference)
        assert table["peak_preservation"] == pytest.approx(1.0)
        assert table["rmse"] == pytest.approx(0.0)
        assert table["iou"] == pytest.approx(1.0)
        assert table.flags_extreme_loss(0.95) is False

    def test_modest_coarsening_is_not_flagged(self, reference: GriddedField):
        """2x coarsening retains the peak; the metric must say so."""
        coarse = resample(reference, reference.latitude[::2], reference.longitude[::2], order=1)
        table = M.evaluate_downscaling(downscale_to_reference(coarse, reference), reference)
        assert table["peak_preservation"] > 0.95
        assert table.flags_extreme_loss(0.95) is False

    def test_heavy_coarsening_is_flagged(self, coarse, reference):
        """10x coarsening genuinely cannot resolve a 0.18 deg peak.

        Both the interpolation baseline and a learned model are expected to lose
        here — the point is that the metric reports the loss instead of hiding it.
        """
        table = M.evaluate_downscaling(downscale_to_reference(coarse, reference), reference)
        assert table["peak_preservation"] < 0.8
        assert table.flags_extreme_loss(0.95) is True

    def test_unmeasurable_peak_counts_as_a_failure(self):
        """A metric that passes because it never ran is the failure mode guarded against."""
        table = MetricTable(metrics={"peak_preservation": None})
        assert table.flags_extreme_loss(0.9) is True


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_peak_preservation_is_none_against_a_zero_peak(self):
        flat = GriddedField(np.zeros((4, 4)), np.arange(4.0), np.arange(4.0))
        assert M.peak_preservation(flat, flat) is None

    def test_rmse_and_mae_are_zero_for_identical_fields(self, reference: GriddedField):
        assert M.rmse(reference, reference) == pytest.approx(0.0)
        assert M.mae(reference, reference) == pytest.approx(0.0)

    def test_rmse_on_a_known_offset(self):
        latitude = np.arange(3.0)
        reference = GriddedField(np.zeros((3, 3)), latitude, latitude)
        predicted = GriddedField(np.full((3, 3), 3.0), latitude, latitude)
        assert M.rmse(predicted, reference) == pytest.approx(3.0)
        assert M.mae(predicted, reference) == pytest.approx(3.0)

    def test_threshold_exceedance_fraction(self):
        latitude = np.arange(2.0)
        field = GriddedField(
            np.array([[1.0, 2.0], [3.0, 4.0]]), latitude, latitude
        )
        assert M.threshold_exceedance_fraction(field, 2.5) == pytest.approx(0.5)

    def test_iou_and_dice_on_perfect_overlap(self, reference: GriddedField):
        threshold = reference.percentile(99)
        assert M.iou(reference, reference, threshold) == pytest.approx(1.0)
        assert M.dice(reference, reference, threshold) == pytest.approx(1.0)

    def test_iou_is_none_when_both_footprints_are_empty(self, reference: GriddedField):
        assert M.iou(reference, reference, 1e6) is None

    def test_percentile_error_is_reported_per_percentile(self, reference: GriddedField):
        result = M.percentile_error(smoothed_field(reference), reference)
        assert set(result) == {"p95", "p99"}
        assert all(value is not None and value > 0 for value in result.values())

    def test_different_shapes_raise_rather_than_broadcast(self, reference: GriddedField):
        small = resample(reference, reference.latitude[::50], reference.longitude[::50])
        with pytest.raises(ValueError, match="different shapes"):
            M.rmse(small, reference)

    def test_misaligned_axes_raise(self, reference: GriddedField):
        shifted = GriddedField(
            reference.values.copy(),
            reference.latitude + 0.01,
            reference.longitude.copy(),
        )
        with pytest.raises(ValueError, match="different coordinate axes"):
            M.rmse(shifted, reference)

    def test_metrics_are_computed_only_where_both_fields_are_finite(self, reference: GriddedField):
        partial = GriddedField(
            reference.values.copy(),
            reference.latitude.copy(),
            reference.longitude.copy(),
        )
        partial.values[:100, :] = np.nan
        table = M.evaluate_downscaling(partial, reference)
        assert table.valid_fraction == pytest.approx(0.5, abs=0.01)
        assert table["rmse"] == pytest.approx(0.0)

    def test_a_mostly_nan_field_reports_nothing(self, reference: GriddedField):
        mostly_nan = GriddedField(
            reference.values.copy(),
            reference.latitude.copy(),
            reference.longitude.copy(),
        )
        mostly_nan.values[:, :] = np.nan
        mostly_nan.values[0, 0] = 1.0
        table = M.evaluate_downscaling(mostly_nan, reference)
        assert table["rmse"] is None

    def test_descending_latitude_does_not_change_a_comparison(self, reference: GriddedField):
        """Guards the class of bug where a flipped axis silently misaligns rows."""
        flipped_reference = GriddedField(
            reference.values[::-1].copy(),
            reference.latitude[::-1].copy(),
            reference.longitude.copy(),
        )
        predicted = smoothed_field(reference)
        as_is = M.evaluate_downscaling(predicted, reference)
        flipped = M.evaluate_downscaling(predicted, flipped_reference)
        assert flipped["rmse"] == pytest.approx(as_is["rmse"])
        assert flipped["peak_preservation"] == pytest.approx(as_is["peak_preservation"])
        assert flipped["iou"] == pytest.approx(as_is["iou"])


class TestMetricTable:
    def test_unavailable_metrics_are_listed(self):
        table = MetricTable(metrics={"rmse": 1.0, "iou": None})
        assert table.unavailable() == ["iou"]

    def test_markdown_renders_null_as_a_dash_not_zero(self):
        table = MetricTable(metrics={"rmse": None}, units="mm/24h")
        rendered = table.to_markdown()
        assert "—" in rendered
        assert "0" not in rendered

    def test_units_are_omitted_for_dimensionless_metrics(self):
        table = MetricTable(
            metrics={"rmse": 1.5, "peak_preservation": 0.9, "iou": 0.8}, units="mm/24h"
        )
        rendered = table.to_markdown()
        assert "1.5 mm/24h" in rendered
        assert "0.9 |" in rendered
        assert "0.9 mm/24h" not in rendered
        assert "0.8 mm/24h" not in rendered

    def test_table_saves_as_json_artefact(self, tmp_path):
        table = MetricTable(metrics={"peak_preservation": 0.94}, units="mm/24h")
        path = table.save_json(tmp_path / "nested" / "metrics.json")
        assert path.exists()
        assert '"peak_preservation": 0.94' in path.read_text(encoding="utf-8")

    def test_extreme_threshold_source_is_recorded(self, reference: GriddedField):
        table = M.evaluate_downscaling(smoothed_field(reference), reference)
        assert table.extreme_threshold is not None
        assert table.extreme_threshold_source == "reference field p99"

    def test_caller_supplied_threshold_is_recorded_as_such(self, reference: GriddedField):
        table = M.evaluate_downscaling(
            smoothed_field(reference), reference, extreme_threshold=12.0
        )
        assert table.extreme_threshold == 12.0
        assert table.extreme_threshold_source == "supplied by caller"

    def test_synthetic_provenance_propagates_into_the_table(self, reference: GriddedField):
        table = M.evaluate_downscaling(smoothed_field(reference), reference)
        assert table.context["synthetic"] is True
        assert table.context["trained"] is False


class TestConfigAgreement:
    def test_every_metric_the_config_names_is_produced(self, reference: GriddedField):
        """configs/validation.yaml lists the downscaling metrics; all must be emitted.

        Same guard as the Threat Object field list: the config and the code must not
        drift apart, or a metric the team believes is being measured will silently
        be absent from every table.
        """
        import yaml

        config_file = REPO_ROOT / "configs" / "validation.yaml"
        declared = yaml.safe_load(config_file.read_text(encoding="utf-8"))["downscaling"][
            "metrics"
        ]

        table = M.evaluate_downscaling(smoothed_field(reference), reference)
        missing = [name for name in declared if name not in table.metrics]
        assert not missing, f"config declares metrics the table does not produce: {missing}"

    def test_metric_direction_table_covers_the_gate_metrics(self):
        """Every metric a gate can be configured on must have a known direction."""
        from src.validation.evaluation import metric_is_higher_better

        config_file = REPO_ROOT / "configs" / "validation.yaml"
        import yaml

        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        primary = config["gates"]["primary_metric"]
        unknown = [
            metric for metric in primary.values() if metric_is_higher_better(metric) is None
        ]
        assert not unknown, f"no direction known for gate metrics: {unknown}"
