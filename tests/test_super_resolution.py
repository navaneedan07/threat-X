"""Tests for the learned downscaler (``src/models/downscaling/super_resolution.py``).

Run:  pytest tests/test_super_resolution.py -v

No raw archives are needed: everything here drives the model through its public
helpers on synthetic fields, so the maths is tested even when the ERA5/ERA5-Land
files are absent. The one real-data path is checked for *refusal*, not for a number.

The property that matters most is at the bottom: the learned filter must be able to
**recover a peak the bilinear baseline flattened**. If a future edit turns the model
into a pure smoother — dropping the residual block, or the intercept — that test
fails, and it should, because a smoother can never beat the baseline it is measured
against.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.downscaling.baseline import downscale_to_reference, resample
from src.downscaling.metrics import evaluate_downscaling
from src.models.downscaling import super_resolution as sr
from src.shared.fields import GriddedField
from src.shared.synthetic import sharp_peak_field

LAT_RANGE = (8.0, 18.0)
LON_RANGE = (75.0, 85.0)


def synthetic_sequence(
    steps: int = 8,
    factor: int = 6,
    centre_lon: float = 77.5,
    centre_lon_step: float = 0.6,
) -> dict:
    """A moving sharp peak, coarsened and refined back — coarse input, fine target.

    The peak marches east across the domain, so a longitudinal holdout genuinely
    contains the extreme at some steps rather than only its tail.
    """
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_times: list[str] = []
    latitude = longitude = None
    for step in range(steps):
        reference = sharp_peak_field(
            lat_range=LAT_RANGE,
            lon_range=LON_RANGE,
            centre=(11.0 + 0.3 * step, centre_lon + centre_lon_step * step),
        )
        # np.linspace rather than a stride slice, so the coarse grid spans the whole fine
        # extent: a stride slice stops short of the far edge and leaves a NaN border that
        # has nothing to do with what is being tested here.
        count = reference.latitude.size // factor
        coarse_latitude = np.linspace(reference.latitude[0], reference.latitude[-1], count)
        coarse_longitude = np.linspace(reference.longitude[0], reference.longitude[-1], count)
        coarse = resample(reference, coarse_latitude, coarse_longitude, order=1)
        upsampled = downscale_to_reference(coarse, reference, order=1)
        inputs.append(upsampled.values)
        targets.append(reference.values)
        latitude, longitude = reference.latitude, reference.longitude
        valid_times.append(f"2020-05-{16 + step:02d} 00:00:00")

    return {
        "event": "synthetic",
        "variable": "tp",
        "units": "mm/24h",
        "latitude": latitude,
        "longitude": longitude,
        "inputs": inputs,
        "targets": targets,
        "valid_times": valid_times,
        "steps": steps,
    }


def fitted(sequence: dict, alpha: float, radius: int = 2, columns: np.ndarray | None = None):
    """Fit a filter with a fixed penalty on the given (or all) columns."""
    if columns is None:
        columns = np.arange(sequence["longitude"].size)
    equations = sr.accumulate(sequence, {"train": columns}, radius)
    return sr.LearnedDownscaler(radius=radius, alpha=alpha).fit(equations["train"])


class TestFeatures:
    def test_count_matches_the_documented_shape(self):
        assert sr.feature_count(0) == 3
        assert sr.feature_count(2) == 27

    def test_negative_radius_is_rejected(self):
        with pytest.raises(ValueError):
            sr.feature_count(-1)

    def test_constant_field_has_zero_residuals_and_constant_centre(self):
        values = np.full((5, 6), 3.5)
        design = sr.build_features(values, radius=1)
        assert design.shape == (5, 6, 11)
        assert np.allclose(design[..., 0], 1.0)
        assert np.allclose(design[..., 1], 3.5)
        assert np.allclose(design[..., 2:], 0.0)

    def test_border_cells_are_padded_not_dropped(self):
        values = np.arange(12, dtype=float).reshape(3, 4)
        design = sr.build_features(values, radius=2)
        assert design.shape == (3, 4, 27)
        assert np.isfinite(design).all()


class TestRidge:
    def test_intercept_is_never_penalised(self):
        """With an overwhelming penalty only the intercept survives, at the mean."""
        target = np.array([10.0, 12.0, 14.0])
        design = np.column_stack([np.ones(3), target, target**2])
        gram = design.T @ design
        rhs = design.T @ target
        weights = sr.solve_ridge(gram, rhs, alpha=1e12)
        assert weights[0] == pytest.approx(target.mean(), rel=1e-6)
        assert np.allclose(weights[1:], 0.0, atol=1e-6)

    def test_negative_alpha_is_rejected(self):
        with pytest.raises(ValueError):
            sr.solve_ridge(np.eye(3), np.zeros(3), alpha=-1.0)

    def test_small_penalty_recovers_an_exact_linear_relation(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        target = 2.0 + 3.0 * x
        design = np.column_stack([np.ones(x.size), x])
        weights = sr.solve_ridge(design.T @ design, design.T @ target, alpha=1e-9)
        assert weights[0] == pytest.approx(2.0, abs=1e-6)
        assert weights[1] == pytest.approx(3.0, abs=1e-6)


class TestNormalEquations:
    def test_residual_sum_of_squares_matches_a_direct_computation(self):
        rng = np.random.default_rng(1)
        design = rng.normal(size=(50, 4))
        target = rng.normal(size=50)
        weights = np.array([1.0, -0.5, 2.0, 0.25])

        equations = sr._NormalEquations(4)
        equations.add(design, target)

        direct = float(np.sum((target - design @ weights) ** 2))
        assert equations.residual_sum_of_squares(weights) == pytest.approx(direct)
        assert equations.count == 50

    def test_accumulating_in_blocks_equals_one_shot(self):
        rng = np.random.default_rng(2)
        design = rng.normal(size=(40, 3))
        target = rng.normal(size=40)
        weights = np.array([0.5, 1.0, -1.5])

        whole = sr._NormalEquations(3)
        whole.add(design, target)
        pieces = sr._NormalEquations(3)
        for start in range(0, 40, 7):
            pieces.add(design[start : start + 7], target[start : start + 7])

        assert pieces.residual_sum_of_squares(weights) == pytest.approx(
            whole.residual_sum_of_squares(weights)
        )
        assert pieces.count == whole.count


class TestRegularisationSelection:
    def test_picks_the_best_candidate_and_reports_the_bilinear_reference(self):
        sequence = synthetic_sequence(steps=3)
        columns = np.arange(sequence["longitude"].size)
        equations = sr.accumulate(sequence, {"fit": columns, "validate": columns}, radius=2)

        alpha, report = sr.select_regularisation(
            equations["fit"], equations["validate"], (1e-4, 1.0, 100.0)
        )

        assert alpha in (1e-4, 1.0, 100.0)
        assert set(report["alphas"]) == {"0.0001", "1", "100"}
        assert report["chosen"] == alpha
        # The identity predictor is exactly "copy the bilinear input".
        identity = np.zeros(sr.feature_count(2))
        identity[1] = 1.0
        assert report["bilinear_rmse"] == pytest.approx(
            np.sqrt(equations["validate"].residual_sum_of_squares(identity) / equations[
                "validate"
            ].count)
        )

    def test_an_empty_block_is_an_error_not_a_guess(self):
        sequence = synthetic_sequence(steps=1)
        columns = np.arange(sequence["longitude"].size)
        equations = sr.accumulate(sequence, {"fit": columns, "validate": columns}, radius=1)
        empty = sr._NormalEquations(sr.feature_count(1))
        with pytest.raises(ValueError, match="cannot select a penalty"):
            sr.select_regularisation(equations["fit"], empty)


class TestHoldoutSplit:
    def test_split_is_spatial_and_disjoint(self):
        longitude = np.linspace(75.0, 85.0, 201)
        train, test = sr.holdout_split(longitude, 0.3)
        assert train.size and test.size
        assert not set(train) & set(test)
        assert longitude[train].max() < longitude[test].min()

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_a_degenerate_fraction_is_rejected(self, fraction):
        with pytest.raises(ValueError):
            sr.holdout_split(np.linspace(75.0, 85.0, 50), fraction)


class TestLearnedDownscaler:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            sr.LearnedDownscaler().predict(np.zeros((4, 4)))

    def test_kernel_puts_the_centre_weight_in_the_middle(self):
        model = sr.LearnedDownscaler(radius=1, alpha=1.0)
        weights = np.arange(sr.feature_count(1), dtype=float)
        model.weights = weights
        kernel = model.kernel()
        assert kernel.shape == (3, 3)
        # residuals are weights[2:], and the centre term is added to the middle cell
        assert kernel[0, 0] == pytest.approx(weights[2])
        assert kernel[1, 1] == pytest.approx(weights[1] + weights[2 + 4])

    def test_floor_clips_an_unphysical_prediction(self):
        sequence = synthetic_sequence(steps=2)
        model = fitted(sequence, alpha=1e-6)
        model.floor = 0.0
        # nanmin: the refined input has NaN just outside the coarse field's extent, and
        # a real run masks those cells away too (see load_event_sequence).
        prediction = model.predict(sequence["inputs"][0])
        assert np.isfinite(prediction).any()
        assert float(np.nanmin(prediction)) >= 0.0

    def test_describe_records_that_it_is_trained_and_not_a_deep_network(self):
        model = fitted(synthetic_sequence(steps=2), alpha=1.0)
        summary = model.describe()
        assert summary["trained"] is True
        assert summary["architecture"] == "learned_linear_filter_5x5"
        assert "not a deep network" in summary["architecture_note"]
        assert len(summary["coefficients"]) == sr.feature_count(2)
        assert summary["training_cells"] > 0


class TestQuantileCalibration:
    """The calibration exists to undo the least-squares shrinkage of the tail."""

    @staticmethod
    def _pools(size: int = 20000, factor: float = 0.8):
        rng = np.random.default_rng(7)
        reference = rng.normal(300.0, 5.0, size=size)
        predicted = 300.0 + factor * (reference - 300.0)
        return predicted, reference

    def test_a_linear_shrinkage_of_the_tail_is_undone(self):
        predicted, reference = self._pools()
        calibration = sr.fit_calibration(predicted, reference)

        before = abs(float(np.percentile(predicted, 99) - np.percentile(reference, 99)))
        after = abs(
            float(
                np.percentile(calibration.apply(predicted), 99)
                - np.percentile(reference, 99)
            )
        )
        assert after < before
        assert after < 0.05 * before

    def test_the_map_is_monotone_by_construction(self):
        predicted, reference = self._pools()
        calibration = sr.fit_calibration(predicted, reference)
        assert calibration.describe()["monotone"] is True
        assert np.all(np.diff(calibration.reference_quantiles) >= 0.0)
        assert np.all(np.diff(calibration.predicted_quantiles) > 0.0)

    def test_applying_is_order_preserving(self):
        predicted, reference = self._pools()
        calibration = sr.fit_calibration(predicted, reference)
        sample = np.array([295.0, 300.0, 305.0, 310.0])
        mapped = calibration.apply(sample)
        assert np.all(np.diff(mapped) > 0.0)

    def test_nan_cells_are_preserved_not_mapped(self):
        predicted, reference = self._pools()
        calibration = sr.fit_calibration(predicted, reference)
        mapped = calibration.apply(np.array([np.nan, 300.0]))
        assert np.isnan(mapped[0])
        assert np.isfinite(mapped[1])

    def test_values_outside_the_fitted_range_clamp_to_the_end_knots(self):
        predicted, reference = self._pools()
        calibration = sr.fit_calibration(predicted, reference)
        mapped = calibration.apply(np.array([-1e6, 1e6]))
        assert mapped[0] == pytest.approx(calibration.reference_quantiles[0])
        assert mapped[1] == pytest.approx(calibration.reference_quantiles[-1])

    def test_paired_but_too_few_cells_is_an_error(self):
        with pytest.raises(ValueError, match="paired cells"):
            sr.fit_calibration(np.arange(10.0), np.arange(10.0), n_quantiles=201)

    def test_a_constant_prediction_cannot_be_calibrated(self):
        with pytest.raises(ValueError, match="constant"):
            sr.fit_calibration(np.full(500, 3.0), np.linspace(0.0, 1.0, 500))

    def test_mismatched_pool_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="paired"):
            sr.fit_calibration(np.zeros(100), np.zeros(101))

    def test_fewer_than_two_knots_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            sr.fit_calibration(np.arange(100.0), np.arange(100.0), n_quantiles=1)

    def test_non_finite_cells_are_dropped_before_fitting(self):
        predicted = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        reference = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        calibration = sr.fit_calibration(predicted, reference, n_quantiles=2)
        assert calibration.fitted_cells == 3

    def test_describe_records_the_provenance(self):
        predicted, reference = self._pools()
        summary = sr.fit_calibration(predicted, reference).describe()
        assert summary["fitted"] is True
        assert summary["n_quantiles"] == sr.DEFAULT_QUANTILES
        assert summary["fitted_cells"] == 20000
        assert "out-of-fold" in summary["note"]


class TestRadiusSweep:
    def test_reports_every_radius_and_picks_the_lowest_validation_rmse(self):
        sequence = synthetic_sequence(steps=4)
        train, test = sr.holdout_split(sequence["longitude"], 0.4)
        report = sr.radius_sweep(sequence, train, test, radii=(1, 2), alphas=(1e-6, 1.0))

        assert set(report["radii"]) == {"1", "2"}
        assert report["radii"]["1"]["parameters"] == sr.feature_count(1)
        assert report["radii"]["2"]["kernel_size"] == 5
        assert report["chosen"] in (1, 2)
        scored = {radius: row["validation_rmse"] for radius, row in report["radii"].items()}
        assert report["chosen_validation_rmse"] == pytest.approx(scored[str(report["chosen"])])
        assert report["chosen_validation_rmse"] == pytest.approx(min(scored.values()))


class TestAccumulate:
    def test_cells_outside_the_reference_footprint_are_excluded(self):
        """A land-only reference defines the training region; ocean cells are dropped."""
        sequence = synthetic_sequence(steps=2)
        blank = 10
        sequence["targets"][0] = sequence["targets"][0].copy()
        columns = np.arange(sequence["longitude"].size)
        rows = int(sequence["targets"][0].shape[0])

        sequence["targets"][0][:, :blank] = np.nan
        blanked = sr.accumulate(sequence, {"train": columns}, radius=1)["train"].count

        sequence["targets"][0][:, :blank] = sequence["targets"][1][:, :blank]
        filled = sr.accumulate(sequence, {"train": columns}, radius=1)["train"].count

        # Blanking a block of the reference removes exactly those cells, and nothing
        # else: the count is what a land-only reference costs the training set.
        assert filled - blanked == rows * blank

    def test_all_nan_targets_yield_no_cells(self):
        sequence = synthetic_sequence(steps=1)
        sequence["targets"][0] = np.full_like(sequence["targets"][0], np.nan)
        columns = np.arange(sequence["longitude"].size)
        equations = sr.accumulate(sequence, {"train": columns}, radius=1)
        assert equations["train"].count == 0


class TestRealDataIsRequired:
    def test_a_missing_archive_raises_instead_of_falling_back(self, monkeypatch):
        monkeypatch.setattr(
            sr, "FINE_FILES", {"heatwave": "data/raw/definitely_not_here.nc"}
        )
        monkeypatch.setattr(
            sr, "COARSE_FILES", {"heatwave": "data/raw/definitely_not_here.nc"}
        )
        with pytest.raises(FileNotFoundError, match="missing"):
            sr.load_event_sequence("heatwave")


def peak_windows(sequence: dict, test_columns: np.ndarray, tolerance: float = 0.99) -> list[int]:
    """Steps whose *global* extreme actually falls inside the holdout columns.

    The same rule the real-data experiment uses: scoring peak preservation on a
    window that does not contain the peak measures nothing, so those steps are left
    out rather than allowed to dilute the mean.
    """
    keep: list[int] = []
    for index, target in enumerate(sequence["targets"]):
        window_peak = float(np.nanmax(target[:, test_columns]))
        if window_peak >= tolerance * float(np.nanmax(target)):
            keep.append(index)
    return keep


class TestPeakRecovery:
    """The capability the whole module exists for: sharpening, not smoothing."""

    def test_learned_filter_beats_bilinear_at_preserving_the_peak(self):
        sequence = synthetic_sequence(
            steps=6, factor=6, centre_lon=76.0, centre_lon_step=1.4
        )
        train_columns, test_columns = sr.holdout_split(sequence["longitude"], 0.5)
        keep = peak_windows(sequence, test_columns)
        assert len(keep) >= 3, "the fixture must place the peak inside the holdout"

        # A small penalty keeps the sharpening the residual block can express. This
        # is deliberately *not* the penalty that minimises RMSE: RMSE is dominated by
        # the vast near-zero surroundings, so it prefers a smoother and flatter field.
        equations = sr.accumulate(sequence, {"train": train_columns}, radius=2)
        model = sr.LearnedDownscaler(radius=2, alpha=1e-6).fit(equations["train"])

        latitude = sequence["latitude"]
        longitude = sequence["longitude"][test_columns]
        learned_scores: list[float] = []
        baseline_scores: list[float] = []
        for index in keep:
            input_values = sequence["inputs"][index]
            target_values = sequence["targets"][index]
            reference = GriddedField(
                values=target_values[:, test_columns],
                latitude=latitude,
                longitude=longitude,
                units="mm/24h",
            )
            learned = GriddedField(
                values=model.predict(input_values)[:, test_columns],
                latitude=latitude,
                longitude=longitude,
                units="mm/24h",
                attrs={"trained": True},
            )
            baseline = GriddedField(
                values=input_values[:, test_columns],
                latitude=latitude,
                longitude=longitude,
                units="mm/24h",
            )
            learned_scores.append(evaluate_downscaling(learned, reference)["peak_preservation"])
            baseline_scores.append(evaluate_downscaling(baseline, reference)["peak_preservation"])

        assert all(score is not None for score in learned_scores + baseline_scores)
        assert np.mean(learned_scores) > np.mean(baseline_scores)

    def test_a_pure_smoother_could_not(self):
        """Sanity check on the previous test: the baseline itself cannot amplify.

        Pinning this makes the property above meaningful — it shows the coarse input
        genuinely arrives with its peak flattened, so the improvement has to come
        from the learned filter rather than from the data already being sharp.
        """
        sequence = synthetic_sequence(
            steps=6, factor=6, centre_lon=76.0, centre_lon_step=1.4
        )
        _, test_columns = sr.holdout_split(sequence["longitude"], 0.5)
        keep = peak_windows(sequence, test_columns)
        preserved = []
        for index in keep:
            input_values = sequence["inputs"][index]
            target_values = sequence["targets"][index]
            latitude = sequence["latitude"]
            longitude = sequence["longitude"][test_columns]
            reference = GriddedField(
                values=target_values[:, test_columns], latitude=latitude, longitude=longitude
            )
            baseline = GriddedField(
                values=input_values[:, test_columns], latitude=latitude, longitude=longitude
            )
            preserved.append(evaluate_downscaling(baseline, reference)["peak_preservation"])
        assert np.mean(preserved) < 1.0
