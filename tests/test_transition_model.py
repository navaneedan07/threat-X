"""Tests for the transition dataset builder and the learned model.

Run:  pytest tests/test_transition_model.py -v

The fixtures here stand in for `data/processed/precursors.json`, which needs the
precursor pipeline and therefore the raw ERA5 archives. Building the fixtures
by hand means this suite runs anywhere, and — more importantly — lets the tests
assert properties that real data cannot demonstrate:

* **No future leakage.** Every row's features are checked against the sample at
  that row's own timestamp, so a feature accidentally taken from `t+1` fails here.
* **A planted lead relationship is recovered.** One fixture makes the pressure
  tendency lead severity by three steps, which is the physical claim the project
  makes. If the model cannot beat the base rate on data built with a real lead,
  the model is broken.
* **Refusal is tested as a feature, not an error.** Each `min_rows` /
  `min_positives` guard has a test asserting it declines with a reason, because a
  logistic regression will happily fit six rows and emit a meaningless number.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.models.transition.transition_model import (
    FEATURE_FIELDS,
    HORIZONS_HOURS,
    MIN_PUBLISHABLE_AUC,
    MIN_PUBLISHABLE_SKILL,
    FittedTransitionModel,
    HorizonResult,
    TransitionDataError,
    TransitionDataset,
    TransitionReport,
    build_dataset,
    derive_threshold,
    load_precursor_entries,
    severity_field_for,
    train,
    train_all_horizons,
)
from src.shared.metrics import MetricTable

START = datetime(2024, 5, 1, tzinfo=timezone.utc)
NULL_PRESSURE_LEVEL_FIELDS = (
    "moisture_flux_convergence_g_kg_s",
    "vorticity_850_s1",
    "theta_e_gradient_k_100km",
)


def _entry(
    threat_id: str = "THR-2024-0001",
    *,
    event_type: str = "extreme_rainfall",
    steps: int = 80,
    lead: int = 3,
    seed: int = 1,
    step_hours: int = 3,
    sparse_features_at: set[int] | None = None,
) -> dict:
    """A synthetic precursor entry in the real pipeline's output shape.

    Severity oscillates and drifts, so it crosses the escalation boundary more
    than once and both label classes are populated. ``lead`` shifts the pressure
    tendency ahead of severity, planting a genuine predictive relationship.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(steps, dtype=float)
    severity = 10.0 + 6.0 * np.sin(index / 6.0) + 0.15 * index

    lead_value = np.concatenate([severity[lead:], np.full(lead, severity[-1])])
    pressure = -(lead_value - lead_value.mean()) * 0.8 + rng.normal(0, 0.15, steps)
    divergence = -1e-5 * (1.0 + 0.02 * index) + rng.normal(0, 2e-7, steps)
    wind = 5.0 + 0.1 * index + rng.normal(0, 0.2, steps)
    temperature = 1.0 + 0.3 * np.sin(index / 5.0)

    sparse = sparse_features_at or set()
    series = []
    for step in range(steps):
        series.append(
            {
                "timestamp": (START + timedelta(hours=step_hours * step)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pressure_tendency_3h_hpa": (
                    None if step in sparse else float(pressure[step])
                ),
                "wind_divergence_s1": float(divergence[step]),
                "wind_speed_ms": float(wind[step]),
                "precipitation_rate_mm3h": float(severity[step]),
                "t2m_anomaly_k": float(temperature[step]),
                # Not in the dataset — must stay null and never become a feature.
                "moisture_flux_convergence_g_kg_s": None,
                "vorticity_850_s1": None,
                "theta_e_gradient_k_100km": None,
            }
        )

    return {
        "threat_id": threat_id,
        "event_type": event_type,
        "time_step_hours": float(step_hours),
        "dataset_key": "synthetic",
        "series": series,
    }


class TestFeatureContract:
    def test_pressure_level_fields_are_not_features(self):
        """They are null for every sample; using them would mean imputing physics."""
        for field in NULL_PRESSURE_LEVEL_FIELDS:
            assert field not in FEATURE_FIELDS

    def test_severity_field_mapping_covers_the_event_types_in_use(self):
        assert severity_field_for("extreme_rainfall") == "precipitation_rate_mm3h"
        assert severity_field_for("heatwave") == "t2m_anomaly_k"
        assert severity_field_for("cyclone") == "precipitation_rate_mm3h"

    def test_unknown_event_type_falls_back_rather_than_raising(self):
        assert severity_field_for("something_new") == "precipitation_rate_mm3h"

    def test_horizons_match_the_documented_set(self):
        assert HORIZONS_HOURS == (6.0, 12.0, 18.0, 24.0)


class TestThresholdDerivation:
    def test_threshold_is_a_quantile_of_the_series(self):
        severity = [float(value) for value in range(1, 101)]
        assert derive_threshold(severity, 0.75) == pytest.approx(75.75, abs=0.5)

    def test_missing_values_are_ignored_not_zeroed(self):
        assert derive_threshold([None, 10.0, None, 20.0], 0.5) == pytest.approx(15.0)

    def test_no_finite_values_gives_none_rather_than_a_default(self):
        assert derive_threshold([None, None], 0.75) is None

    def test_absolute_quantiles_are_refused(self):
        with pytest.raises(ValueError, match="strictly between"):
            derive_threshold([1.0, 2.0], 1.0)


class TestDatasetConstruction:
    def test_rows_come_from_a_single_series(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        assert len(dataset) > 0
        assert dataset.features.shape[1] == len(FEATURE_FIELDS)
        assert dataset.feature_names == list(FEATURE_FIELDS)

    def test_features_are_taken_from_the_rows_own_timestamp(self):
        """The no-future-leakage check: features must equal the sample at `t`."""
        entry = _entry()
        dataset = build_dataset({"THR-2024-0001": entry}, horizon_hours=12.0)
        by_stamp = {sample["timestamp"]: sample for sample in entry["series"]}
        for row, stamp in enumerate(dataset.timestamps):
            sample = by_stamp[stamp.strftime("%Y-%m-%dT%H:%M:%SZ")]
            expected = [sample[name] for name in FEATURE_FIELDS]
            assert list(dataset.features[row]) == pytest.approx(expected)

    def test_rows_whose_horizon_runs_past_the_end_are_dropped(self):
        """Shortening the series must shrink the dataset, not invent labels."""
        long_dataset = build_dataset({"THR-2024-0001": _entry(steps=80)}, horizon_hours=24.0)
        short_dataset = build_dataset({"THR-2024-0001": _entry(steps=40)}, horizon_hours=24.0)
        assert len(short_dataset) < len(long_dataset)

    def test_longer_horizon_yields_fewer_rows(self):
        """Each horizon has its own sample count, which is why they are not comparable."""
        six = build_dataset({"THR-2024-0001": _entry(steps=80)}, horizon_hours=6.0)
        twenty_four = build_dataset({"THR-2024-0001": _entry(steps=80)}, horizon_hours=24.0)
        assert len(twenty_four) < len(six)

    def test_rows_already_at_the_boundary_are_excluded(self):
        """A step already at the target has nothing left to escalate to."""
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        for value in dataset.features[:, FEATURE_FIELDS.index("precipitation_rate_mm3h")]:
            assert value < dataset.target.threshold

    def test_rows_with_a_missing_feature_are_dropped(self):
        full = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        sparse = build_dataset(
            {"THR-2024-0001": _entry(sparse_features_at={5, 6, 7})}, horizon_hours=12.0
        )
        assert len(sparse) < len(full)
        assert np.isfinite(sparse.features).all()

    def test_both_label_classes_are_present(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        assert dataset.positives > 0
        assert dataset.negatives > 0
        assert 0.0 < dataset.base_rate < 1.0

    def test_labels_are_binary(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        assert set(np.unique(dataset.labels)) <= {0, 1}

    def test_step_hours_is_inferred_from_the_series(self):
        dataset = build_dataset({"THR-2024-0001": _entry(step_hours=3)}, horizon_hours=12.0)
        assert dataset.target.step_hours == pytest.approx(3.0)
        assert dataset.target.steps_ahead == 4

    def test_pooling_events_with_different_severity_fields_is_refused(self):
        """A rainfall threshold and a temperature threshold cannot share a dataset."""
        entries = {
            "THR-2024-0001": _entry("THR-2024-0001", event_type="extreme_rainfall"),
            "THR-2024-0002": _entry("THR-2024-0002", event_type="heatwave"),
        }
        with pytest.raises(TransitionDataError, match="differs from"):
            build_dataset(entries, horizon_hours=12.0)

    def test_pooling_same_type_events_across_two_boundaries_is_refused(self):
        """Two events of one type still have different severity distributions.

        The boundary is a quantile of each event's *own* severity, so pooling them
        would silently label the second event against the first event's boundary —
        the same failure as mixing event types, and just as invisible.
        """
        entries = {
            "THR-2024-0001": _entry("THR-2024-0001", steps=80),
            "THR-2024-0002": _entry("THR-2024-0002", steps=60),
        }
        with pytest.raises(TransitionDataError, match="escalation threshold"):
            build_dataset(entries, horizon_hours=12.0)

    def test_series_with_no_finite_severity_is_skipped_with_a_reason(self):
        entry = _entry()
        for sample in entry["series"]:
            sample["precipitation_rate_mm3h"] = None
        dataset = build_dataset({"THR-2024-0001": entry}, horizon_hours=12.0)
        assert len(dataset) == 0
        assert any("no finite" in note for note in dataset.notes)

    def test_empty_input_gives_an_empty_dataset_not_an_exception(self):
        dataset = build_dataset({}, horizon_hours=12.0)
        assert len(dataset) == 0
        assert dataset.base_rate is None
        assert any("no usable series" in note for note in dataset.notes)

    def test_dataset_serialises_the_target_that_produced_it(self):
        payload = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0).to_dict()
        assert payload["horizon_hours"] == 12.0
        assert payload["threshold"] is not None
        assert "precipitation_rate_mm3h" in payload["target"]


class TestTrainingRefusals:
    """Each guard must decline with a reason, never fit anyway."""

    def test_too_few_rows_is_refused(self):
        dataset = build_dataset({"THR-2024-0001": _entry(steps=12)}, horizon_hours=12.0)
        model, reason = train(dataset, min_rows=20)
        assert model is None
        assert reason is not None and "rows" in reason

    def test_too_few_positives_is_refused(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        model, reason = train(dataset, min_positives=len(dataset) + 1)
        assert model is None
        assert reason is not None and "positive" in reason

    def test_no_negatives_is_refused(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        model, reason = train(dataset, min_rows=1, min_positives=1)
        # Force the degenerate case: every label is 1.
        dataset.labels[:] = 1
        model, reason = train(dataset, min_rows=1, min_positives=1)
        assert model is None
        assert reason is not None and "negative" in reason


class TestFittedModel:
    @pytest.fixture(scope="class")
    def fitted(self):
        dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
        model, reason = train(dataset)
        assert reason is None, f"fixture failed to fit: {reason}"
        return dataset, model

    def test_model_id_names_the_severity_field_and_horizon(self, fitted):
        _, model = fitted
        assert "precipitation_rate_mm3h" in model.model_id
        assert "h12" in model.model_id

    def test_probability_is_in_the_unit_interval(self, fitted):
        dataset, model = fitted
        probability = model.predict_probability(dataset.features[0])
        assert 0.0 <= probability <= 1.0

    def test_missing_feature_yields_none_not_a_number(self, fitted):
        _, model = fitted
        incomplete = {name: 1.0 for name in model.feature_names}
        incomplete["wind_speed_ms"] = None
        assert model.predict_probability(incomplete) is None

    def test_dict_input_is_ordered_by_feature_name(self, fitted):
        _, model = fitted
        vector = [0.5] * len(model.feature_names)
        as_dict = dict(zip(model.feature_names, vector, strict=True))
        assert model.predict_probability(as_dict) == pytest.approx(
            model.predict_probability(vector)
        )

    def test_wrong_length_vector_is_refused(self, fitted):
        _, model = fitted
        with pytest.raises(ValueError, match="expected"):
            model.predict_probability([1.0, 2.0])

    def test_estimate_without_a_probability_publishes_no_model_id(self, fitted):
        """Contract rule: a probability needs a model, and a null probability needs none."""
        _, model = fitted
        incomplete = {name: None for name in model.feature_names}
        estimate = model.to_estimate(incomplete)
        assert estimate.probability is None
        assert estimate.model_id is None
        assert estimate.is_publishable is False

    def test_estimate_with_a_probability_is_publishable(self, fitted):
        dataset, model = fitted
        estimate = model.to_estimate(dataset.features[0])
        assert estimate.probability is not None
        assert estimate.model_id == model.model_id
        assert estimate.is_publishable is True
        assert estimate.horizon_hours == 12

    def test_estimate_is_not_claimed_as_calibrated(self, fitted):
        dataset, model = fitted
        assert model.to_estimate(dataset.features[0]).calibrated is False


class TestSignalRecovery:
    def test_planted_lead_relationship_beats_the_base_rate(self):
        """The fixture makes pressure tendency lead severity by 3 steps.

        That is the physical claim the project makes, so a model that cannot beat
        the constant base-rate forecast on data with a real lead is broken.
        """
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        result = report.results[0]
        assert result.trained, result.reason
        skill = result.metrics.get("skill_score")
        assert skill is not None
        assert skill > 0.0, "model failed to use a planted lead relationship"

    def test_a_short_horizon_can_be_correctly_refused(self):
        """A 6h horizon is 2 steps, so almost nothing crosses the boundary in time.

        The correct outcome is an explicit refusal, not a model fitted on four
        positives. This already happens on synthetic data and will be far more
        likely on the ~130 real rows, which is why every horizon reports its own
        row count and untrained horizons keep a null probability.
        """
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(6.0,))
        result = report.results[0]
        assert not result.trained
        assert result.reason is not None and "positive" in result.reason
        assert report.trained_horizons() == []

    def test_metrics_are_out_of_fold(self):
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        table = report.results[0].metrics
        assert table is not None
        assert "out-of-fold" in table.notes["evaluation"]
        assert table.context["evaluation"] == "5-fold out-of-fold"

    def test_base_rate_is_reported_alongside_the_score(self):
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        table = report.results[0].metrics
        assert table.get("base_rate") is not None
        assert table.get("brier_score") is not None

    def test_table_is_marked_as_a_proxy_target(self):
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        assert report.results[0].metrics.context["proxy_target"] is True

    def test_autocorrelation_caveat_is_recorded(self):
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        note = report.results[0].metrics.notes["evaluation"]
        assert "autocorrelation" in note


class TestHorizonReport:
    def test_each_horizon_reports_its_own_row_count(self):
        report = train_all_horizons({"THR-2024-0001": _entry()})
        counts = [len(result.dataset) for result in report.results]
        assert counts == sorted(counts, reverse=True), "longer horizons must yield fewer rows"

    def test_all_four_horizons_are_attempted(self):
        report = train_all_horizons({"THR-2024-0001": _entry()})
        assert [result.horizon_hours for result in report.results] == list(HORIZONS_HOURS)

    def test_untrained_horizons_carry_a_reason(self):
        report = train_all_horizons({"THR-2024-0001": _entry(steps=20)}, horizons=(24.0,))
        result = report.results[0]
        if not result.trained:
            assert result.reason

    def test_insufficient_data_is_reported_as_untrained_not_faked(self):
        """A tiny series must yield an explicit refusal, never a fitted model."""
        report = train_all_horizons({"THR-2024-0001": _entry(steps=10)}, horizons=(24.0,))
        assert report.trained_horizons() == []
        assert report.untrained_horizons() == [24.0]
        assert any("honest outcome" in note for note in report.notes)

    def test_markdown_renders_untrained_horizons_with_a_dash(self):
        report = train_all_horizons({"THR-2024-0001": _entry(steps=10)}, horizons=(24.0,))
        markdown = report.to_markdown()
        assert "| 24h |" in markdown
        assert "| — |" in markdown

    def test_report_saves_json_and_markdown(self, tmp_path):
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        paths = report.save(tmp_path / "validation")
        assert paths["json"].exists()
        assert paths["markdown"].exists()

    def test_saved_report_marks_itself_as_a_prototype(self, tmp_path):
        import json

        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        payload = json.loads(
            report.save(tmp_path / "v")["json"].read_text(encoding="utf-8")
        )
        assert "PROXY" in payload["_meta"]["note"]
        assert payload["horizons"][0]["model"]["trained"] is True


class TestLoading:
    def test_missing_file_names_the_command_to_run(self, tmp_path):
        with pytest.raises(TransitionDataError, match="precursors.pipeline"):
            load_precursor_entries(tmp_path / "absent.json")

    def test_file_without_entries_is_refused(self, tmp_path):
        path = tmp_path / "precursors.json"
        path.write_text('{"precursors": {}}', encoding="utf-8")
        with pytest.raises(TransitionDataError, match="no precursor entries"):
            load_precursor_entries(path)


# ---------------------------------------------------------------------------
# Publishing gate.
#
# These tests exist because the honest result on real ERA5 data is that the
# model has no useful skill (docs/experiments.md -> Transition intelligence).
# "It fitted" and "it may be published" are different claims, and the second
# one is enforced here so that no future edit can quietly start serving a
# negative-skill probability through the API.
# ---------------------------------------------------------------------------


def _scored(
    base: HorizonResult,
    skill: float | None,
    auc: float | None,
    *,
    horizon: float | None = None,
    reason: str | None = None,
) -> HorizonResult:
    """A real fitted horizon with its metrics swapped for controlled values.

    Reusing a real ``HorizonResult`` means the gate is exercised on the same
    object the report produces, not on a hand-built stand-in that could drift
    away from the real one.
    """
    if reason is not None:
        return replace(base, model=None, metrics=None, reason=reason)
    table = MetricTable(
        metrics={
            "skill_score": skill,
            "roc_auc": auc,
            "brier_score": 0.2,
            "base_rate": 0.3,
        }
    )
    if horizon is None:
        return replace(base, metrics=table)
    return replace(base, horizon_hours=horizon, metrics=table)


@pytest.fixture(scope="module")
def fitted_result() -> HorizonResult:
    """One real fit on the planted-lead fixture, shared by the gate tests."""
    report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
    result = report.results[0]
    assert result.trained, result.reason
    return result


@pytest.fixture(scope="module")
def fitted_12h() -> tuple[TransitionDataset, FittedTransitionModel]:
    """The fitted dataset and model behind ``fitted_result``."""
    dataset = build_dataset({"THR-2024-0001": _entry()}, horizon_hours=12.0)
    model, reason = train(dataset)
    assert model is not None, reason
    return dataset, model


class TestPublishingGate:
    def test_thresholds_are_the_two_no_skill_references(self):
        """The gate may contain exactly two numbers, and both mean "no information".

        Anything stricter would be a tuned threshold, which this project does not
        allow to be invented before the metric distribution is measured.
        """
        assert MIN_PUBLISHABLE_SKILL == 0.0
        assert MIN_PUBLISHABLE_AUC == 0.5

    def test_beating_both_references_is_publishable(self, fitted_result):
        result = _scored(fitted_result, 0.2, 0.75)
        assert result.publish_blockers == []
        assert result.publishable is True

    def test_negative_skill_is_withheld_even_with_a_good_auc(self, fitted_result):
        """A model can rank cases well and still lose to the base rate on Brier.

        Amphan 18h is exactly this case on real data (AUC 0.80, skill −0.03).
        """
        result = _scored(fitted_result, -0.03, 0.80)
        assert result.publishable is False
        assert any("base-rate forecast" in blocker for blocker in result.publish_blockers)

    def test_chance_level_auc_is_withheld_even_with_positive_skill(self, fitted_result):
        result = _scored(fitted_result, 0.2, 0.5)
        assert result.publishable is False
        assert any("chance" in blocker for blocker in result.publish_blockers)

    def test_an_unfitted_horizon_is_withheld_with_its_own_reason(self, fitted_result):
        result = _scored(fitted_result, None, None, reason="3 positives, minimum is 5")
        assert result.trained is False
        assert result.publishable is False
        assert result.publish_blockers == ["no model was fitted (3 positives, minimum is 5)"]

    def test_missing_metrics_withhold_rather_than_assume_skill(self, fitted_result):
        result = replace(
            fitted_result,
            metrics=MetricTable(metrics={"skill_score": None, "roc_auc": None}),
        )
        assert result.trained is True
        assert result.publishable is False
        assert len(result.publish_blockers) == 2

    def test_report_cannot_publish_when_every_horizon_is_withheld(self, fitted_result):
        report = TransitionReport(
            results=[
                _scored(fitted_result, -0.4, 0.7, horizon=12.0),
                _scored(fitted_result, 0.1, 0.5, horizon=18.0),
            ]
        )
        assert report.publishable_horizons() == []
        assert report.withheld_horizons() == [12.0, 18.0]
        assert report.can_publish() is False

    def test_report_can_publish_when_one_horizon_clears_the_gate(self, fitted_result):
        report = TransitionReport(
            results=[
                _scored(fitted_result, -0.4, 0.7, horizon=12.0),
                _scored(fitted_result, 0.1, 0.62, horizon=18.0),
            ]
        )
        assert report.publishable_horizons() == [18.0]
        assert report.withheld_horizons() == [12.0]
        assert report.can_publish() is True

    def test_dict_exposes_the_gate_for_the_api_layer(self, fitted_result):
        report = TransitionReport(results=[_scored(fitted_result, -0.4, 0.7)])
        payload = report.to_dict()
        assert payload["can_publish"] is False
        assert payload["publishable_horizons"] == []
        assert payload["horizons"][0]["publishable"] is False
        assert payload["horizons"][0]["publish_blockers"]

    def test_markdown_names_every_withheld_horizon_and_why(self, fitted_result):
        report = TransitionReport(
            results=[_scored(fitted_result, -0.4, 0.7, horizon=18.0)]
        )
        markdown = report.to_markdown()
        assert "### Publishing gate" in markdown
        assert "**Publishable:** none" in markdown
        assert "**18h withheld**" in markdown
        assert "base-rate forecast" in markdown

    def test_the_real_report_path_carries_the_gate(self, fitted_12h):
        """The gate is wired into the report, not only callable by hand."""
        report = train_all_horizons({"THR-2024-0001": _entry()}, horizons=(12.0,))
        assert report.can_publish() is report.results[0].publishable
        assert "Publishing gate" in report.to_markdown()
        assert any("Publishing gate" in note for note in report.notes)
