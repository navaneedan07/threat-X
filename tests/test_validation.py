"""Tests for the transition metrics and the validation gate.

Run:  pytest tests/test_validation.py -v

Two properties are the point of this file.

**The gate must refuse to guess.** ``configs/validation.yaml`` ships its
thresholds as ``null``, so against the real config every verdict must come back
undecided with the exact keys to fill in — not PASS, and not a defaulted number.
``TestGateAgainstShippedConfig`` proves that.

**A transition score must never be reported without its base rate.** A model that
predicts "no transition" everywhere scores a good Brier score on a rare event, so
``TestTransitionMetrics`` checks that the base rate and the skill score are always
present and that the no-skill case is labelled as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.contracts import GateVerdict
from src.shared.metrics import MetricTable
from src.validation.evaluation import (
    GateConfig,
    StageGate,
    decide_stage,
    default_output_dir,
    evaluate_validation,
    metric_is_higher_better,
)
from src.validation.transition_metrics import (
    base_rate,
    brier_score,
    brier_skill_score,
    calibration_table,
    evaluate_transition,
    expected_calibration_error,
    lead_time_error,
    roc_auc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Transition metrics
# ---------------------------------------------------------------------------


class TestTransitionMetrics:
    def test_base_rate_is_a_fraction(self):
        assert base_rate([0, 0, 0, 1]) == pytest.approx(0.25)
        assert base_rate([]) is None

    def test_brier_score_on_a_known_case(self):
        assert brier_score([0.8, 0.2], [1, 0]) == pytest.approx(0.04)

    def test_brier_score_rejects_out_of_range_probabilities(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score([1.4, 0.2], [1, 0])

    def test_brier_score_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError, match="pair up"):
            brier_score([0.5, 0.5], [1])

    def test_nan_probability_is_refused(self):
        """An unknown probability is null, not NaN."""
        with pytest.raises(ValueError, match="non-finite"):
            brier_score([0.5, float("nan")], [1, 0])

    def test_non_binary_outcomes_are_refused(self):
        with pytest.raises(ValueError, match="binary"):
            brier_score([0.5, 0.5], [1, 2])

    def test_skill_score_is_high_for_a_genuinely_skilful_forecast(self):
        assert brier_skill_score([0.8, 0.2], [1, 0]) == pytest.approx(0.84)

    def test_constant_base_rate_forecast_has_zero_skill(self):
        """The whole reason the base rate is mandatory: this is the reference."""
        outcomes = [0, 0, 0, 0, 1]
        rate = base_rate(outcomes)
        assert brier_skill_score([rate] * 5, outcomes) == pytest.approx(0.0, abs=1e-12)

    def test_predicting_no_transition_ever_scores_zero_skill_not_a_good_score(self):
        """The trap this whole metric set exists to expose.

        Predicting 0.0 for a 3%-base-rate event gives a Brier score of 0.03, which
        reads as excellent in isolation. The skill score against the constant
        base-rate forecast reveals it is in fact slightly *worse* than that.
        """
        outcomes = [0] * 97 + [1, 1, 1]
        no_skill = brier_score([0.0] * 100, outcomes)
        assert no_skill is not None and no_skill < 0.05  # superficially "good"
        assert brier_skill_score([0.0] * 100, outcomes) < 0

    def test_skill_score_is_none_for_a_degenerate_sample(self):
        assert brier_skill_score([0.5, 0.5], [0, 0]) is None

    def test_roc_auc_is_one_for_perfect_separation(self):
        assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)

    def test_roc_auc_is_half_for_random_ordering(self):
        assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)

    def test_roc_auc_is_none_with_only_one_class(self):
        """Reporting 0.5 here would imply a measured discrimination that never happened."""
        assert roc_auc([0.9, 0.2], [1, 1]) is None

    def test_calibration_table_marks_empty_bins_null(self):
        table = calibration_table([0.05], [0], bins=5)
        assert table[0]["count"] == 1
        assert table[0]["observed_frequency"] == pytest.approx(0.0)
        assert table[4]["count"] == 0
        assert table[4]["observed_frequency"] is None

    def test_expected_calibration_error_on_a_known_gap(self):
        assert expected_calibration_error([0.05] * 10, [0] * 10) == pytest.approx(0.05)

    def test_lead_time_error_skips_unpaired_samples(self):
        assert lead_time_error([10.0, 20.0, None], [12.0, 18.0, 5.0]) == pytest.approx(2.0)

    def test_lead_time_error_is_none_when_nothing_is_comparable(self):
        """Reporting 0.0 would read as a perfect lead time."""
        assert lead_time_error([None, None], [1.0, 2.0]) is None

    def test_lead_time_error_requires_equal_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            lead_time_error([1.0], [1.0, 2.0])


class TestTransitionTable:
    def test_base_rate_is_always_present(self):
        table = evaluate_transition([0.8, 0.2], [1, 0])
        assert "base_rate" in table.metrics
        assert table.get("base_rate") == pytest.approx(0.5)

    def test_no_skill_is_labelled_as_such(self):
        outcomes = [0, 0, 0, 0, 1]
        table = evaluate_transition([0.0] * 5, outcomes)
        assert table.get("skill_score") < 0
        assert "not a working model" in table.notes["skill_score"]

    def test_skill_improvement_is_labelled(self):
        table = evaluate_transition([0.8, 0.2], [1, 0])
        assert "improvement" in table.notes["skill_score"]

    def test_metric_names_match_the_config(self):
        config = (REPO_ROOT / "configs" / "validation.yaml").read_text(encoding="utf-8")
        table = evaluate_transition([0.8, 0.2], [1, 0])
        for name in ("brier_score", "roc_auc", "calibration", "lead_time_error_hours"):
            assert name in table.metrics
            assert f"- {name}" in config or name in config

    def test_uncalibrated_result_is_marked_untrained(self):
        table = evaluate_transition([0.8, 0.2], [1, 0])
        assert table.context["trained"] is False
        assert table.context["model_id"] is None

    def test_a_named_model_marks_the_table_as_trained(self):
        table = evaluate_transition([0.8, 0.2], [1, 0], model_id="logreg_precursor_v1")
        assert table.context["trained"] is True

    def test_missing_window_observations_are_explained(self):
        table = evaluate_transition([0.8, 0.2], [1, 0])
        assert table.get("lead_time_error_hours") is None
        assert "no observed transition times" in table.notes["lead_time_error_hours"]


# ---------------------------------------------------------------------------
# The gate against the shipped config
# ---------------------------------------------------------------------------


class TestGateAgainstShippedConfig:
    """configs/validation.yaml ships null thresholds; the gate must not guess."""

    def test_primary_metrics_are_declared_per_stage(self):
        config = GateConfig.from_config()
        assert config.stage("downscaling").primary_metric == "peak_preservation"
        assert config.stage("transition").primary_metric == "brier_score"
        assert config.stage("detection").primary_metric == "f1"
        assert config.stage("tracking").primary_metric == "track_continuity"

    def test_no_stage_is_configured(self):
        config = GateConfig.from_config()
        assert all(not gate.configured for gate in config.stages.values())

    def test_every_verdict_is_undecided(self):
        tables = {
            "detection": MetricTable(metrics={"f1": 0.9}),
            "tracking": MetricTable(metrics={"track_continuity": 0.95}),
            "downscaling": MetricTable(metrics={"peak_preservation": 0.97}),
            "transition": MetricTable(metrics={"brier_score": 0.02, "base_rate": 0.05}),
        }
        report = evaluate_validation(tables)
        # Every metric above would comfortably pass any sensible threshold — which
        # is exactly why an undecided verdict here is the correct answer.
        assert report.overall_verdict is None
        assert set(report.undecided()) == {
            "detection",
            "tracking",
            "downscaling",
            "transition",
        }

    def test_the_reason_names_the_keys_to_fill_in(self):
        report = evaluate_validation(
            {"downscaling": MetricTable(metrics={"peak_preservation": 0.97})}
        )
        reason = report.stage("downscaling").reason
        assert "configs/validation.yaml" in reason
        assert "gates.per_stage.downscaling.pass" in reason
        assert "will not guess" in reason

    def test_a_missing_table_is_suppressed_not_passed(self):
        """An absent stage is not a passing stage."""
        report = evaluate_validation({"downscaling": MetricTable(metrics={})})
        assert report.stage("detection").verdict is GateVerdict.SUPPRESS
        assert "no output" in report.stage("detection").reason

    def test_an_undeclared_table_produces_no_verdict_and_says_why(self):
        report = evaluate_validation({"gnn": MetricTable(metrics={"loss": 0.1})})
        assert report.stage("gnn") is None
        assert any("gnn" in note and "no gates.primary_metric" in note for note in report.notes)


# ---------------------------------------------------------------------------
# The gate once thresholds are supplied
# ---------------------------------------------------------------------------


def _configured(**stages) -> dict:
    return {
        "gates": {
            "primary_metric": {name: spec["metric"] for name, spec in stages.items()},
            "per_stage": {
                name: {key: value for key, value in spec.items() if key != "metric"}
                for name, spec in stages.items()
            },
        },
        "reporting": {"output_dir": "./data/processed/validation"},
    }


class TestGateDecisions:
    def test_higher_is_better_pass(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        assert decide_stage(gate, 0.94).verdict is GateVerdict.PASS

    def test_higher_is_better_degrade(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        verdict = decide_stage(gate, 0.80)
        assert verdict.verdict is GateVerdict.DEGRADE
        assert "below pass" in verdict.reason

    def test_higher_is_better_suppress(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        assert decide_stage(gate, 0.62).verdict is GateVerdict.SUPPRESS

    def test_lower_is_better_runs_the_other_way(self):
        """brier_score and f1 point in opposite directions; the gate must know."""
        gate = StageGate("transition", "brier_score", pass_at=0.10, degrade_at=0.20)
        assert decide_stage(gate, 0.05).verdict is GateVerdict.PASS
        assert decide_stage(gate, 0.15).verdict is GateVerdict.DEGRADE
        assert decide_stage(gate, 0.30).verdict is GateVerdict.SUPPRESS

    def test_boundary_values_pass(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        assert decide_stage(gate, 0.90).verdict is GateVerdict.PASS
        assert decide_stage(gate, 0.70).verdict is GateVerdict.DEGRADE

    def test_inconsistent_thresholds_are_refused(self):
        """A pass bar below the degrade bar cannot produce the intended ordering."""
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.70, degrade_at=0.90)
        verdict = decide_stage(gate, 0.8)
        assert verdict.verdict is None
        assert "inconsistent" in verdict.reason

    def test_unknown_metric_direction_is_refused_not_assumed(self):
        gate = StageGate("mystery", "not_a_metric", pass_at=0.9, degrade_at=0.7)
        verdict = decide_stage(gate, 0.95)
        assert verdict.verdict is None
        assert "direction" in verdict.reason

    def test_uncomputable_metric_is_undecided_not_a_pass(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        verdict = decide_stage(gate, None)
        assert verdict.verdict is None
        assert "not a pass" in verdict.reason

    def test_stage_that_did_not_run_is_suppressed(self):
        gate = StageGate("downscaling", "peak_preservation", pass_at=0.90, degrade_at=0.70)
        assert decide_stage(gate, None, produced_output=False).verdict is GateVerdict.SUPPRESS

    def test_direction_table_covers_every_gate_metric_in_the_config(self):
        config = GateConfig.from_config()
        unknown = [
            gate.primary_metric
            for gate in config.stages.values()
            if metric_is_higher_better(gate.primary_metric) is None
        ]
        assert not unknown, f"no direction known for {unknown}"

    def test_config_parses_per_stage_overrides(self):
        config = GateConfig.from_config(
            _configured(
                downscaling={
                    "metric": "peak_preservation",
                    "pass": 0.90,
                    "degrade": 0.70,
                    "justification": "baseline reaches 0.91",
                }
            )
        )
        gate = config.stage("downscaling")
        assert gate.configured
        assert gate.justification == "baseline reaches 0.91"

    def test_shared_thresholds_apply_when_no_override_exists(self):
        shared = {
            "gates": {
                "primary_metric": {"downscaling": "peak_preservation"},
                "pass": 0.95,
                "degrade": 0.80,
            }
        }
        gate = GateConfig.from_config(shared).stage("downscaling")
        assert gate.pass_at == 0.95
        assert gate.degrade_at == 0.80

    def test_zero_is_accepted_as_a_real_threshold(self):
        """0.0 is a value, not a missing one — the falsy-value trap."""
        config = GateConfig.from_config(
            {
                "gates": {
                    "primary_metric": {"transition": "false_alarm_rate"},
                    "per_stage": {"transition": {"pass": 0.0, "degrade": 0.05}},
                }
            }
        )
        gate = config.stage("transition")
        assert gate.pass_at == 0.0
        assert gate.configured


class TestOverallVerdict:
    def _tables(self, peak: float) -> dict[str, MetricTable]:
        return {"downscaling": MetricTable(metrics={"peak_preservation": peak})}

    def test_overall_is_the_worst_stage(self):
        report = evaluate_validation(
            self._tables(0.75),
            config=_configured(
                downscaling={"metric": "peak_preservation", "pass": 0.90, "degrade": 0.70}
            ),
        )
        assert report.overall_verdict is GateVerdict.DEGRADE

    def test_overall_is_none_while_any_gated_stage_is_undecided(self):
        """Reporting the best decided stage would hide exactly the gaps the gate exists for.

        ``detection`` is declared in the config but has no thresholds, so it is
        undecided even though it supplied a table — and that must block the overall
        verdict despite downscaling passing outright.
        """
        config = _configured(
            downscaling={"metric": "peak_preservation", "pass": 0.90, "degrade": 0.70}
        )
        config["gates"]["primary_metric"]["detection"] = "f1"
        report = evaluate_validation(
            {**self._tables(0.99), "detection": MetricTable(metrics={"f1": 0.91})},
            config=config,
        )
        assert report.stage("downscaling").verdict is GateVerdict.PASS
        assert report.stage("detection").verdict is None
        assert report.overall_verdict is None

    def test_suppress_dominates_pass(self):
        report = evaluate_validation(
            {
                "downscaling": MetricTable(metrics={"peak_preservation": 0.99}),
                "detection": MetricTable(metrics={"f1": 0.10}),
            },
            config=_configured(
                downscaling={"metric": "peak_preservation", "pass": 0.90, "degrade": 0.70},
                detection={"metric": "f1", "pass": 0.80, "degrade": 0.60},
            ),
        )
        assert report.stage("downscaling").verdict is GateVerdict.PASS
        assert report.overall_verdict is GateVerdict.SUPPRESS

    def test_base_rate_is_surfaced_in_the_notes(self):
        report = evaluate_validation(
            {"transition": MetricTable(metrics={"brier_score": 0.04, "base_rate": 0.07})}
        )
        assert any("base rate 7.0%" in note for note in report.notes)

    def test_a_transition_score_without_a_base_rate_is_called_out(self):
        report = evaluate_validation({"transition": MetricTable(metrics={"brier_score": 0.04})})
        assert any("cannot be interpreted" in note for note in report.notes)


class TestReportArtefact:
    def _configured_report(self):
        """Only downscaling gated, with thresholds: yields a decidable verdict."""
        return evaluate_validation(
            {"downscaling": MetricTable(metrics={"peak_preservation": 0.75}, units="mm/24h")},
            config=_configured(
                downscaling={"metric": "peak_preservation", "pass": 0.90, "degrade": 0.70}
            ),
            subject="THR-2026-0001",
        )

    def _shipped_config_report(self):
        """The real config, whose thresholds are null: nothing can be decided."""
        return evaluate_validation(
            {"downscaling": MetricTable(metrics={"peak_preservation": 0.97})},
            subject="THR-2026-0001",
        )

    def test_saves_json_and_markdown(self, tmp_path):
        paths = self._configured_report().save(tmp_path / "validation")
        assert paths["json"].exists()
        assert paths["markdown"].exists()

    def test_json_distinguishes_undecided_from_suppressed(self, tmp_path):
        """These two null-ish outcomes mean different things and must not be conflated.

        ``downscaling`` supplied a metric but has no thresholds, so it is genuinely
        **undecided**. The other three produced no table at all, which is a
        definite SUPPRESS — an absent stage is not an unknown one.
        """
        paths = self._shipped_config_report().save(tmp_path / "validation")
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["_meta"]["note"].startswith("Null means not computable")
        assert payload["overall_verdict"] is None
        assert payload["undecided_stages"] == ["downscaling"]
        verdicts = {row["stage"]: row["verdict"] for row in payload["stages"]}
        assert verdicts["detection"] == "SUPPRESS"
        assert verdicts["tracking"] == "SUPPRESS"
        assert verdicts["transition"] == "SUPPRESS"
        assert verdicts["downscaling"] is None

    def test_markdown_says_not_decided_against_the_shipped_config(self, tmp_path):
        markdown = self._shipped_config_report().save(tmp_path / "validation")[
            "markdown"
        ].read_text(encoding="utf-8")
        assert "**Overall verdict: NOT DECIDED**" in markdown

    def test_markdown_states_a_decided_verdict_in_words(self, tmp_path):
        markdown = self._configured_report().save(tmp_path / "validation")[
            "markdown"
        ].read_text(encoding="utf-8")
        assert "**Overall verdict: DEGRADE**" in markdown

    def test_markdown_includes_the_metric_tables(self, tmp_path):
        markdown = self._configured_report().save(tmp_path / "validation")[
            "markdown"
        ].read_text(encoding="utf-8")
        assert "`peak_preservation`" in markdown

    def test_default_output_dir_matches_the_config(self):
        expected = REPO_ROOT / "data" / "processed" / "validation"
        assert default_output_dir() == expected
