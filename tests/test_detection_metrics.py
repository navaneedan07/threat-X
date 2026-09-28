"""Tests for the anomaly-detection metrics.

Run:  pytest tests/test_detection_metrics.py -v

The load-bearing property: a metric whose denominator is zero must be ``None``,
never ``1.0``. A detector that flags nothing has no precision to report, and a
reference with no extremes has no recall to report — printing ``1.0`` in either
case is how a useless detector reads as perfect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.validation import detection_metrics as D

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[cell == "1" for cell in row] for row in rows], dtype=bool)


class TestConfusionCounts:
    def test_counts_a_known_case(self):
        predicted = _mask(["1100", "1000"])
        reference = _mask(["1100", "0100"])
        counts = D.confusion_counts(predicted, reference)
        assert counts.true_positive == 2
        assert counts.false_positive == 1
        assert counts.false_negative == 1
        assert counts.true_negative == 4
        assert counts.observed_positive == 3
        assert counts.predicted_positive == 3

    def test_identical_masks_have_no_errors(self):
        mask = _mask(["1100", "0000"])
        counts = D.confusion_counts(mask, mask)
        assert counts.false_positive == 0
        assert counts.false_negative == 0

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="must share a grid"):
            D.confusion_counts(_mask(["10", "10"]), _mask(["100", "000", "000"]))

    def test_non_boolean_values_are_refused(self):
        with pytest.raises(ValueError, match="boolean or 0/1"):
            D.confusion_counts(np.array([[2.0]]), np.array([[1.0]]))


class TestIndividualMetrics:
    def test_precision_recall_f1_on_a_known_case(self):
        counts = D.confusion_counts(_mask(["110", "100"]), _mask(["110", "010"]))
        assert D.precision(counts) == pytest.approx(2 / 3)
        assert D.recall(counts) == pytest.approx(2 / 3)
        assert D.f1(counts) == pytest.approx(2 / 3)

    def test_false_alarm_rate_is_the_complement_of_precision(self):
        counts = D.confusion_counts(_mask(["111", "000"]), _mask(["100", "000"]))
        assert D.precision(counts) == pytest.approx(1 / 3)
        assert D.false_alarm_rate(counts) == pytest.approx(2 / 3)

    def test_miss_rate_is_the_complement_of_recall(self):
        counts = D.confusion_counts(_mask(["100", "000"]), _mask(["111", "000"]))
        assert D.recall(counts) == pytest.approx(1 / 3)
        assert D.miss_rate(counts) == pytest.approx(2 / 3)

    def test_a_detector_that_flags_nothing_has_no_precision(self):
        """Not 1.0 — a detector that never fires has no precision to report."""
        counts = D.confusion_counts(_mask(["000", "000"]), _mask(["100", "000"]))
        assert D.precision(counts) is None
        assert D.false_alarm_rate(counts) is None
        assert D.f1(counts) is None
        assert D.recall(counts) == pytest.approx(0.0)

    def test_a_reference_with_no_extremes_has_no_recall(self):
        counts = D.confusion_counts(_mask(["100", "000"]), _mask(["000", "000"]))
        assert D.recall(counts) is None
        assert D.miss_rate(counts) is None
        assert D.f1(counts) is None
        # The only flagged cell is a false positive against an empty truth.
        assert D.precision(counts) == pytest.approx(0.0)

    def test_perfect_agreement(self):
        mask = _mask(["110", "000"])
        counts = D.confusion_counts(mask, mask)
        assert D.precision(counts) == pytest.approx(1.0)
        assert D.recall(counts) == pytest.approx(1.0)
        assert D.f1(counts) == pytest.approx(1.0)
        assert D.false_alarm_rate(counts) == pytest.approx(0.0)
        assert D.miss_rate(counts) == pytest.approx(0.0)


class TestDetectionTable:
    def test_table_emits_every_config_metric(self):
        """configs/validation.yaml lists the detection metrics; all must be present."""
        import yaml

        declared = yaml.safe_load(
            (REPO_ROOT / "configs" / "validation.yaml").read_text(encoding="utf-8")
        )["detection"]["metrics"]
        table = D.evaluate_detection(_mask(["110", "100"]), _mask(["110", "010"]))
        missing = [name for name in declared if name not in table.metrics]
        assert not missing, f"config declares metrics the table does not produce: {missing}"

    def test_counts_are_carried_in_context(self):
        table = D.evaluate_detection(_mask(["110"]), _mask(["100"]))
        assert table.context["predicted_positive_cells"] == 2
        assert table.context["reference_positive_cells"] == 1
        assert table.context["confusion"]["true_positive"] == 1

    def test_no_flagged_cells_is_explained_in_the_notes(self):
        table = D.evaluate_detection(_mask(["000"]), _mask(["100"]))
        assert table["precision"] is None
        assert "undefined" in table.notes["precision"]
        assert table["f1"] is None

    def test_table_has_no_units(self):
        """Precision is dimensionless; a field unit must not be attached to it."""
        table = D.evaluate_detection(_mask(["110"]), _mask(["110"]))
        assert table.units is None
        assert "mm" not in table.to_markdown()
