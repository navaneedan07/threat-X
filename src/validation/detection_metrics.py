"""Metrics for the extreme-anomaly detection stage.

Precision, recall, F1, false-alarm rate and miss rate, computed from a predicted
anomaly mask against a reference mask.

What "a candidate" means here
-----------------------------
``configs/validation.yaml`` describes precision as *correct candidates / all
candidates*. A candidate is a grid cell flagged as anomalous: this module
evaluates the **cell-level** mask, which is the version that can be computed
without reference region labels. Region-level matching (one predicted blob
against one reference blob, by IoU) is the natural extension, but it needs
hand-labelled reference regions that the project does not have — so it is not
faked here. The cells are the observable, so the cells are what is scored.

Null discipline
---------------
Every metric returns ``None`` when its denominator is zero, and ``None`` renders
as ``—``. Precision with no predicted positives and recall with no reference
positives are both undefined, and reporting ``1.0`` for either would read as a
perfect score for a detector that found nothing.

No thresholds. ``evaluate_detection`` takes the two masks and computes what they
imply; deciding what counts as good is the gate's job
(``src/validation/evaluation.py``), and those thresholds stay ``null`` until real
distributions are measured.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.metrics import MetricTable

__all__ = [
    "ConfusionCounts",
    "confusion_counts",
    "evaluate_detection",
    "false_alarm_rate",
    "f1",
    "miss_rate",
    "precision",
    "recall",
]


def _as_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Coerce an array-like to a boolean mask, requiring a matching shape."""
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError(f"{name}: empty mask")
    if not np.isin(array, (0, 1, True, False)).all():
        raise ValueError(f"{name}: mask must be boolean or 0/1, got values {array!r}")
    return array.astype(bool)


def _pair(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = _as_mask(predicted, "predicted")
    right = _as_mask(reference, "reference")
    if left.shape != right.shape:
        raise ValueError(
            f"predicted {left.shape} and reference {right.shape} masks must share a "
            "grid; regrid before comparing"
        )
    return left, right


class ConfusionCounts:
    """Cell-level confusion counts for one threshold."""

    __slots__ = ("true_positive", "false_positive", "false_negative", "true_negative")

    def __init__(
        self,
        true_positive: int,
        false_positive: int,
        false_negative: int,
        true_negative: int,
    ) -> None:
        self.true_positive = int(true_positive)
        self.false_positive = int(false_positive)
        self.false_negative = int(false_negative)
        self.true_negative = int(true_negative)

    @property
    def observed_positive(self) -> int:
        """Cells the reference flagged (TP + FN)."""
        return self.true_positive + self.false_negative

    @property
    def predicted_positive(self) -> int:
        """Cells the prediction flagged (TP + FP)."""
        return self.true_positive + self.false_positive

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


def confusion_counts(predicted: np.ndarray, reference: np.ndarray) -> ConfusionCounts:
    """Count TP / FP / FN / TN over the two boolean masks."""
    left, right = _pair(predicted, reference)
    return ConfusionCounts(
        true_positive=int(np.sum(left & right)),
        false_positive=int(np.sum(left & ~right)),
        false_negative=int(np.sum(~left & right)),
        true_negative=int(np.sum(~left & ~right)),
    )


def precision(counts: ConfusionCounts) -> float | None:
    """``TP / (TP + FP)`` — share of flagged cells that were genuinely extreme.

    ``None`` when the prediction flagged nothing: a detector that never fires has
    no precision to report, and ``1.0`` there would be a lie.
    """
    denominator = counts.predicted_positive
    return None if denominator == 0 else float(counts.true_positive / denominator)


def recall(counts: ConfusionCounts) -> float | None:
    """``TP / (TP + FN)`` — share of true extremes that were flagged.

    ``None`` when the reference has no extremes: recall against an empty truth is
    undefined, not perfect.
    """
    denominator = counts.observed_positive
    return None if denominator == 0 else float(counts.true_positive / denominator)


def f1(counts: ConfusionCounts) -> float | None:
    """Harmonic mean of precision and recall, or ``None`` if either is undefined."""
    precision_value = precision(counts)
    recall_value = recall(counts)
    if precision_value is None or recall_value is None:
        return None
    total = precision_value + recall_value
    return None if total == 0 else float(2.0 * precision_value * recall_value / total)


def false_alarm_rate(counts: ConfusionCounts) -> float | None:
    """``FP / (TP + FP)`` — share of flagged cells that were false alarms.

    The complement of precision, computed from the same counts so the two can
    never disagree; ``None`` when nothing was flagged.
    """
    value = precision(counts)
    return None if value is None else float(1.0 - value)


def miss_rate(counts: ConfusionCounts) -> float | None:
    """``FN / (TP + FN)`` — share of true extremes that were missed."""
    value = recall(counts)
    return None if value is None else float(1.0 - value)


def evaluate_detection(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    context: dict[str, Any] | None = None,
) -> MetricTable:
    """Full detection metric table for one threshold.

    Metric names match ``configs/validation.yaml`` → ``detection.metrics``
    (``precision``, ``recall``, ``f1``, ``false_alarm_rate``, ``miss_rate``).
    The confusion counts are carried in ``context`` so a precision of ``None`` can
    always be traced to the reason rather than read as a missing computation.
    """
    counts = confusion_counts(predicted, reference)
    table = MetricTable(
        units=None,
        context={
            "cells": int(counts.true_positive + counts.false_positive
                         + counts.false_negative + counts.true_negative),
            "predicted_positive_cells": counts.predicted_positive,
            "reference_positive_cells": counts.observed_positive,
            "confusion": counts.to_dict(),
            **(context or {}),
        },
    )

    table.metrics["precision"] = precision(counts)
    table.metrics["recall"] = recall(counts)
    table.metrics["f1"] = f1(counts)
    table.metrics["false_alarm_rate"] = false_alarm_rate(counts)
    table.metrics["miss_rate"] = miss_rate(counts)

    if counts.predicted_positive == 0:
        table.notes["precision"] = "no cells were flagged, so precision is undefined"
        table.notes["false_alarm_rate"] = "no cells were flagged"
    if counts.observed_positive == 0:
        table.notes["recall"] = "the reference has no extreme cells, so recall is undefined"
        table.notes["miss_rate"] = "the reference has no extreme cells"
    if counts.predicted_positive == 0 or counts.observed_positive == 0:
        table.notes["f1"] = "undefined while precision or recall is undefined"
    return table
